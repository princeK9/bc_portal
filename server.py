import logging
import shutil
import tempfile
import uuid
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path

import psycopg2
from celery import group
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import jobs
from cache import get_cached, set_cached
from celery_app import app as celery
from config import (
    CORS_ORIGINS,
    DB_CONFIG,
    FRONTEND_DIR,
    MAX_SINGLE_FILE_BYTES,
    MAX_ZIP_COMPRESSION_RATIO,
    MAX_ZIP_FILES,
    MAX_ZIP_TOTAL_BYTES,
    SPOOL_DIR,
    setup_logging,
)
from extraction import SUPPORTED_EXTENSIONS, embed_text, get_embedder
from tasks import process_resume

setup_logging()
logger = logging.getLogger(__name__)

# Streamed to disk in chunks this size rather than read whole.
UPLOAD_CHUNK_BYTES = 1024 * 1024

# Returned to clients in place of raw exception text. Exception detail can
# carry connection strings, table names and credentials, so it is logged
# server-side and never serialized into a response body.
GENERIC_SEARCH_ERROR = "Search failed. Please try again."
GENERIC_UPSTREAM_ERROR = "Search backend is unavailable. Please try again shortly."


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Warm the encoder at boot so the first search request is not the one that
    pays the model-load cost. Import stays cheap regardless, because
    extraction.py loads the model lazily.
    """
    logger.info("Booting up AI Engine...")
    SPOOL_DIR.mkdir(parents=True, exist_ok=True)
    get_embedder()
    logger.info("Embedding model ready.")
    yield


app = FastAPI(title="Blue Collar Portal AI Engine", lifespan=lifespan)

# CORS is now opt-in and empty by default.
#
# The previous config was allow_origins=["*"] with allow_credentials=True, which
# is not merely loose but invalid: browsers refuse a wildcard origin on a
# credentialed request, so the credentials flag never did anything. It looked
# like a permissive-but-working policy and was actually dead config.
#
# Now that this process serves the frontend itself, the page and the API share
# an origin and CORS does not enter into it -- no middleware is required for the
# bundled UI at all. The middleware is only attached when CORS_ORIGINS names
# real origins, for the case where the page is hosted separately. Credentials
# stay off because nothing here uses cookies, sessions or auth headers; the API
# is unauthenticated, and enabling credentials would only widen what a
# cross-origin page could do with it.
if CORS_ORIGINS:
    logger.info("CORS enabled for origins: %s", CORS_ORIGINS)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

# Frontend assets. index.html is currently self-contained (inline CSS and JS),
# but this makes frontend/ the working place to add a stylesheet or image later
# without another server change.
if FRONTEND_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/", include_in_schema=False)
async def serve_frontend():
    """
    Serve the recruiter UI from the API process.

    Declared as an explicit route rather than mounting StaticFiles at "/",
    because a mount at the root matches by prefix and would shadow every API
    path that is not declared before it -- a fragile ordering dependency for no
    benefit when there is exactly one page to serve.

    Serving it here is what makes the frontend same-origin with the API, which
    in turn is what allows CORS to be switched off rather than wildcarded.
    """
    index = FRONTEND_DIR / "index.html"
    if not index.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Frontend is not available in this deployment.",
        )
    return FileResponse(index)


def calibrate_score(raw_sim: float) -> tuple[float, str]:
    """
    Normalizes raw cosine similarity (typically 0.2 - 0.8 range for MiniLM)
    into a human-readable 0-100% score and assigns a confidence level.
    """
    normalized = min(max((raw_sim - 0.2) / 0.6 * 100, 0), 100)

    if normalized >= 70:
        confidence = "High Match"
    elif normalized >= 40:
        confidence = "Moderate Match"
    else:
        confidence = "Low Match"

    return round(normalized, 1), confidence


def _is_candidate_file(name: str) -> bool:
    """Filter ZIP entries down to resumes we can actually parse."""
    if name.startswith("__MACOSX") or name.endswith("/"):
        return False
    leaf = name.split("/")[-1]
    # Skip directory entries and hidden/AppleDouble files.
    if not leaf or leaf.startswith("."):
        return False
    return leaf.lower().endswith(SUPPORTED_EXTENSIONS)


def inspect_archive(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    """
    Validate a zip before any work is dispatched, and return its usable members.

    Checked up front rather than per task: fanning out 10,000 tasks and then
    discovering the archive was junk costs real Gemini spend and floods the
    queue. Raises HTTPException(400) with a specific reason on rejection --
    "too big" and "wrong format" need different fixes from the caller.

    The compression-ratio check is the zip-bomb guard: a few hundred KB of
    archive can expand to gigabytes, and the size check alone reads the
    attacker-supplied header rather than the real expanded size.
    """
    members = [info for info in archive.infolist() if _is_candidate_file(info.filename)]

    if not members:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "No supported resume files found in archive. Supported formats: "
                + ", ".join(sorted(SUPPORTED_EXTENSIONS))
            ),
        )

    if len(members) > MAX_ZIP_FILES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Archive contains {len(members)} files; the limit is {MAX_ZIP_FILES}.",
        )

    total_uncompressed = sum(info.file_size for info in members)
    if total_uncompressed > MAX_ZIP_TOTAL_BYTES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Archive expands to {total_uncompressed} bytes; "
                f"the limit is {MAX_ZIP_TOTAL_BYTES}."
            ),
        )

    for info in members:
        if info.file_size > MAX_SINGLE_FILE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"{info.filename} is {info.file_size} bytes; "
                    f"the per-file limit is {MAX_SINGLE_FILE_BYTES}."
                ),
            )
        # compress_size 0 means a stored (uncompressed) entry, ratio 1.
        if info.compress_size > 0:
            ratio = info.file_size / info.compress_size
            if ratio > MAX_ZIP_COMPRESSION_RATIO:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        f"{info.filename} has a compression ratio of {ratio:.0f}:1, "
                        f"above the {MAX_ZIP_COMPRESSION_RATIO}:1 limit."
                    ),
                )

    return members


def spool_members(archive: zipfile.ZipFile, members: list[zipfile.ZipInfo],
                  batch_dir: Path) -> list[tuple[Path, str]]:
    """
    Copy each member out of the archive onto disk, streamed.

    Never holds a whole file in memory, and never trusts the member's path --
    entries are written under a generated name, so an entry called
    "../../etc/passwd" cannot escape the spool directory.
    """
    batch_dir.mkdir(parents=True, exist_ok=True)
    spooled = []

    for index, info in enumerate(members):
        suffix = Path(info.filename).suffix.lower()
        target = batch_dir / f"{index:05d}{suffix}"
        with archive.open(info) as source, target.open("wb") as sink:
            shutil.copyfileobj(source, sink, length=UPLOAD_CHUNK_BYTES)
        spooled.append((target, info.filename))

    return spooled


@app.get("/search")
def search_candidates(query: str, limit: int = 5):
    """
    Semantic candidate search, cached for a short TTL.

    The query is embedded with the same MiniLM model and text conventions used
    to embed candidates, then ranked by pgvector cosine distance (`<=>`).
    Raw similarity is returned alongside the calibrated score so the ranking
    stays auditable rather than being a black-box percentage.
    """
    cached = get_cached(query, limit)
    if cached is not None:
        return {**cached, "cached": True}

    conn = None
    cursor = None
    try:
        # 1. Convert search query into vector representation
        query_vector = embed_text(query)

        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # 2. Query pgvector using Cosine Distance (<=> operator)
        #
        # The vector is bound once in a CTE and referenced twice, rather than
        # being passed as two parameters. That is a transport fix, not a query
        # -planning one: a 384-dim vector serializes to ~8,400 characters, and
        # sending it twice measured 47ms against 4.6ms for once -- ten times the
        # cost for twice the payload. Server-side execution was already ~1ms
        # either way, so nearly all of the old latency was moving the parameter.
        #
        # The CTE does not defeat the HNSW index: EXPLAIN confirms
        # "Index Scan using candidates_embedding_hnsw_idx" for this form, and
        # results were verified byte-identical to the two-parameter version
        # across 21 query/limit combinations including exact float equality on
        # raw_similarity.
        sql = """
            WITH q AS (SELECT %s::vector AS v)
            SELECT c.id, c.name, c.raw_title, c.experience_years, c.skills,
                   1 - (c.embedding <=> q.v) AS raw_similarity,
                   c.phone, c.email
            FROM candidates c, q
            WHERE c.name IS NOT NULL
            ORDER BY c.embedding <=> q.v
            LIMIT %s;
        """

        cursor.execute(sql, (str(query_vector), limit))
        results = cursor.fetchall()

        # 3. Format and calibrate output
        matches = []
        for row in results:
            raw_sim = float(row[5])
            calibrated_score, confidence_tier = calibrate_score(raw_sim)

            matches.append({
                "id": row[0],
                "name": row[1],
                "title": row[2],
                "experience_years": row[3],
                "skills": row[4],
                "raw_similarity": round(raw_sim, 3),
                "match_score": calibrated_score,
                "confidence_tier": confidence_tier,
                "phone": row[6],
                "email": row[7],
            })

        payload = {
            "query": query,
            "total_evaluated": len(results),
            "results": matches,
        }
        set_cached(query, limit, payload)
        return {**payload, "cached": False}

    except psycopg2.OperationalError:
        # Could not reach the database at all -- the service is degraded rather
        # than the request being wrong, so 503 with a Retry-After-ish message.
        logger.exception("Database unreachable during search for query=%r", query)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=GENERIC_UPSTREAM_ERROR,
        )
    except Exception:
        logger.exception("Search failed for query=%r", query)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=GENERIC_SEARCH_ERROR,
        )
    finally:
        if cursor is not None:
            cursor.close()
        if conn is not None:
            conn.close()


@app.post("/upload-resumes/")
async def upload_resumes(file: UploadFile = File(...)):
    """
    Accept a ZIP of resumes and dispatch one Celery task per file.

    Returns immediately with a job id; extraction happens on the workers. The
    previous version did the whole batch inline, which meant a 200-file upload
    held an HTTP connection open for roughly an hour and lost everything if the
    client disconnected.

    The archive is streamed to disk and unpacked to disk, never buffered whole
    in RAM -- a 500MB zip previously became 500MB of resident memory in the API
    process, plus another copy per file being read.
    """
    batch_id = uuid.uuid4().hex
    batch_dir = SPOOL_DIR / batch_id

    # Created here as well as at startup: the directory lives on a mounted
    # volume, and relying solely on boot-time creation means a remount between
    # restarts turns every upload into a 500.
    SPOOL_DIR.mkdir(parents=True, exist_ok=True)

    # Stream the upload to a temp file rather than await file.read().
    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip", dir=str(SPOOL_DIR)) as tmp:
        tmp_path = Path(tmp.name)
        while chunk := await file.read(UPLOAD_CHUNK_BYTES):
            tmp.write(chunk)

    try:
        try:
            archive = zipfile.ZipFile(tmp_path)
        except zipfile.BadZipFile:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid file format. Please upload a valid .zip archive.",
            )

        with archive:
            members = inspect_archive(archive)
            spooled = spool_members(archive, members, batch_dir)
    except Exception:
        # Nothing was dispatched, so leave no spool directory behind.
        shutil.rmtree(batch_dir, ignore_errors=True)
        raise
    finally:
        tmp_path.unlink(missing_ok=True)

    # One task per file, dispatched as a group so they run in parallel and fail
    # independently.
    job = group(
        process_resume.s(str(path), original_name, batch_id)
        for path, original_name in spooled
    ).apply_async()
    job.save()

    manifest = [
        {"task_id": result.id, "file": original_name}
        for result, (_, original_name) in zip(job.results, spooled)
    ]
    jobs.save_manifest(batch_id, manifest)

    logger.info("Job %s dispatched with %d task(s).", batch_id, len(manifest))
    return {
        "batch_id": batch_id,
        "queued": len(manifest),
        "files": [entry["file"] for entry in manifest],
        "status_url": f"/status/{batch_id}",
    }


@app.get("/status/{batch_id}")
def job_status(batch_id: str):
    """
    Per-file progress for an upload batch.

    Reports done/failed/pending counts and the state of each individual file,
    rather than a single flag. "Job failed" is not actionable when 197 of 200
    files succeeded; the caller needs to know which three to resubmit.
    """
    manifest = jobs.load_manifest(batch_id)
    if manifest is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Unknown or expired job id.",
        )
    return jobs.summarize(celery, manifest)


@app.get("/health")
async def health_check():
    """Liveness probe. Does not touch the database or the model on purpose,
    so it stays fast and cannot fail for reasons unrelated to the process."""
    return {"status": "online"}
