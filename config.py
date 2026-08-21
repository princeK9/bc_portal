"""
Shared configuration: environment loading, database connection, paths, logging.

Every module reads settings from here instead of calling os.getenv itself, so
there is exactly one place to look when a setting is wrong and one place to
change when a setting moves. Previously DB_CONFIG was declared identically in
three files, which is the same duplication class that let the embedding format
drift apart in Level 2.

Importing this module loads .env from the repo root regardless of the current
working directory -- the CLI scripts under scripts/ are routinely run from
elsewhere, and a cwd-relative .env silently yields None for every credential.
"""

import logging
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent

load_dotenv(REPO_ROOT / ".env")

# --- Database ---------------------------------------------------------------

# Supabase Transaction Pooler connection.
DB_CONFIG = {
    "dbname": os.getenv("DB_NAME", "postgres"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT", "6543"),
}

# --- Gemini -----------------------------------------------------------------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# Overridable so a model change is a config change, not a code edit.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")

# --- Redis / Celery ---------------------------------------------------------

# Redis has two distinct jobs here, on two logical databases so a cache flush
# can never drop queued work:
#   db 0 -> Celery broker + result backend (durable, must not be flushed)
#   db 1 -> short-TTL read cache in front of /search (safe to flush any time)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
REDIS_CACHE_URL = os.getenv("REDIS_CACHE_URL", "redis://localhost:6379/1")

CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", REDIS_URL)
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", REDIS_URL)

# See celery_app.py for the arithmetic behind this number -- it is derived from
# the Gemini rate-limit budget, not picked as a round default.
WORKER_CONCURRENCY = int(os.getenv("WORKER_CONCURRENCY", "4"))
# Celery rate-limit string. The real ceiling on throughput; concurrency only
# determines whether we can reach it.
#
# 14/m, not the 90/m this previously held. The free-tier quota for
# gemini-3.5-flash-lite is 15 requests/minute, confirmed by a live 429 during a
# 46-file load test:
#   "Quota exceeded for metric: generate_content_free_tier_requests,
#    limit: 15, model: gemini-3.5-flash-lite"
# The earlier 90/m came from an unverified 35-104/min assumption and admitted
# roughly 5x the real allowance; 8 of 46 files exhausted their retries and
# failed permanently. 14 leaves one request of headroom for clock skew between
# our limiter and Google's window accounting.
GEMINI_RATE_LIMIT = os.getenv("GEMINI_RATE_LIMIT", "14/m")

# Hard ceiling on a single extraction. One hung Gemini call must not hold a
# worker slot open forever. soft < hard so the task can raise SoftTimeLimit and
# clean up before SIGKILL.
TASK_SOFT_TIME_LIMIT = int(os.getenv("TASK_SOFT_TIME_LIMIT", "120"))
TASK_TIME_LIMIT = int(os.getenv("TASK_TIME_LIMIT", "150"))
TASK_MAX_RETRIES = int(os.getenv("TASK_MAX_RETRIES", "5"))

# Redis is not a real broker and has no server-side ack. Kombu emulates acking
# by holding delivered messages in an `unacked` hash and only returning them to
# the queue once this timeout elapses. task_reject_on_worker_lost cannot help
# there -- there is no broker to reject the message back to -- so THIS value,
# not that flag, is the actual recovery time for a SIGKILLed worker's in-flight
# tasks. Observed directly: killing a worker with the kombu default left four
# tasks stranded in `unacked` and the job stuck at 19/23 indefinitely.
#
# Must exceed the longest a *healthy* task can legitimately hold a message, or
# a slow-but-alive task gets redelivered and runs twice:
#   task_time_limit           = 150s
#   longest retry backoff     = MAX_RETRY_COUNTDOWN 600s x 1.2 jitter = 720s
# 900s clears both with margin. Lowering it speeds up recovery from a dead
# worker at the cost of duplicating long-backoff retries.
BROKER_VISIBILITY_TIMEOUT = int(os.getenv("BROKER_VISIBILITY_TIMEOUT", "900"))

# --- CORS -------------------------------------------------------------------

# Empty by default, and that is the correct default now that FastAPI serves the
# frontend itself: a same-origin request never triggers CORS at all, so no
# middleware is needed for the bundled UI. This exists only for the case where
# someone hosts the page separately, and then the origins must be named
# explicitly rather than wildcarded.
#
# Replaces allow_origins=["*"] paired with allow_credentials=True, which was an
# invalid combination -- browsers reject a wildcard origin on a credentialed
# request, so the credentials flag was dead config that merely looked permissive.
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "").split(",")
    if origin.strip()
]

# --- Search cache -----------------------------------------------------------

# Deliberately short. Newly uploaded candidates should become searchable in
# about a minute; a long TTL would make the upload flow look broken.
SEARCH_CACHE_TTL = int(os.getenv("SEARCH_CACHE_TTL", "60"))
SEARCH_CACHE_ENABLED = os.getenv("SEARCH_CACHE_ENABLED", "true").lower() == "true"

# --- Upload limits ----------------------------------------------------------

# Checked before any task is dispatched. Rejecting a 10k-file zip after fanning
# out 10k tasks is far more expensive than rejecting it at the door.
MAX_ZIP_FILES = int(os.getenv("MAX_ZIP_FILES", "500"))
MAX_ZIP_TOTAL_BYTES = int(os.getenv("MAX_ZIP_TOTAL_BYTES", str(500 * 1024 * 1024)))
MAX_SINGLE_FILE_BYTES = int(os.getenv("MAX_SINGLE_FILE_BYTES", str(25 * 1024 * 1024)))
# Guards against a zip bomb: a small archive that expands enormously.
MAX_ZIP_COMPRESSION_RATIO = int(os.getenv("MAX_ZIP_COMPRESSION_RATIO", "100"))

BATCH_TTL_SECONDS = int(os.getenv("BATCH_TTL_SECONDS", str(24 * 3600)))

# --- Paths ------------------------------------------------------------------

FRONTEND_DIR = REPO_ROOT / "frontend"
UPLOADS_DIR = REPO_ROOT / "uploads"
DATA_DIR = REPO_ROOT / "data"
# Where an upload is unpacked to before its tasks run. The API container writes
# here and the worker container reads, so under Docker this must be a shared
# volume -- a task holding a path the worker cannot see fails on every attempt.
SPOOL_DIR = Path(os.getenv("SPOOL_DIR", str(REPO_ROOT / "spool")))
CANDIDATES_JSON = DATA_DIR / "candidates.json"
SAMPLE_DATA_DIR = REPO_ROOT / "sample_data"

# --- Logging ----------------------------------------------------------------

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s | %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"

# These libraries log per-request detail at INFO. Left alone they bury the
# application's own output -- a single model load emits ~30 HTTP lines.
NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "huggingface_hub",
    "sentence_transformers",
    "urllib3",
    "filelock",
)


def setup_logging(level: Optional[str] = None) -> None:
    """
    Configure root logging. Call once from an entry point.

    Library modules must not call this -- they should only do
    `logger = logging.getLogger(__name__)` and let whatever runs them decide
    where output goes. That distinction matters more once Celery workers exist,
    since the worker owns its own logging setup.
    """
    logging.basicConfig(
        level=level or LOG_LEVEL,
        format=LOG_FORMAT,
        datefmt=LOG_DATEFMT,
    )
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
