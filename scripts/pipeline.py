"""
Batch resume ingestion CLI.

Reads every supported resume in an input directory (default: uploads/),
extracts it via the shared extraction module, and writes the results to
data/candidates.json for scripts/migrate.py to load.

Resumable: candidates.json is re-read on startup and any file already recorded
there is skipped, then the file is rewritten after every successful extraction.
A run that dies halfway through 200 resumes -- rate limit, network drop, Ctrl-C
-- keeps everything it had already paid Gemini for and picks up where it left
off on the next invocation.

Usage:
    python scripts/pipeline.py [input_dir]
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

# config.py and extraction.py live at the repo root; running this file directly
# only puts scripts/ on sys.path, so add the root explicitly.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config import CANDIDATES_JSON, UPLOADS_DIR, setup_logging  # noqa: E402
from extraction import (  # noqa: E402
    SUPPORTED_EXTENSIONS,
    ExtractionError,
    extract_candidate_with_backoff,
)

logger = logging.getLogger(__name__)

# Stay inside the Gemini free-tier request budget.
THROTTLE_SECONDS = 6


def load_existing(path: Path) -> list:
    """Load prior results, tolerating a missing or truncated file."""
    if not path.exists():
        return []
    try:
        with path.open(encoding="utf-8") as handle:
            records = json.load(handle)
    except json.JSONDecodeError as exc:
        logger.warning("%s is not valid JSON (%s). Starting fresh.", path, exc)
        return []

    if not isinstance(records, list):
        logger.warning("%s is not a JSON list. Starting fresh.", path)
        return []
    return records


def save_atomic(records: list, path: Path) -> None:
    """
    Write results via a temp file + rename.

    Writing in place would leave a truncated, unparseable candidates.json if the
    process died mid-write -- which for a resumable job means losing the whole
    run. os.replace is atomic, so the file is either the old version or the new
    one, never half of each.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    tmp_path.replace(path)


def process_all_resumes(input_dir: Path) -> None:
    """
    Extract every not-yet-processed resume in input_dir into candidates.json.

    Files are matched against prior results by filename, so re-running after an
    interruption only pays for what is genuinely missing.
    """
    input_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not files:
        logger.info("No supported resumes found in %s.", input_dir)
        logger.info("Supported formats: %s", ", ".join(sorted(SUPPORTED_EXTENSIONS)))
        return

    records = load_existing(CANDIDATES_JSON)
    already_done = {
        record.get("source_file")
        for record in records
        if isinstance(record, dict)
    }

    pending = [path for path in files if path.name not in already_done]
    skipped = len(files) - len(pending)

    logger.info("Found %d resume(s) in %s.", len(files), input_dir)
    if skipped:
        logger.info("Skipping %d already present in %s.", skipped, CANDIDATES_JSON.name)
    if not pending:
        logger.info("Nothing left to do.")
        return

    logger.info("Processing %d new file(s).", len(pending))

    failures: list[str] = []

    for index, path in enumerate(pending):
        logger.info("[%d/%d] Processing: %s", index + 1, len(pending), path.name)

        try:
            candidate = extract_candidate_with_backoff(path.name, path.read_bytes())
        except ExtractionError as exc:
            # Deliberately not written to candidates.json. An unrecorded file is
            # picked up again by the next run, which is the behaviour we want for
            # a transient failure and harmless for a permanent one.
            logger.error("Skipping %s: %s", path.name, exc.message)
            failures.append(path.name)
            if index < len(pending) - 1:
                time.sleep(THROTTLE_SECONDS)
            continue

        records.append(candidate)

        # Flush after every file so an interrupted run loses at most one record.
        save_atomic(records, CANDIDATES_JSON)

        logger.info(
            "Saved: name=%s skills=%s",
            candidate.get("name"), candidate.get("skills"),
        )

        if index < len(pending) - 1:
            time.sleep(THROTTLE_SECONDS)

    logger.info("Batch complete. %d total record(s) in %s", len(records), CANDIDATES_JSON)
    if failures:
        logger.warning(
            "%d file(s) failed and were not recorded; rerun to retry: %s",
            len(failures), ", ".join(failures),
        )


def main() -> None:
    """CLI entry point. Optional argv[1] overrides the input directory."""
    setup_logging()
    input_dir = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else UPLOADS_DIR
    process_all_resumes(input_dir)


if __name__ == "__main__":
    main()
