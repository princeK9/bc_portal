"""
One Celery task per resume file.

Per-file rather than per-batch, deliberately. A batch task means one bad file
fails the whole upload, a retry re-pays Gemini for every file already done, and
a 200-file zip occupies one worker slot for an hour while the others idle.
Per-file tasks fail independently, retry independently, and fan out across
every available slot.
"""

from __future__ import annotations

import logging
import os
import random
import shutil
from pathlib import Path

from celery.exceptions import SoftTimeLimitExceeded

import jobs
from cache import flush_search_cache
from celery_app import app
from config import SPOOL_DIR, TASK_MAX_RETRIES
from db import connection, upsert_candidate
from extraction import (
    ExtractionError,
    PermanentExtractionError,
    TransientExtractionError,
    extract_candidate,
)

logger = logging.getLogger(__name__)

# Backoff ceiling. Beyond this, waiting longer does not meaningfully improve the
# odds and just holds the job open.
MAX_RETRY_COUNTDOWN = 600


def _retry_countdown(retries: int) -> int:
    """
    Exponential backoff with jitter.

    Jitter matters here specifically: a rate-limit response tends to hit every
    in-flight task at once, and without it all of them would retry in the same
    second and trip the limit again.
    """
    base = min((2 ** retries) + 2, MAX_RETRY_COUNTDOWN)
    return int(base * random.uniform(0.8, 1.2))


def _discard(spool_path: str) -> None:
    """Remove the spooled upload. Never raises -- cleanup must not mask a result."""
    try:
        os.unlink(spool_path)
    except FileNotFoundError:
        pass
    except OSError:
        logger.warning("Could not remove spool file %s", spool_path, exc_info=True)


def _finalize(spool_path: str, batch_id: str) -> None:
    """
    Terminal-outcome bookkeeping: drop the file, and if this was the last file
    in the batch, drop the batch directory and invalidate the search cache.

    Must be called on every outcome a file cannot come back from -- success,
    permanent failure, and exhausted retries alike. Calling it on a retry would
    delete the file the retry needs and count the batch finished early.

    Never raises. A batch must not fail because its cleanup did.
    """
    _discard(spool_path)

    try:
        finished, total = jobs.mark_file_finished(batch_id)
    except Exception:
        logger.warning("Could not update finished counter for batch %s", batch_id,
                       exc_info=True)
        return

    if not total or finished < total:
        return

    # Last file in the batch. The corpus has changed, so every cached search
    # response is now describing a database that no longer exists.
    flush_search_cache()

    # The per-file spool files are already gone; this removes the now-empty
    # batch directory, which otherwise accumulates one entry per upload forever.
    batch_dir = Path(spool_path).parent
    try:
        if batch_dir.is_dir() and batch_dir != SPOOL_DIR:
            shutil.rmtree(batch_dir, ignore_errors=True)
            logger.info("Batch %s complete: spool directory removed.", batch_id)
    except OSError:
        logger.warning("Could not remove spool directory %s", batch_dir, exc_info=True)


def _retry_or_give_up(self, exc, spool_path: str, filename: str, batch_id: str,
                      reason: str) -> None:
    """
    Schedule another attempt, or finalize if the retry budget is spent.

    The exhausted case is the one that matters. Previously it fell straight
    through self.retry() without any cleanup, so a file that used up all its
    retries left its spool file behind forever -- the 8 rate-limited files in an
    earlier 46-file load test would each have orphaned one. Checking the budget
    here rather than letting self.retry() raise is what makes that path
    reachable for cleanup at all.
    """
    if self.request.retries >= TASK_MAX_RETRIES:
        _finalize(spool_path, batch_id)
        logger.error(
            "Giving up on %s after %d attempts (%s).",
            filename, TASK_MAX_RETRIES + 1, reason,
        )
        raise exc

    countdown = _retry_countdown(self.request.retries)
    logger.warning(
        "Retrying %s in %ss (attempt %d/%d): %s",
        filename, countdown, self.request.retries + 1, TASK_MAX_RETRIES + 1, reason,
    )
    raise self.retry(exc=exc, countdown=countdown)


@app.task(
    bind=True,
    name="tasks.process_resume",
    max_retries=TASK_MAX_RETRIES,
    # Inherited from celery_app for clarity at the call site; both matter for
    # redelivery when a worker is killed mid-task.
    acks_late=True,
    reject_on_worker_lost=True,
)
def process_resume(self, spool_path: str, filename: str, batch_id: str) -> dict:
    """
    Extract one resume and write it to the database.

    Retries are scheduled with self.retry(countdown=...), never time.sleep().
    Sleeping inside the task would hold a worker slot open for the entire
    backoff -- with concurrency 4 and a 60s backoff, four rate-limited files
    would idle the whole worker for a minute. self.retry re-queues the message
    and frees the slot immediately.

    Returns a small JSON-serializable summary; the extracted record itself is
    not returned, because a 384-float embedding per file would bloat the result
    backend for no benefit.
    """
    path = Path(spool_path)

    try:
        if not path.exists():
            # The spool file is removed on success and on permanent failure. If
            # it is gone on a redelivery, the previous attempt already finished
            # and the upsert has the row; re-running would be a no-op at best.
            logger.warning(
                "Spool file %s for %s is gone; treating as already processed.",
                spool_path, filename,
            )
            return {"file": filename, "status": "skipped", "reason": "already_processed"}

        file_bytes = path.read_bytes()
        record = extract_candidate(filename, file_bytes)

        with connection() as conn:
            with conn.cursor() as cursor:
                strategy = upsert_candidate(cursor, record, batch_id=batch_id)
            conn.commit()

        _finalize(spool_path, batch_id)
        logger.info(
            "Processed %s (batch %s) name=%s dedup=%s",
            filename, batch_id, record.get("name"), strategy,
        )
        return {
            "file": filename,
            "status": "done",
            "name": record.get("name"),
            "dedup_strategy": strategy,
        }

    except PermanentExtractionError as exc:
        # Retrying an unsupported file type or a malformed document burns the
        # entire retry budget to fail identically. Fail now, keep the reason.
        _finalize(spool_path, batch_id)
        logger.error("Permanent failure on %s: %s", filename, exc.message)
        raise

    except SoftTimeLimitExceeded as exc:
        # The call overran its slot. Almost always a stalled upstream request
        # rather than a bad file, so it is worth one more attempt.
        _retry_or_give_up(self, exc, spool_path, filename, batch_id,
                               reason="timed out")

    except TransientExtractionError as exc:
        _retry_or_give_up(self, exc, spool_path, filename, batch_id,
                               reason=exc.message)

    except ExtractionError:
        # Any future ExtractionError subclass that is neither transient nor
        # permanent should surface loudly rather than be silently retried.
        _finalize(spool_path, batch_id)
        logger.exception("Unclassified extraction failure on %s", filename)
        raise

    except Exception as exc:
        # Database or infrastructure failure. The spool file is deliberately
        # kept so the retry can re-read it -- unless this was the last attempt.
        _retry_or_give_up(self, exc, spool_path, filename, batch_id,
                               reason="infrastructure failure")
