"""
Job bookkeeping in Redis: the manifest for an upload batch and its status roll-up.

Celery's result backend already knows each task's state. What it does not know
is which file a task belongs to, or which tasks made up a job. That mapping is
stored here so /status can report per-file detail rather than one opaque flag.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

import redis

from config import BATCH_TTL_SECONDS, REDIS_URL

logger = logging.getLogger(__name__)

# Celery states grouped into the three outcomes a caller actually cares about.
PENDING_STATES = frozenset({"PENDING", "RECEIVED", "STARTED", "RETRY"})
DONE_STATES = frozenset({"SUCCESS"})
FAILED_STATES = frozenset({"FAILURE", "REVOKED"})

_client: Optional[redis.Redis] = None


def get_redis() -> redis.Redis:
    """Redis client for job bookkeeping, created on first use."""
    global _client
    if _client is None:
        _client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    return _client


def _manifest_key(batch_id: str) -> str:
    return f"batch:{batch_id}:manifest"


def save_manifest(batch_id: str, entries: list[dict]) -> None:
    """
    Record which task id handles which file.

    entries: [{"task_id": ..., "file": ...}, ...] in dispatch order.
    Expires with BATCH_TTL_SECONDS so finished jobs do not accumulate forever.
    """
    payload = json.dumps({"batch_id": batch_id, "files": entries})
    get_redis().set(_manifest_key(batch_id), payload, ex=BATCH_TTL_SECONDS)


def load_manifest(batch_id: str) -> Optional[dict]:
    """Return the stored manifest, or None if unknown or expired."""
    raw = get_redis().get(_manifest_key(batch_id))
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Manifest for job %s is not valid JSON; treating as missing.", batch_id)
        return None


def mark_file_finished(batch_id: str) -> tuple[int, int]:
    """
    Record that one file reached a terminal outcome. Returns (finished, total).

    Uses INCR rather than counting SUCCESS/FAILURE states in the result backend,
    because a task asking "is my batch done?" cannot see its own result yet --
    it is still running at that point, so it would never observe completion.
    INCR is atomic, so concurrent workers finishing simultaneously cannot both
    miss the moment the last file lands.

    Counts terminal outcomes only. A retry must not increment, or a batch with
    retries would appear complete before its work is.
    """
    redis_client = get_redis()
    key = f"batch:{batch_id}:finished"
    finished = redis_client.incr(key)
    # Expire alongside the manifest so counters do not outlive their batch.
    redis_client.expire(key, BATCH_TTL_SECONDS)

    manifest = load_manifest(batch_id)
    total = len(manifest.get("files", [])) if manifest else 0
    return finished, total


def classify(state: str) -> str:
    """Collapse a Celery task state into pending / done / failed."""
    if state in DONE_STATES:
        return "done"
    if state in FAILED_STATES:
        return "failed"
    return "pending"


def summarize(celery_app, manifest: dict) -> dict:
    """
    Build the /status payload by asking the result backend about each task.

    Reports counts *and* per-file detail: an aggregate "3 failed" is not
    actionable without knowing which three files, and re-uploading a 500-file
    zip to find out is not a reasonable answer.
    """
    files = manifest.get("files", [])
    counts = {"done": 0, "failed": 0, "pending": 0}
    details = []

    for entry in files:
        task_id = entry.get("task_id")
        filename = entry.get("file")
        async_result = celery_app.AsyncResult(task_id)
        state = async_result.state
        bucket = classify(state)
        counts[bucket] += 1

        detail = {"file": filename, "task_id": task_id, "state": state, "status": bucket}
        if bucket == "failed":
            # Exception *type* only. The message can carry connection strings
            # and row data, same reasoning as the /search error handling.
            info = async_result.result
            detail["error"] = type(info).__name__ if isinstance(info, BaseException) else "Unknown"
        details.append(detail)

    total = len(files)
    finished = counts["done"] + counts["failed"]
    return {
        "batch_id": manifest.get("batch_id"),
        "total": total,
        "done": counts["done"],
        "failed": counts["failed"],
        "pending": counts["pending"],
        "complete": finished == total and total > 0,
        "progress": round(finished / total, 3) if total else 0.0,
        "files": details,
    }
