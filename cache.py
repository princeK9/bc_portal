"""
Short-TTL Redis cache in front of /search.

Every search embeds the query with MiniLM and then runs a pgvector scan. Both
are repeated verbatim whenever a recruiter refines and re-runs a query, pages
through results, or reloads the page. Caching the response collapses that to
one round trip.

Uses a separate Redis logical database from the Celery broker (db 1 vs db 0) so
that flushing the cache can never discard queued work. A cache is disposable by
definition; a job queue is not, and sharing one keyspace makes it far too easy
to destroy the second while clearing the first.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Optional

import redis

from config import REDIS_CACHE_URL, SEARCH_CACHE_ENABLED, SEARCH_CACHE_TTL

logger = logging.getLogger(__name__)

CACHE_PREFIX = "search:v1:"

_client: Optional[redis.Redis] = None


def get_cache() -> redis.Redis:
    """Cache client, created on first use."""
    global _client
    if _client is None:
        _client = redis.Redis.from_url(REDIS_CACHE_URL, decode_responses=True)
    return _client


def cache_key(query: str, limit: int) -> str:
    """
    Stable key for a query/limit pair.

    Hashed rather than interpolated: a raw query string can be long enough to
    make an unwieldy key and can contain characters that complicate Redis
    tooling. The version prefix means a change to the response shape can be
    rolled out by bumping it, instead of serving stale payloads that no longer
    match what the client expects.
    """
    digest = hashlib.sha256(f"{query}\x00{limit}".encode("utf-8")).hexdigest()
    return f"{CACHE_PREFIX}{digest}"


def get_cached(query: str, limit: int) -> Optional[dict]:
    """
    Return a cached response, or None on a miss.

    Never raises: Redis being unreachable must degrade /search to "slower",
    not "broken". A cache that can take the endpoint down with it is a
    liability rather than an optimisation.
    """
    if not SEARCH_CACHE_ENABLED:
        return None
    try:
        raw = get_cache().get(cache_key(query, limit))
    except redis.RedisError:
        logger.warning("Search cache unavailable on read; serving uncached.", exc_info=True)
        return None
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Discarding malformed cache entry for query=%r", query)
        return None


def flush_search_cache() -> int:
    """
    Drop every cached search response. Returns how many keys were removed.

    Called when an upload batch finishes, because the corpus has just changed
    and any cached result is now describing a database that no longer exists.
    Without this a recruiter can run a search straight after an ingestion and
    be served pre-upload results for up to the TTL, with nothing to indicate
    the answer is stale.

    Safe because this client is connected to REDIS_CACHE_URL, a different
    logical database from the broker: FLUSHDB affects only the currently
    selected db, so queued Celery work on db 0 cannot be touched. That
    separation exists precisely so this operation is available at all --
    FLUSHALL, or a shared keyspace, would make it unthinkable.
    """
    if not SEARCH_CACHE_ENABLED:
        return 0
    try:
        client = get_cache()
        removed = client.dbsize()
        client.flushdb()
        logger.info("Search cache flushed (%d key(s)) after corpus change.", removed)
        return removed
    except redis.RedisError:
        # Same posture as the rest of this module: a cache failure degrades
        # freshness, it must not fail the batch that triggered it.
        logger.warning("Could not flush search cache.", exc_info=True)
        return 0


def set_cached(query: str, limit: int, payload: dict) -> None:
    """Store a response under the configured TTL. Never raises."""
    if not SEARCH_CACHE_ENABLED:
        return
    try:
        get_cache().set(
            cache_key(query, limit),
            json.dumps(payload),
            ex=SEARCH_CACHE_TTL,
        )
    except (redis.RedisError, TypeError):
        logger.warning("Could not write search cache for query=%r", query, exc_info=True)
