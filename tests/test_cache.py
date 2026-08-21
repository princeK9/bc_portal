"""
Search cache behaviour, including its failure modes.

The most important property here is that Redis being down degrades /search to
"slower", never to "broken". A cache that can take the endpoint down with it is
a liability rather than an optimisation.
"""

import json

import pytest
import redis

import cache
from cache import CACHE_PREFIX, cache_key, get_cached, set_cached


class _FakeRedis:
    def __init__(self, failing=False):
        self.store = {}
        self.failing = failing
        self.set_calls = []

    def get(self, key):
        if self.failing:
            raise redis.ConnectionError("redis is down")
        return self.store.get(key)

    def set(self, key, value, ex=None):
        if self.failing:
            raise redis.ConnectionError("redis is down")
        self.set_calls.append({"key": key, "ex": ex})
        self.store[key] = value

    def dbsize(self):
        if self.failing:
            raise redis.ConnectionError("redis is down")
        return len(self.store)

    def flushdb(self):
        if self.failing:
            raise redis.ConnectionError("redis is down")
        self.store.clear()


@pytest.fixture
def fake_redis(monkeypatch):
    client = _FakeRedis()
    monkeypatch.setattr(cache, "get_cache", lambda: client)
    monkeypatch.setattr(cache, "SEARCH_CACHE_ENABLED", True)
    return client


@pytest.fixture
def broken_redis(monkeypatch):
    client = _FakeRedis(failing=True)
    monkeypatch.setattr(cache, "get_cache", lambda: client)
    monkeypatch.setattr(cache, "SEARCH_CACHE_ENABLED", True)
    return client


class TestCacheKey:
    def test_is_stable_for_the_same_inputs(self):
        assert cache_key("welder", 5) == cache_key("welder", 5)

    def test_differs_by_query(self):
        assert cache_key("welder", 5) != cache_key("plumber", 5)

    def test_differs_by_limit(self):
        """Same query at limit 5 and limit 50 are different responses."""
        assert cache_key("welder", 5) != cache_key("welder", 50)

    def test_is_versioned(self):
        """Bumping the prefix retires stale payloads when the response shape
        changes, instead of serving clients a format they no longer parse."""
        assert cache_key("welder", 5).startswith(CACHE_PREFIX)

    def test_separator_prevents_ambiguity(self):
        """('welder1', 0) and ('welder', 10) must not collide."""
        assert cache_key("welder1", 0) != cache_key("welder", 10)


class TestRoundTrip:
    def test_miss_returns_none(self, fake_redis):
        assert get_cached("nothing stored", 5) is None

    def test_stores_and_retrieves(self, fake_redis):
        payload = {"query": "welder", "results": [{"id": 1}]}
        set_cached("welder", 5, payload)
        assert get_cached("welder", 5) == payload

    def test_ttl_is_applied(self, fake_redis, monkeypatch):
        """An unbounded entry would keep serving results that no longer include
        candidates uploaded since."""
        monkeypatch.setattr(cache, "SEARCH_CACHE_TTL", 60)
        set_cached("welder", 5, {"results": []})
        assert fake_redis.set_calls[0]["ex"] == 60


class TestDegradesRatherThanFails:
    def test_read_failure_returns_none(self, broken_redis):
        assert get_cached("welder", 5) is None

    def test_write_failure_is_swallowed(self, broken_redis):
        set_cached("welder", 5, {"results": []})

    def test_malformed_entry_is_discarded(self, fake_redis):
        """A truncated or hand-edited value must not raise into the endpoint."""
        fake_redis.store[cache_key("welder", 5)] = "{not json"
        assert get_cached("welder", 5) is None

    def test_unserializable_payload_does_not_raise(self, fake_redis):
        set_cached("welder", 5, {"bad": object()})


class TestDisabled:
    def test_reads_bypass_when_disabled(self, fake_redis, monkeypatch):
        monkeypatch.setattr(cache, "SEARCH_CACHE_ENABLED", False)
        fake_redis.store[cache_key("welder", 5)] = json.dumps({"results": []})
        assert get_cached("welder", 5) is None

    def test_writes_bypass_when_disabled(self, fake_redis, monkeypatch):
        monkeypatch.setattr(cache, "SEARCH_CACHE_ENABLED", False)
        set_cached("welder", 5, {"results": []})
        assert fake_redis.set_calls == []


class TestFlushSearchCache:
    """
    Invalidation after an upload. Without it, a recruiter searching immediately
    after an ingestion is served pre-upload results for up to the TTL, with
    nothing on screen to suggest the answer is stale.
    """

    def test_flush_removes_cached_entries(self, fake_redis):
        set_cached("welder", 5, {"results": [{"id": 1}]})
        assert get_cached("welder", 5) is not None

        cache.flush_search_cache()
        assert get_cached("welder", 5) is None

    def test_flush_reports_how_many_keys_went(self, fake_redis):
        set_cached("welder", 5, {"results": []})
        set_cached("plumber", 5, {"results": []})
        assert cache.flush_search_cache() == 2

    def test_flush_uses_flushdb_not_flushall(self, fake_redis, monkeypatch):
        """
        FLUSHALL would wipe every logical database on the server, including
        db 0 -- the Celery broker. That would destroy queued work as a side
        effect of a cache invalidation.
        """
        called = []
        monkeypatch.setattr(fake_redis, "flushdb", lambda: called.append("flushdb"), raising=False)
        monkeypatch.setattr(fake_redis, "flushall",
                            lambda: called.append("flushall"), raising=False)
        cache.flush_search_cache()
        assert called == ["flushdb"]

    def test_flush_targets_the_cache_database_not_the_broker(self):
        """The two clients must resolve to different logical databases, which is
        what makes flushing the cache safe at all."""
        import config

        assert config.REDIS_URL.rsplit("/", 1)[-1] != config.REDIS_CACHE_URL.rsplit("/", 1)[-1]

    def test_flush_survives_redis_being_down(self, broken_redis):
        """A cache failure must not fail the batch that triggered the flush."""
        assert cache.flush_search_cache() == 0

    def test_flush_is_a_noop_when_cache_disabled(self, fake_redis, monkeypatch):
        monkeypatch.setattr(cache, "SEARCH_CACHE_ENABLED", False)
        assert cache.flush_search_cache() == 0
