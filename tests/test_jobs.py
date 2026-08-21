"""Job status aggregation: per-file states rolled up into actionable counts."""

import json

import pytest

import jobs
from jobs import classify, summarize


class _FakeAsyncResult:
    def __init__(self, state, result=None):
        self.state = state
        self.result = result


class _FakeCeleryApp:
    def __init__(self, states):
        self._states = states

    def AsyncResult(self, task_id):
        entry = self._states.get(task_id, "PENDING")
        if isinstance(entry, tuple):
            return _FakeAsyncResult(entry[0], entry[1])
        return _FakeAsyncResult(entry)


def _manifest(*task_ids):
    return {
        "batch_id": "job1",
        "files": [{"task_id": t, "file": f"{t}.pdf"} for t in task_ids],
    }


class TestClassify:
    @pytest.mark.parametrize("state", ["PENDING", "RECEIVED", "STARTED", "RETRY"])
    def test_in_flight_states_are_pending(self, state):
        """RETRY in particular must not read as failed -- the file is still
        going to be processed, and reporting it failed would prompt a needless
        re-upload."""
        assert classify(state) == "pending"

    def test_success_is_done(self):
        assert classify("SUCCESS") == "done"

    @pytest.mark.parametrize("state", ["FAILURE", "REVOKED"])
    def test_terminal_failures_are_failed(self, state):
        assert classify(state) == "failed"

    def test_unknown_state_is_pending_not_failed(self):
        """Guessing 'failed' for an unrecognised state would report data loss
        that has not happened."""
        assert classify("SOMETHING_NEW") == "pending"


class TestSummarize:
    def test_counts_each_bucket(self):
        app = _FakeCeleryApp({"a": "SUCCESS", "b": "FAILURE", "c": "STARTED"})
        summary = summarize(app, _manifest("a", "b", "c"))
        assert summary["done"] == 1
        assert summary["failed"] == 1
        assert summary["pending"] == 1
        assert summary["total"] == 3

    def test_reports_per_file_detail_not_just_counts(self):
        """'3 failed' is not actionable without knowing which three; the
        alternative is re-uploading a 500-file zip to find out."""
        app = _FakeCeleryApp({"a": "SUCCESS", "b": "FAILURE"})
        summary = summarize(app, _manifest("a", "b"))
        by_file = {entry["file"]: entry for entry in summary["files"]}
        assert by_file["a.pdf"]["status"] == "done"
        assert by_file["b.pdf"]["status"] == "failed"

    def test_complete_only_when_nothing_is_pending(self):
        app = _FakeCeleryApp({"a": "SUCCESS", "b": "STARTED"})
        assert summarize(app, _manifest("a", "b"))["complete"] is False

    def test_complete_with_mixed_terminal_states(self):
        """A job with failures is still finished; 'complete' means no work
        remains, not that everything succeeded."""
        app = _FakeCeleryApp({"a": "SUCCESS", "b": "FAILURE"})
        assert summarize(app, _manifest("a", "b"))["complete"] is True

    def test_progress_counts_failures_as_finished(self):
        app = _FakeCeleryApp({"a": "SUCCESS", "b": "FAILURE", "c": "PENDING", "d": "PENDING"})
        assert summarize(app, _manifest("a", "b", "c", "d"))["progress"] == 0.5

    def test_empty_job_is_not_reported_complete(self):
        app = _FakeCeleryApp({})
        summary = summarize(app, {"batch_id": "job1", "files": []})
        assert summary["complete"] is False
        assert summary["progress"] == 0.0

    def test_failure_detail_exposes_type_not_message(self):
        """Same reasoning as /search: an exception message can carry connection
        strings and row data; the class name cannot."""
        boom = RuntimeError("host=secret.internal password=hunter2")
        app = _FakeCeleryApp({"a": ("FAILURE", boom)})
        summary = summarize(app, _manifest("a"))
        detail = summary["files"][0]
        assert detail["error"] == "RuntimeError"
        assert "hunter2" not in json.dumps(summary)

    def test_successful_file_has_no_error_field(self):
        app = _FakeCeleryApp({"a": "SUCCESS"})
        assert "error" not in summarize(app, _manifest("a"))["files"][0]


class TestManifest:
    def test_missing_manifest_returns_none(self, monkeypatch):
        class _Redis:
            def get(self, key): return None

        monkeypatch.setattr(jobs, "get_redis", lambda: _Redis())
        assert jobs.load_manifest("nope") is None

    def test_malformed_manifest_returns_none(self, monkeypatch):
        class _Redis:
            def get(self, key): return "{not json"

        monkeypatch.setattr(jobs, "get_redis", lambda: _Redis())
        assert jobs.load_manifest("job1") is None

    def test_manifest_round_trips(self, monkeypatch):
        class _Redis:
            def __init__(self): self.store = {}
            def set(self, key, value, ex=None): self.store[key] = value
            def get(self, key): return self.store.get(key)

        # One shared instance: a factory returning a fresh fake per call would
        # have save and load talking to different stores.
        client = _Redis()
        monkeypatch.setattr(jobs, "get_redis", lambda: client)
        entries = [{"task_id": "t1", "file": "a.pdf"}]
        jobs.save_manifest("job1", entries)
        assert jobs.load_manifest("job1")["files"] == entries

    def test_manifest_expires(self, monkeypatch):
        """Finished jobs must not accumulate in Redis forever."""
        recorded = {}

        class _Redis:
            def set(self, key, value, ex=None): recorded["ex"] = ex

        monkeypatch.setattr(jobs, "get_redis", lambda: _Redis())
        jobs.save_manifest("job1", [])
        assert recorded["ex"] == jobs.BATCH_TTL_SECONDS
