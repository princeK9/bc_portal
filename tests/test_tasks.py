"""
Per-file task behaviour: retry scheduling, failure classification, cleanup.

The central claim under test is that backoff is scheduled via
self.retry(countdown=...) and never by sleeping. Sleeping inside a task holds a
worker slot open for the whole backoff, so four rate-limited files would idle a
concurrency-4 worker completely.
"""

import time

import pytest
from celery.exceptions import SoftTimeLimitExceeded

import tasks
from extraction import PermanentExtractionError, TransientExtractionError
from tasks import MAX_RETRY_COUNTDOWN, _retry_countdown, process_resume


class _RetrySignal(Exception):
    """Stands in for celery.exceptions.Retry so tests can inspect the call."""


@pytest.fixture
def spooled(tmp_path):
    path = tmp_path / "00000.pdf"
    path.write_bytes(b"resume bytes")
    return path


@pytest.fixture
def captured_retry(monkeypatch):
    """Replace self.retry with a recorder that raises, as the real one does."""
    calls = []

    def _fake_retry(exc=None, countdown=None, **kwargs):
        calls.append({"exc": exc, "countdown": countdown})
        raise _RetrySignal()

    monkeypatch.setattr(process_resume, "retry", _fake_retry)
    return calls


@pytest.fixture
def no_sleep(monkeypatch):
    """Fail loudly if the task ever blocks its slot with time.sleep."""
    def _forbidden(seconds):
        raise AssertionError(
            f"task called time.sleep({seconds}); backoff must use self.retry(countdown=)"
        )

    monkeypatch.setattr(time, "sleep", _forbidden)
    monkeypatch.setattr(tasks.time if hasattr(tasks, "time") else time, "sleep", _forbidden)


def _patch_extract(monkeypatch, behaviour):
    monkeypatch.setattr(tasks, "extract_candidate", behaviour)


def _patch_db(monkeypatch, recorder=None):
    class _Cursor:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def execute(self, *a, **k): pass

    class _Conn:
        def cursor(self): return _Cursor()
        def commit(self):
            if recorder is not None:
                recorder.append("commit")

    class _ConnCtx:
        def __enter__(self): return _Conn()
        def __exit__(self, *a): return False

    monkeypatch.setattr(tasks, "connection", lambda: _ConnCtx())
    monkeypatch.setattr(tasks, "upsert_candidate", lambda cur, rec, batch_id=None: "phone")


class TestRetryCountdown:
    def test_grows_exponentially(self):
        assert _retry_countdown(0) < _retry_countdown(4)

    def test_is_capped(self):
        """Unbounded backoff holds a job open for hours to no benefit."""
        assert _retry_countdown(50) <= MAX_RETRY_COUNTDOWN * 1.2

    def test_is_jittered(self):
        """A rate limit hits every in-flight task at once; without jitter they
        all retry in the same second and trip it again."""
        samples = {_retry_countdown(6) for _ in range(40)}
        assert len(samples) > 1

    def test_is_always_positive(self):
        assert all(_retry_countdown(n) > 0 for n in range(10))


class TestSuccessPath:
    def test_writes_and_reports_done(self, monkeypatch, spooled, no_sleep):
        commits = []
        _patch_extract(monkeypatch, lambda name, data: {
            "name": "Marcus", "phone": "9182468135", "embedding": [0.0] * 384,
            "source_file": name,
        })
        _patch_db(monkeypatch, commits)

        result = process_resume(str(spooled), "marcus.pdf", "job1")

        assert result["status"] == "done"
        assert result["file"] == "marcus.pdf"
        assert commits == ["commit"]

    def test_spool_file_is_removed_after_success(self, monkeypatch, spooled, no_sleep):
        _patch_extract(monkeypatch, lambda name, data: {
            "name": "X", "phone": None, "embedding": [0.0] * 384, "source_file": name,
        })
        _patch_db(monkeypatch)

        process_resume(str(spooled), "x.pdf", "job1")
        assert not spooled.exists()

    def test_result_excludes_the_embedding(self, monkeypatch, spooled, no_sleep):
        """384 floats per file would bloat the result backend for no benefit."""
        _patch_extract(monkeypatch, lambda name, data: {
            "name": "X", "phone": None, "embedding": [0.1] * 384, "source_file": name,
        })
        _patch_db(monkeypatch)

        result = process_resume(str(spooled), "x.pdf", "job1")
        assert "embedding" not in result


class TestTransientFailure:
    def test_schedules_a_retry_with_a_countdown(self, monkeypatch, spooled,
                                                captured_retry, no_sleep):
        def _boom(name, data):
            raise TransientExtractionError(name, "503 upstream")

        _patch_extract(monkeypatch, _boom)
        _patch_db(monkeypatch)

        with pytest.raises(_RetrySignal):
            process_resume(str(spooled), "a.pdf", "job1")

        assert len(captured_retry) == 1
        assert captured_retry[0]["countdown"] is not None
        assert captured_retry[0]["countdown"] > 0

    def test_spool_file_survives_for_the_retry(self, monkeypatch, spooled,
                                              captured_retry, no_sleep):
        """Deleting it here would make every subsequent attempt fail on a
        missing path -- the retry would be guaranteed to fail."""
        def _boom(name, data):
            raise TransientExtractionError(name, "503 upstream")

        _patch_extract(monkeypatch, _boom)
        _patch_db(monkeypatch)

        with pytest.raises(_RetrySignal):
            process_resume(str(spooled), "a.pdf", "job1")

        assert spooled.exists()


class TestPermanentFailure:
    def test_is_not_retried(self, monkeypatch, spooled, captured_retry, no_sleep):
        """Retrying an unsupported file type burns the whole budget to fail
        identically five more times."""
        def _boom(name, data):
            raise PermanentExtractionError(name, "unsupported file type")

        _patch_extract(monkeypatch, _boom)
        _patch_db(monkeypatch)

        with pytest.raises(PermanentExtractionError):
            process_resume(str(spooled), "a.rtf", "job1")

        assert captured_retry == []

    def test_spool_file_is_cleaned_up(self, monkeypatch, spooled, no_sleep):
        def _boom(name, data):
            raise PermanentExtractionError(name, "unsupported file type")

        _patch_extract(monkeypatch, _boom)
        _patch_db(monkeypatch)

        with pytest.raises(PermanentExtractionError):
            process_resume(str(spooled), "a.rtf", "job1")

        assert not spooled.exists()


class TestTimeout:
    def test_soft_timeout_is_retried(self, monkeypatch, spooled, captured_retry, no_sleep):
        """A stalled upstream call is usually worth one more attempt."""
        def _hang(name, data):
            raise SoftTimeLimitExceeded()

        _patch_extract(monkeypatch, _hang)
        _patch_db(monkeypatch)

        with pytest.raises(_RetrySignal):
            process_resume(str(spooled), "a.pdf", "job1")

        assert len(captured_retry) == 1


class TestInfrastructureFailure:
    def test_database_error_is_retried(self, monkeypatch, spooled, captured_retry, no_sleep):
        _patch_extract(monkeypatch, lambda name, data: {
            "name": "X", "phone": None, "embedding": [0.0] * 384, "source_file": name,
        })

        def _explode():
            raise RuntimeError("connection pool exhausted")

        monkeypatch.setattr(tasks, "connection", _explode)

        with pytest.raises(_RetrySignal):
            process_resume(str(spooled), "a.pdf", "job1")

        assert len(captured_retry) == 1

    def test_spool_file_survives_a_db_failure(self, monkeypatch, spooled,
                                              captured_retry, no_sleep):
        _patch_extract(monkeypatch, lambda name, data: {
            "name": "X", "phone": None, "embedding": [0.0] * 384, "source_file": name,
        })

        def _explode():
            raise RuntimeError("connection pool exhausted")

        monkeypatch.setattr(tasks, "connection", _explode)

        with pytest.raises(_RetrySignal):
            process_resume(str(spooled), "a.pdf", "job1")

        assert spooled.exists()


class TestRedelivery:
    def test_missing_spool_file_is_treated_as_already_done(self, tmp_path, no_sleep):
        """acks_late means a task can be redelivered after it finished. The
        spool file is gone because the first run completed, so re-running would
        add nothing -- report skipped rather than failing the job."""
        missing = tmp_path / "gone.pdf"
        result = process_resume(str(missing), "gone.pdf", "job1")
        assert result["status"] == "skipped"
        assert result["reason"] == "already_processed"


class _FakeJobs:
    """Stands in for the Redis-backed finished counter."""

    def __init__(self, total):
        self.total = total
        self.finished = 0

    def mark_file_finished(self, batch_id):
        self.finished += 1
        return self.finished, self.total


@pytest.fixture
def batch_env(monkeypatch, tmp_path):
    """A spool directory laid out the way the API creates it, plus fakes for
    the counter and the cache flush."""
    batch_dir = tmp_path / "spool" / "batch1"
    batch_dir.mkdir(parents=True)
    monkeypatch.setattr(tasks, "SPOOL_DIR", tmp_path / "spool")

    flushes = []
    monkeypatch.setattr(tasks, "flush_search_cache", lambda: flushes.append(1))

    fake_jobs = _FakeJobs(total=2)
    monkeypatch.setattr(tasks.jobs, "mark_file_finished", fake_jobs.mark_file_finished)
    return {"dir": batch_dir, "flushes": flushes, "jobs": fake_jobs}


class TestSpoolCleanup:
    """
    Measured behaviour before this was added: per-file spool files were removed
    on success, but the batch directory was left behind on every upload, and a
    file that exhausted its retries kept its spool file forever.
    """

    def test_file_removed_and_batch_dir_kept_while_work_remains(self, batch_env):
        spool = batch_env["dir"] / "00000.pdf"
        spool.write_bytes(b"x")

        tasks._finalize(str(spool), "batch1")

        assert not spool.exists(), "processed file must be removed"
        assert batch_env["dir"].exists(), "directory must survive until the batch ends"
        assert batch_env["flushes"] == [], "cache must not flush mid-batch"

    def test_batch_dir_removed_and_cache_flushed_on_last_file(self, batch_env):
        for name in ("00000.pdf", "00001.pdf"):
            spool = batch_env["dir"] / name
            spool.write_bytes(b"x")
            tasks._finalize(str(spool), "batch1")

        assert not batch_env["dir"].exists(), "empty batch directory must not accumulate"
        assert batch_env["flushes"] == [1], "cache flushed exactly once, on completion"

    def test_finalize_never_removes_the_spool_root(self, batch_env, monkeypatch):
        """A path directly under SPOOL_DIR must not take the whole spool with it."""
        root_file = (batch_env["dir"].parent) / "loose.pdf"
        root_file.write_bytes(b"x")
        batch_env["jobs"].total = 1

        tasks._finalize(str(root_file), "batch1")

        assert batch_env["dir"].parent.exists(), "SPOOL_DIR itself must survive"

    def test_counter_failure_does_not_break_the_task(self, batch_env, monkeypatch):
        def _boom(batch_id):
            raise RuntimeError("redis down")

        monkeypatch.setattr(tasks.jobs, "mark_file_finished", _boom)
        spool = batch_env["dir"] / "00000.pdf"
        spool.write_bytes(b"x")

        tasks._finalize(str(spool), "batch1")   # must not raise
        assert not spool.exists(), "file still removed even if bookkeeping failed"


class TestRetryExhaustionCleansUp:
    """The path that previously leaked: a file using its whole retry budget."""

    def test_exhausted_retries_finalize_the_file(self, monkeypatch, spooled, no_sleep):
        finalized = []
        monkeypatch.setattr(tasks, "_finalize",
                            lambda p, b: finalized.append((p, b)))
        monkeypatch.setattr(process_resume, "retry",
                            lambda **kw: pytest.fail("must not retry once budget is spent"))
        monkeypatch.setattr(type(process_resume.request), "retries",
                            property(lambda self: tasks.TASK_MAX_RETRIES), raising=False)

        def _boom(name, data):
            raise TransientExtractionError(name, "429 rate limited")

        _patch_extract(monkeypatch, _boom)
        _patch_db(monkeypatch)

        with pytest.raises(TransientExtractionError):
            process_resume(str(spooled), "a.pdf", "batch1")

        assert len(finalized) == 1, "exhausted retries must finalize, not leak the file"

    def test_retry_still_available_does_not_finalize(self, monkeypatch, spooled,
                                                    captured_retry, no_sleep):
        finalized = []
        monkeypatch.setattr(tasks, "_finalize", lambda p, b: finalized.append(p))

        def _boom(name, data):
            raise TransientExtractionError(name, "429 rate limited")

        _patch_extract(monkeypatch, _boom)
        _patch_db(monkeypatch)

        with pytest.raises(_RetrySignal):
            process_resume(str(spooled), "a.pdf", "batch1")

        assert finalized == [], "a retry must not delete the file it will re-read"
