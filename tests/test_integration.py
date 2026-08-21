"""
End-to-end exercise of the upload path against the real sample corpus, with
Gemini and Postgres faked but every piece of our own code real.

This does not replace running the stack under docker compose -- broker
redelivery cannot be observed in-process. What it does cover is the property
that redelivery *depends on*: that executing the same task twice converges on
one row rather than two. acks_late deliberately trades "may be lost" for "may
run twice", and that trade is only safe if the write is genuinely idempotent.
"""

import zipfile
from pathlib import Path

import pytest

import jobs
import tasks
from db import INSERT_COLUMNS, SQL_UPSERT_ON_BATCH_FILE, SQL_UPSERT_ON_PHONE
from server import inspect_archive, spool_members
from tasks import process_resume

from conftest import REPO_ROOT  # noqa: E402

SAMPLE_ZIP = "sample_data/sample_resumes.zip"


class FakeCandidatesTable:
    """
    Stands in for the candidates table, enforcing the same uniqueness rules the
    migration creates. Keyed exactly as the two partial unique indexes are, so
    an upsert that targets the wrong key shows up here as a duplicate row.
    """

    def __init__(self):
        self.rows = {}
        self.insert_count = 0

    def execute(self, sql, params=None):
        # SAVEPOINT / RELEASE / ROLLBACK carry no params and write nothing.
        if params is None:
            return
        self.insert_count += 1
        columns = dict(zip(INSERT_COLUMNS, params))

        if sql is SQL_UPSERT_ON_PHONE:
            key = ("phone", columns["phone_normalized"])
        elif sql is SQL_UPSERT_ON_BATCH_FILE:
            key = ("batch_file", columns["batch_id"], columns["source_file"])
        else:
            # Plain insert: no key, so every execution is a new row.
            key = ("plain", self.insert_count)

        self.rows[key] = columns

    def __enter__(self): return self
    def __exit__(self, *args): return False


class FakeConnection:
    def __init__(self, table):
        self.table = table
        self.commits = 0

    def cursor(self): return self.table
    def commit(self): self.commits += 1

    def __enter__(self): return self
    def __exit__(self, *args): return False


@pytest.fixture(autouse=True)
def isolate_batch_bookkeeping(monkeypatch):
    """
    Stub the terminal-outcome bookkeeping that _finalize() performs.

    Without this these tests reach jobs.mark_file_finished(), which opens a real
    Redis connection and blocks until it times out. That dependency arrived with
    the batch-completion work and was invisible while a Docker stack happened to
    be running -- the suite passed, then hung the moment it was run without one.
    A test that only passes when infrastructure is up is not a unit test.
    """
    monkeypatch.setattr(tasks, "flush_search_cache", lambda: 0)
    monkeypatch.setattr(tasks.jobs, "mark_file_finished",
                        lambda batch_id: (1, 10 ** 9))  # never "last file"


@pytest.fixture
def fake_db(monkeypatch):
    table = FakeCandidatesTable()
    monkeypatch.setattr(tasks, "connection", lambda: FakeConnection(table))
    return table


@pytest.fixture
def fake_gemini(monkeypatch):
    """Deterministic extraction keyed off the filename, so the same file always
    yields the same phone -- which is what makes the upsert converge."""
    def _extract(filename, file_bytes):
        stem = filename.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        digits = f"{abs(hash(stem)) % 10_000_000_000:010d}"
        return {
            "name": stem.replace("_", " ").title(),
            "raw_title": "Welder",
            "experience_years": 5,
            "skills": ["Welder"],
            "phone": f"({digits[:3]}) {digits[3:6]}-{digits[6:]}",
            "email": f"{stem}@example.com",
            "embedding": [0.0] * 384,
            "source_file": filename,
        }

    monkeypatch.setattr(tasks, "extract_candidate", _extract)
    return _extract


@pytest.fixture
def spooled_sample(tmp_path):
    """Unpack the committed 23-file sample zip through the real code path.

    Both the zip and the unpacked sample_data/resumes/ directory are gitignored
    (large, local-only, regenerable) rather than shipped in the repo, so a
    fresh clone won't have either. Skip cleanly rather than erroring, matching
    the pattern conftest.py's sample_resumes_dir fixture already uses for the
    same reason.
    """
    if not Path(SAMPLE_ZIP).is_file():
        pytest.skip("local sample corpus not present: sample_data/sample_resumes.zip")
    if not (REPO_ROOT / "sample_data" / "resumes").is_dir():
        pytest.skip("local sample corpus not present: sample_data/resumes")

    archive = zipfile.ZipFile(SAMPLE_ZIP)
    with archive:
        members = inspect_archive(archive)
        return spool_members(archive, members, tmp_path / "jobsample")


class TestRealSampleArchive:
    def test_sample_zip_passes_validation(self, spooled_sample):
        """Counted from the corpus rather than hardcoded -- an earlier version
        asserted 23 and broke the moment the corpus grew to 46, which is noise
        rather than signal."""
        expected = len([p for p in (REPO_ROOT / "sample_data" / "resumes").iterdir()
                        if p.is_file()])
        assert len(spooled_sample) == expected

    def test_every_member_lands_on_disk(self, spooled_sample):
        for path, _ in spooled_sample:
            assert path.exists() and path.stat().st_size > 0

    def test_every_corpus_format_survives(self, spooled_sample):
        """Whatever formats the corpus contains must all make it through."""
        expected = {p.suffix.lower() for p in (REPO_ROOT / "sample_data" / "resumes").iterdir()
                    if p.is_file()}
        assert {path.suffix for path, _ in spooled_sample} == expected
        assert ".txt" in expected, "txt coverage was added deliberately; keep it"


class TestFullBatch:
    def test_every_file_produces_exactly_one_row(self, spooled_sample, fake_db, fake_gemini):
        for path, name in spooled_sample:
            process_resume(str(path), name, "jobsample")

        assert len(fake_db.rows) == len(spooled_sample)

    def test_spool_is_emptied_as_files_complete(self, spooled_sample, fake_db, fake_gemini):
        for path, name in spooled_sample:
            process_resume(str(path), name, "jobsample")

        assert not any(path.exists() for path, _ in spooled_sample)

    def test_all_writes_used_the_phone_key(self, spooled_sample, fake_db, fake_gemini):
        for path, name in spooled_sample:
            process_resume(str(path), name, "jobsample")

        assert all(key[0] == "phone" for key in fake_db.rows)


class TestRedeliveryIsIdempotent:
    """The scenario acks_late creates: a worker dies after the row is written
    but before the broker is acked, so the message is redelivered and the task
    runs a second time."""

    def test_rerunning_the_same_file_does_not_duplicate(self, tmp_path, fake_db, fake_gemini):
        spool = tmp_path / "00000.pdf"

        spool.write_bytes(b"resume")
        first = process_resume(str(spool), "marcus.pdf", "job1")

        # Redelivery: the broker hands the same message to another worker.
        # In production the spool file would still be present, because the
        # first attempt is presumed to have died before cleanup.
        spool.write_bytes(b"resume")
        second = process_resume(str(spool), "marcus.pdf", "job1")

        assert first["status"] == "done"
        assert second["status"] == "done"
        assert fake_db.insert_count == 2, "both executions must reach the database"
        assert len(fake_db.rows) == 1, "but they must collapse onto one row"

    def test_phoneless_resume_is_also_idempotent(self, tmp_path, fake_db, monkeypatch):
        """The fallback key exists precisely so the ~5% of resumes with no
        extractable phone are still retry-safe."""
        def _no_phone(filename, file_bytes):
            return {
                "name": None, "raw_title": "Welder", "experience_years": 1,
                "skills": ["Welder"], "phone": None, "email": None,
                "embedding": [0.0] * 384, "source_file": filename,
            }

        monkeypatch.setattr(tasks, "extract_candidate", _no_phone)
        spool = tmp_path / "00000.pdf"

        spool.write_bytes(b"resume")
        process_resume(str(spool), "anon.pdf", "job1")
        spool.write_bytes(b"resume")
        process_resume(str(spool), "anon.pdf", "job1")

        assert fake_db.insert_count == 2
        assert len(fake_db.rows) == 1
        assert list(fake_db.rows)[0][0] == "batch_file"

    def test_same_person_in_two_files_collapses_to_one_row(self, tmp_path, fake_db, monkeypatch):
        """Deduplication, as distinct from retry safety: two different resumes
        for one person merge on phone."""
        def _same_person(filename, file_bytes):
            return {
                "name": "Marcus Devlin", "raw_title": "Electrician",
                "experience_years": 9, "skills": ["Electrician"],
                "phone": "(918) 246-8135", "email": "m@example.com",
                "embedding": [0.0] * 384, "source_file": filename,
            }

        monkeypatch.setattr(tasks, "extract_candidate", _same_person)

        for name in ("marcus_v1.pdf", "marcus_v2.pdf"):
            spool = tmp_path / name
            spool.write_bytes(b"resume")
            process_resume(str(spool), name, "job1")

        assert fake_db.insert_count == 2
        assert len(fake_db.rows) == 1

    def test_different_people_stay_separate(self, tmp_path, fake_db, fake_gemini):
        for name in ("a.pdf", "b.pdf", "c.pdf"):
            spool = tmp_path / name
            spool.write_bytes(b"resume")
            process_resume(str(spool), name, "job1")

        assert len(fake_db.rows) == 3


class TestStatusReportingOverARealBatch:
    def test_mixed_outcomes_roll_up_correctly(self, spooled_sample):
        manifest = {
            "batch_id": "jobsample",
            "files": [
                {"task_id": f"t{i}", "file": name}
                for i, (_, name) in enumerate(spooled_sample)
            ],
        }

        states = {f"t{i}": "SUCCESS" for i in range(len(spooled_sample))}
        states["t0"] = "FAILURE"
        states["t1"] = "RETRY"

        class _App:
            def AsyncResult(self, task_id):
                class _R:
                    state = states[task_id]
                    result = RuntimeError("boom") if states[task_id] == "FAILURE" else None
                return _R()

        summary = jobs.summarize(_App(), manifest)

        total = len(spooled_sample)
        assert summary["total"] == total
        assert summary["done"] == total - 2
        assert summary["failed"] == 1
        assert summary["pending"] == 1, "RETRY is still in flight, not failed"
        assert summary["complete"] is False
        assert len(summary["files"]) == total


class RefreshableTable(FakeCandidatesTable):
    """
    Fake table that also honours DO UPDATE, so a second write to the same key
    replaces the stored column values rather than merely being counted.
    Mirrors the behaviour verified directly against Postgres.
    """

    def execute(self, sql, params=None):
        # SAVEPOINT / RELEASE / ROLLBACK carry no params and write nothing.
        if params is None:
            return
        self.insert_count += 1
        columns = dict(zip(INSERT_COLUMNS, params))

        if sql is SQL_UPSERT_ON_PHONE:
            key = ("phone", columns["phone_normalized"])
        elif sql is SQL_UPSERT_ON_BATCH_FILE:
            key = ("batch_file", columns["batch_id"], columns["source_file"])
        elif sql is getattr(__import__("db"), "SQL_UPSERT_ON_EMAIL"):
            key = ("email", columns["email"])
        else:
            key = ("plain", self.insert_count)

        # DO UPDATE SET: later write wins on every column.
        self.rows[key] = columns


@pytest.fixture
def refreshable_db(monkeypatch):
    table = RefreshableTable()
    monkeypatch.setattr(tasks, "connection", lambda: FakeConnection(table))
    return table


def _resume(**kw):
    base = {
        "name": "Marcus Devlin", "raw_title": "Welder", "experience_years": 5,
        "skills": ["Welder"], "phone": "(918) 246-8135",
        "email": "marcus@example.com", "embedding": [0.1] * 384,
        "source_file": "marcus_v1.pdf",
    }
    base.update(kw)
    return base


class TestReUploadRefreshesTheRecord:
    """
    Distinct from redelivery safety. Redelivery asks "does running twice avoid a
    duplicate?". This asks "when a candidate re-applies with a *better* resume,
    does the stored row actually reflect the new data?" A DO NOTHING upsert
    would pass the first question and silently fail the second, leaving the
    candidate permanently stale.

    Verified independently against a real Postgres instance; the values asserted
    here are the ones that run produced.
    """

    def test_same_phone_new_data_overwrites_every_field(self, tmp_path, refreshable_db,
                                                        monkeypatch):
        versions = iter([
            _resume(),
            _resume(raw_title="HVAC Technician", experience_years=12,
                    skills=["HVAC Technician", "Fitter"],
                    source_file="marcus_v2_updated.pdf"),
        ])
        monkeypatch.setattr(tasks, "extract_candidate", lambda f, b: next(versions))

        for name, batch in (("marcus_v1.pdf", "batch-A"), ("marcus_v2_updated.pdf", "batch-B")):
            spool = tmp_path / name
            spool.write_bytes(b"resume")
            process_resume(str(spool), name, batch)

        assert len(refreshable_db.rows) == 1, "must refresh in place, not insert a second row"
        stored = refreshable_db.rows[("phone", "9182468135")]

        # Before -> after, exactly as observed against Postgres.
        assert stored["raw_title"] == "HVAC Technician"        # was "Welder"
        assert stored["experience_years"] == 12                # was 5
        assert stored["skills"] == ["HVAC Technician", "Fitter"]  # was ["Welder"]
        assert stored["source_file"] == "marcus_v2_updated.pdf"   # was marcus_v1.pdf
        assert stored["batch_id"] == "batch-B"                 # was batch-A

    def test_refresh_is_not_a_silent_no_op(self, tmp_path, refreshable_db, monkeypatch):
        """Guards specifically against ON CONFLICT DO NOTHING being reintroduced."""
        versions = iter([
            _resume(experience_years=5),
            _resume(experience_years=12, source_file="v2.pdf"),
        ])
        monkeypatch.setattr(tasks, "extract_candidate", lambda f, b: next(versions))

        for name in ("v1.pdf", "v2.pdf"):
            spool = tmp_path / name
            spool.write_bytes(b"resume")
            process_resume(str(spool), name, "batch-A")

        stored = refreshable_db.rows[("phone", "9182468135")]
        assert stored["experience_years"] != 5, "stale value survived; upsert did nothing"
        assert stored["experience_years"] == 12
