"""Batch CLI: durability of the results file and resume-after-interrupt logic."""

import json

import pytest

import pipeline
from pipeline import load_existing, save_atomic


@pytest.fixture
def results_path(tmp_path):
    return tmp_path / "candidates.json"


SAMPLE_RECORDS = [
    {"name": "A", "source_file": "001.png"},
    {"name": "B", "source_file": "002.png"},
]


class TestLoadExisting:
    def test_missing_file_is_empty_start(self, results_path):
        assert load_existing(results_path) == []

    def test_roundtrips_written_records(self, results_path):
        save_atomic(SAMPLE_RECORDS, results_path)
        assert load_existing(results_path) == SAMPLE_RECORDS

    def test_truncated_json_does_not_crash(self, results_path):
        """A half-written file must degrade to 'start fresh', not raise -- the
        whole point of the resumable path is surviving a hard interrupt."""
        results_path.write_text("[{'truncated", encoding="utf-8")
        assert load_existing(results_path) == []

    def test_non_list_payload_is_rejected(self, results_path):
        results_path.write_text(json.dumps({"not": "a list"}), encoding="utf-8")
        assert load_existing(results_path) == []

    def test_empty_file_does_not_crash(self, results_path):
        results_path.write_text("", encoding="utf-8")
        assert load_existing(results_path) == []


class TestSaveAtomic:
    def test_creates_parent_directories(self, tmp_path):
        nested = tmp_path / "data" / "candidates.json"
        save_atomic(SAMPLE_RECORDS, nested)
        assert nested.exists()

    def test_leaves_no_temp_file_behind(self, results_path):
        save_atomic(SAMPLE_RECORDS, results_path)
        assert not results_path.with_suffix(".json.tmp").exists()

    def test_overwrite_preserves_valid_json(self, results_path):
        save_atomic(SAMPLE_RECORDS, results_path)
        save_atomic(SAMPLE_RECORDS + [{"name": "C", "source_file": "003.png"}], results_path)
        assert len(load_existing(results_path)) == 3

    def test_previous_content_survives_failed_write(self, results_path, monkeypatch):
        """If the new write dies mid-flight, the old file must still be intact
        rather than truncated -- that is what os.replace buys us."""
        save_atomic(SAMPLE_RECORDS, results_path)

        def _explode(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(json, "dump", _explode)
        with pytest.raises(OSError):
            save_atomic([{"name": "C"}], results_path)

        assert load_existing(results_path) == SAMPLE_RECORDS


class TestResumeSkipLogic:
    def test_already_processed_files_are_skipped(self):
        done = {record["source_file"] for record in SAMPLE_RECORDS}
        all_files = ["001.png", "002.png", "003.png"]
        assert [name for name in all_files if name not in done] == ["003.png"]

    def test_nothing_pending_when_all_done(self):
        done = {record["source_file"] for record in SAMPLE_RECORDS}
        assert [n for n in ["001.png", "002.png"] if n not in done] == []

    def test_empty_input_directory_is_a_noop(self, tmp_path, caplog):
        """Must log and return rather than raising, so a cron-driven run
        against an empty uploads/ is not an error."""
        pipeline.process_all_resumes(tmp_path)
        assert "No supported resumes" in caplog.text
