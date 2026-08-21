"""
Archive validation and spooling.

Limits are enforced before any task is dispatched. Fanning out 10,000 tasks and
then discovering the archive was junk costs real Gemini spend and floods the
queue behind legitimate work.
"""

import io
import zipfile

import pytest
from fastapi import HTTPException

import server
from server import inspect_archive, spool_members


def _build_zip(entries, compression=zipfile.ZIP_DEFLATED):
    """entries: {name: bytes}"""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression) as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    buffer.seek(0)
    return zipfile.ZipFile(buffer)


class TestMemberSelection:
    def test_keeps_supported_files(self):
        archive = _build_zip({"a.pdf": b"x", "b.docx": b"y", "c.png": b"z"})
        assert len(inspect_archive(archive)) == 3

    def test_drops_junk_entries(self):
        archive = _build_zip({
            "a.pdf": b"x",
            "__MACOSX/a.pdf": b"junk",
            ".DS_Store": b"junk",
            "notes.md": b"junk",
        })
        members = inspect_archive(archive)
        assert [m.filename for m in members] == ["a.pdf"]

    def test_archive_with_no_resumes_is_rejected(self):
        archive = _build_zip({"readme.md": b"x", "notes.txt.bak": b"y"})
        with pytest.raises(HTTPException) as info:
            inspect_archive(archive)
        assert info.value.status_code == 400
        assert "No supported resume files" in info.value.detail


class TestLimits:
    def test_too_many_files_is_rejected(self, monkeypatch):
        monkeypatch.setattr(server, "MAX_ZIP_FILES", 3)
        archive = _build_zip({f"{i}.pdf": b"x" for i in range(5)})
        with pytest.raises(HTTPException) as info:
            inspect_archive(archive)
        assert info.value.status_code == 400
        assert "the limit is 3" in info.value.detail

    def test_at_the_limit_is_accepted(self, monkeypatch):
        """Boundary must be inclusive, or the documented limit is off by one."""
        monkeypatch.setattr(server, "MAX_ZIP_FILES", 3)
        archive = _build_zip({f"{i}.pdf": b"x" for i in range(3)})
        assert len(inspect_archive(archive)) == 3

    def test_total_expanded_size_is_capped(self, monkeypatch):
        monkeypatch.setattr(server, "MAX_ZIP_TOTAL_BYTES", 100)
        archive = _build_zip({f"{i}.pdf": b"x" * 60 for i in range(3)})
        with pytest.raises(HTTPException) as info:
            inspect_archive(archive)
        assert "expands to" in info.value.detail

    def test_single_oversized_file_is_rejected(self, monkeypatch):
        monkeypatch.setattr(server, "MAX_SINGLE_FILE_BYTES", 50)
        archive = _build_zip({"big.pdf": b"x" * 200})
        with pytest.raises(HTTPException) as info:
            inspect_archive(archive)
        assert "per-file limit" in info.value.detail

    def test_zip_bomb_ratio_is_rejected(self, monkeypatch):
        """Highly compressible content: a small archive that expands enormously.
        The size check alone reads the header, which an attacker controls."""
        monkeypatch.setattr(server, "MAX_ZIP_TOTAL_BYTES", 10 ** 9)
        monkeypatch.setattr(server, "MAX_SINGLE_FILE_BYTES", 10 ** 9)
        monkeypatch.setattr(server, "MAX_ZIP_COMPRESSION_RATIO", 10)
        archive = _build_zip({"bomb.pdf": b"\0" * 5_000_000})
        with pytest.raises(HTTPException) as info:
            inspect_archive(archive)
        assert "compression ratio" in info.value.detail

    def test_normal_file_passes_the_ratio_check(self, monkeypatch):
        monkeypatch.setattr(server, "MAX_ZIP_COMPRESSION_RATIO", 100)
        archive = _build_zip({"a.pdf": bytes(range(256)) * 20})
        assert len(inspect_archive(archive)) == 1

    def test_stored_uncompressed_entry_does_not_divide_by_zero(self):
        archive = _build_zip({"a.pdf": b""}, compression=zipfile.ZIP_STORED)
        assert len(inspect_archive(archive)) == 1


class TestSpooling:
    def test_writes_each_member_to_disk(self, tmp_path):
        archive = _build_zip({"a.pdf": b"alpha", "b.pdf": b"beta"})
        members = inspect_archive(archive)
        spooled = spool_members(archive, members, tmp_path / "batch1")

        assert len(spooled) == 2
        for path, _ in spooled:
            assert path.exists()

    def test_original_filename_is_preserved_for_reporting(self, tmp_path):
        archive = _build_zip({"resumes/marcus_devlin.pdf": b"x"})
        members = inspect_archive(archive)
        spooled = spool_members(archive, members, tmp_path / "batch1")
        assert spooled[0][1] == "resumes/marcus_devlin.pdf"

    def test_content_survives_the_round_trip(self, tmp_path):
        archive = _build_zip({"a.pdf": b"exact bytes"})
        members = inspect_archive(archive)
        spooled = spool_members(archive, members, tmp_path / "batch1")
        assert spooled[0][0].read_bytes() == b"exact bytes"

    def test_path_traversal_cannot_escape_the_spool_dir(self, tmp_path):
        """A member named ../../etc/passwd must not be written outside the job
        directory. Entries are written under generated names, never the
        attacker-supplied path."""
        batch_dir = tmp_path / "batch1"
        archive = _build_zip({"../../../evil.pdf": b"x"})
        members = inspect_archive(archive)
        spooled = spool_members(archive, members, batch_dir)

        for path, _ in spooled:
            assert batch_dir.resolve() in path.resolve().parents

    def test_extension_is_retained_for_format_detection(self, tmp_path):
        """extract_candidate routes on the suffix, so the spooled name must
        keep it or every file would be rejected as unsupported."""
        archive = _build_zip({"a.docx": b"x", "b.png": b"y"})
        members = inspect_archive(archive)
        spooled = spool_members(archive, members, tmp_path / "batch1")
        assert sorted(p.suffix for p, _ in spooled) == [".docx", ".png"]

    def test_names_are_unique_even_for_colliding_basenames(self, tmp_path):
        """Two entries in different folders can share a basename; if the spool
        names collided one file would silently overwrite the other."""
        archive = _build_zip({"one/a.pdf": b"first", "two/a.pdf": b"second"})
        members = inspect_archive(archive)
        spooled = spool_members(archive, members, tmp_path / "batch1")
        paths = {p for p, _ in spooled}
        assert len(paths) == 2
