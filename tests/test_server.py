"""API surface: score calibration, ZIP filtering, status codes, error hygiene."""

import io
import zipfile

import psycopg2
import pytest
from fastapi.testclient import TestClient

import server
from server import _is_candidate_file, calibrate_score


@pytest.fixture
def client():
    """TestClient without entering the lifespan context, so the suite does not
    pay the model-load cost for endpoint tests that never embed anything."""
    return TestClient(server.app)


def _zip_bytes(names: list[str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name in names:
            archive.writestr(name, b"synthetic content")
    return buffer.getvalue()


class TestCalibrateScore:
    @pytest.mark.parametrize("raw_sim,expected_tier", [
        (0.80, "High Match"),
        (0.55, "Moderate Match"),
        (0.25, "Low Match"),
    ])
    def test_tiers(self, raw_sim, expected_tier):
        _, tier = calibrate_score(raw_sim)
        assert tier == expected_tier

    @pytest.mark.parametrize("raw_sim", [-1.0, 0.0, 0.2, 0.8, 1.0, 2.0])
    def test_score_is_always_clamped_to_percentage_range(self, raw_sim):
        score, _ = calibrate_score(raw_sim)
        assert 0.0 <= score <= 100.0

    def test_monotonic_in_similarity(self):
        scores = [calibrate_score(s)[0] for s in (0.2, 0.4, 0.6, 0.8)]
        assert scores == sorted(scores)


class TestZipEntryFiltering:
    @pytest.mark.parametrize("name", [
        "resumes/a.pdf", "resumes/a.docx", "resumes/a.txt",
        "resumes/a.png", "resumes/a.jpg", "resumes/a.jpeg",
        "A.PDF",
    ])
    def test_accepts_supported(self, name):
        assert _is_candidate_file(name)

    @pytest.mark.parametrize("name", [
        "__MACOSX/a.pdf",       # macOS metadata sidecar
        "resumes/.DS_Store",    # hidden file
        "resumes/",             # directory entry
        "resumes/a.exe",        # unsupported type
        "resumes/notes.md",
        "",
    ])
    def test_rejects_unsupported(self, name):
        assert not _is_candidate_file(name)


class TestHealth:
    def test_returns_online(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "online"}


class TestUploadValidation:
    def test_non_zip_payload_is_400(self, client):
        response = client.post(
            "/upload-resumes/", files={"file": ("x.zip", b"not a zip", "application/zip")}
        )
        assert response.status_code == 400
        assert "valid .zip" in response.json()["detail"]

    def test_zip_without_supported_files_is_400(self, client):
        payload = _zip_bytes(["readme.md", "notes.txt.bak", "__MACOSX/x.pdf"])
        response = client.post(
            "/upload-resumes/", files={"file": ("x.zip", payload, "application/zip")}
        )
        assert response.status_code == 400
        assert "No supported resume files" in response.json()["detail"]


class TestSearchErrorHandling:
    """Failures must map to honest status codes and must not leak internals."""

    def test_database_unreachable_returns_503(self, client, monkeypatch):
        monkeypatch.setattr(server, "embed_text", lambda text: [0.0] * 384)

        def _refuse(**kwargs):
            raise psycopg2.OperationalError(
                "could not connect to server: host=secret.internal user=admin password=hunter2"
            )

        monkeypatch.setattr(server.psycopg2, "connect", _refuse)

        response = client.get("/search", params={"query": "welder"})
        assert response.status_code == 503

    def test_unexpected_failure_returns_500(self, client, monkeypatch):
        monkeypatch.setattr(server, "embed_text", lambda text: [0.0] * 384)

        def _explode(**kwargs):
            raise RuntimeError("unexpected internal failure")

        monkeypatch.setattr(server.psycopg2, "connect", _explode)

        response = client.get("/search", params={"query": "welder"})
        assert response.status_code == 500

    @pytest.mark.parametrize("exception,secret", [
        (psycopg2.OperationalError("host=secret.internal password=hunter2"), "hunter2"),
        (RuntimeError("table candidates_internal blew up"), "candidates_internal"),
    ])
    def test_response_never_leaks_exception_text(self, client, monkeypatch, exception, secret):
        """Raw exception strings can carry hostnames, table names and
        credentials. They are logged server-side, never serialized."""
        monkeypatch.setattr(server, "embed_text", lambda text: [0.0] * 384)

        def _raise(**kwargs):
            raise exception

        monkeypatch.setattr(server.psycopg2, "connect", _raise)

        response = client.get("/search", params={"query": "welder"})
        assert secret not in response.text
        assert response.json()["detail"] in (
            server.GENERIC_SEARCH_ERROR,
            server.GENERIC_UPSTREAM_ERROR,
        )
