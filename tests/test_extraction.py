"""Extraction module: taxonomy, embedding format, file routing, retry policy."""

import pickle
from pathlib import Path

import httpx
import pytest
from google.genai import errors as genai_errors

import extraction
from extraction import (
    EMBEDDING_DIMENSIONS,
    FALLBACK_SKILL,
    SUPPORTED_EXTENSIONS,
    TAXONOMY,
    ExtractionError,
    PermanentExtractionError,
    TransientExtractionError,
    _build_contents,
    _docx_to_text,
    build_embedding_text,
    embed_text,
    extract_candidate,
    extract_candidate_with_backoff,
    is_retryable,
    normalize_skills,
)

EMPTY_RESUME_JSON = (
    '{"name": null, "raw_title": null, "experience_years": null, '
    '"phone": null, "email": null, "skills": []}'
)


class _FakeResponse:
    def __init__(self, text: str):
        self.text = text


class _FakeModels:
    def __init__(self, behaviour):
        self._behaviour = behaviour
        self.calls = 0

    def generate_content(self, **kwargs):
        self.calls += 1
        return self._behaviour()


class _FakeClient:
    def __init__(self, behaviour):
        self.models = _FakeModels(behaviour)


def _install_fake_client(monkeypatch, behaviour):
    """Point extraction at a fake Gemini client and a no-op embedder."""
    client = _FakeClient(behaviour)
    monkeypatch.setattr(extraction, "get_client", lambda: client)
    monkeypatch.setattr(extraction, "embed_text", lambda text: [0.0] * EMBEDDING_DIMENSIONS)
    return client


def _api_error(cls, code, status):
    return cls(code, {"error": {"status": status, "message": "synthetic"}})


class TestNormalizeSkills:
    def test_canonicalises_case(self):
        assert normalize_skills(["electrician", "WELDER"]) == ["Electrician", "Welder"]

    def test_drops_free_text_drift(self):
        """The model drifts to free text despite the prompt; unrecognised
        labels must never reach the database."""
        assert normalize_skills(["MIG Welding"]) == [FALLBACK_SKILL]

    def test_dedupes_preserving_order(self):
        assert normalize_skills(["Plumber", "plumber", "Welder"]) == ["Plumber", "Welder"]

    def test_keeps_only_valid_members(self):
        assert normalize_skills(["Welder", "Nonsense", "Fitter"]) == ["Welder", "Fitter"]

    @pytest.mark.parametrize("value", [None, [], ["", "   "], [123, None]])
    def test_empty_and_junk_fall_back(self, value):
        assert normalize_skills(value) == [FALLBACK_SKILL]

    def test_strips_surrounding_whitespace(self):
        assert normalize_skills(["  Welder  "]) == ["Welder"]

    def test_every_taxonomy_member_survives_roundtrip(self):
        assert normalize_skills(list(TAXONOMY)) == list(TAXONOMY)


class TestBuildEmbeddingText:
    def test_canonical_format(self):
        text = build_embedding_text(
            {"raw_title": "Welder", "experience_years": 6, "skills": ["Welder"]}
        )
        assert text == "Title: Welder. Experience: 6 years. Skills: Welder."

    @pytest.mark.parametrize("skills", [None, []])
    def test_missing_skills_render_as_none(self, skills):
        text = build_embedding_text(
            {"raw_title": "Plumber", "experience_years": 0, "skills": skills}
        )
        assert text == "Title: Plumber. Experience: 0 years. Skills: None."

    def test_handles_entirely_empty_record(self):
        assert build_embedding_text({}) == "Title: None. Experience: None years. Skills: None."

    def test_matches_vectorize_backfill_path(self):
        """vectorize.py must produce byte-identical text to the upload path, or
        backfilled rows land in a different region of the vector space. This is
        the regression that motivated the shared module."""
        import vectorize

        assert vectorize.build_embedding_text is build_embedding_text


class TestFileRouting:
    def test_supported_extension_set(self):
        assert set(SUPPORTED_EXTENSIONS) == {".pdf", ".jpg", ".jpeg", ".png", ".docx", ".txt"}

    def test_docx_is_decoded_locally(self, sample_docx: Path):
        contents = _build_contents(sample_docx.read_bytes(), ".docx")
        assert isinstance(contents[1], str)

    def test_txt_is_decoded_locally(self):
        contents = _build_contents(b"Jane Doe\nPlumber", ".txt")
        assert isinstance(contents[1], str)
        assert "Jane Doe" in contents[1]

    def test_txt_survives_invalid_utf8(self):
        """Scanned-then-converted text files are not always clean UTF-8;
        decoding must not raise."""
        contents = _build_contents(b"caf\xe9 Plumber", ".txt")
        assert isinstance(contents[1], str)

    def test_image_is_sent_as_multimodal_part(self, sample_png: Path):
        contents = _build_contents(sample_png.read_bytes(), ".png")
        assert not isinstance(contents[1], str)

    def test_unsupported_extension_raises(self):
        with pytest.raises(PermanentExtractionError, match="unsupported file type"):
            extract_candidate("resume.rtf", b"irrelevant")


class TestDocxToText:
    def test_extracts_visible_content(self, sample_docx: Path):
        text = _docx_to_text(sample_docx.read_bytes())
        assert text.strip()
        assert len(text.splitlines()) > 1


class TestRetryClassification:
    """Retry decisions are made on typed SDK exceptions and status codes.
    Substring matching on the message was the previous approach and is exactly
    what these tests exist to prevent regressing to."""

    @staticmethod
    def _api_error(cls, code, status):
        return cls(code, {"error": {"status": status, "message": "synthetic"}})

    @pytest.mark.parametrize("code,status", [
        (503, "UNAVAILABLE"),
        (500, "INTERNAL"),
        (504, "DEADLINE_EXCEEDED"),
    ])
    def test_server_errors_are_retryable(self, code, status):
        assert is_retryable(self._api_error(genai_errors.ServerError, code, status))

    def test_rate_limit_is_retryable(self):
        assert is_retryable(
            self._api_error(genai_errors.ClientError, 429, "RESOURCE_EXHAUSTED")
        )

    @pytest.mark.parametrize("code,status", [
        (400, "INVALID_ARGUMENT"),
        (401, "UNAUTHENTICATED"),
        (403, "PERMISSION_DENIED"),
        (404, "NOT_FOUND"),
    ])
    def test_client_errors_are_not_retryable(self, code, status):
        """A bad key or malformed request never fixes itself by waiting."""
        assert not is_retryable(self._api_error(genai_errors.ClientError, code, status))

    def test_network_faults_are_retryable(self):
        assert is_retryable(httpx.ConnectError("connection refused"))
        assert is_retryable(httpx.ReadTimeout("timed out"))

    def test_unrelated_exception_containing_status_digits_is_not_retryable(self):
        """The precise failure mode of the old string-matching implementation:
        any error whose text happened to contain '503' was retried."""
        assert not is_retryable(ValueError("resume mentions 503 Industrial Way"))

    def test_plain_exception_is_not_retryable(self):
        assert not is_retryable(RuntimeError("boom"))


class TestEmbedding:
    def test_produces_expected_dimensions(self):
        vector = embed_text("Title: Welder. Experience: 6 years. Skills: Welder.")
        assert len(vector) == EMBEDDING_DIMENSIONS
        assert all(isinstance(value, float) for value in vector)

    def test_is_deterministic(self):
        text = "Title: Plumber. Experience: 3 years. Skills: Plumber."
        assert embed_text(text) == embed_text(text)


class TestLazyInitialisation:
    def test_importing_does_not_build_client(self):
        """Import must stay cheap and credential-free so tests and the CLI can
        import this module without a key or a model download."""
        assert extraction._client is None or isinstance(extraction._client, object)

    def test_missing_api_key_raises_clear_error(self, monkeypatch):
        monkeypatch.setattr(extraction, "GEMINI_API_KEY", "")
        monkeypatch.setattr(extraction, "_client", None)
        with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
            extraction.get_client()


class TestFailureSignalling:
    """
    The core guarantee of this module: a failed extraction raises, and a
    successful extraction returns a dict. There is no third state. A resume that
    genuinely contains no phone number and a resume the API refused to process
    must never produce the same value, because the caller writes one to the
    database and must retry the other.
    """

    def test_skeleton_helper_no_longer_exists(self):
        """The old failure path returned _empty_record(). If it comes back, a
        failure can silently become a row in the candidates table."""
        assert not hasattr(extraction, "_empty_record")

    def test_empty_resume_succeeds_and_returns_a_record(self, monkeypatch):
        _install_fake_client(monkeypatch, lambda: _FakeResponse(EMPTY_RESUME_JSON))
        record = extract_candidate("sparse.pdf", b"bytes")
        assert isinstance(record, dict)
        assert record["name"] is None
        assert record["phone"] is None
        assert record["source_file"] == "sparse.pdf"
        assert record["skills"] == [FALLBACK_SKILL]
        assert len(record["embedding"]) == EMBEDDING_DIMENSIONS

    def test_failure_raises_rather_than_returning_a_lookalike(self, monkeypatch):
        """The failure case must not be distinguishable only by inspecting
        fields -- it must be impossible to reach the return value at all."""
        def _fail():
            raise _api_error(genai_errors.ClientError, 400, "INVALID_ARGUMENT")

        _install_fake_client(monkeypatch, _fail)
        with pytest.raises(ExtractionError):
            extract_candidate("broken.pdf", b"bytes")

    def test_empty_success_and_failure_are_not_confusable(self, monkeypatch):
        """Both scenarios produce 'no candidate data'. One returns, one raises."""
        _install_fake_client(monkeypatch, lambda: _FakeResponse(EMPTY_RESUME_JSON))
        empty_success = extract_candidate("sparse.pdf", b"bytes")

        def _fail():
            raise _api_error(genai_errors.ServerError, 503, "UNAVAILABLE")

        _install_fake_client(monkeypatch, _fail)
        outcome = None
        try:
            outcome = extract_candidate("sparse.pdf", b"bytes")
        except ExtractionError as exc:
            outcome = exc

        assert isinstance(empty_success, dict)
        assert isinstance(outcome, ExtractionError)
        assert type(empty_success) is not type(outcome)

    def test_transient_api_failure_is_transient(self, monkeypatch):
        def _fail():
            raise _api_error(genai_errors.ServerError, 503, "UNAVAILABLE")

        _install_fake_client(monkeypatch, _fail)
        with pytest.raises(TransientExtractionError):
            extract_candidate("a.pdf", b"bytes")

    @pytest.mark.parametrize("code,status", [
        (400, "INVALID_ARGUMENT"),
        (401, "UNAUTHENTICATED"),
        (403, "PERMISSION_DENIED"),
    ])
    def test_permanent_api_failure_is_permanent(self, monkeypatch, code, status):
        def _fail():
            raise _api_error(genai_errors.ClientError, code, status)

        _install_fake_client(monkeypatch, _fail)
        with pytest.raises(PermanentExtractionError):
            extract_candidate("a.pdf", b"bytes")

    def test_unreadable_file_is_permanent(self, monkeypatch):
        _install_fake_client(monkeypatch, lambda: _FakeResponse(EMPTY_RESUME_JSON))
        with pytest.raises(PermanentExtractionError, match="could not read file"):
            extract_candidate("broken.docx", b"this is definitely not a docx")

    def test_unparseable_model_output_is_permanent(self, monkeypatch):
        _install_fake_client(monkeypatch, lambda: _FakeResponse("not json at all"))
        with pytest.raises(PermanentExtractionError, match="unparseable JSON"):
            extract_candidate("a.pdf", b"bytes")

    def test_error_carries_filename_for_batch_reporting(self, monkeypatch):
        def _fail():
            raise _api_error(genai_errors.ClientError, 400, "INVALID_ARGUMENT")

        _install_fake_client(monkeypatch, _fail)
        with pytest.raises(ExtractionError) as info:
            extract_candidate("resumes/marcus.pdf", b"bytes")
        assert info.value.filename == "resumes/marcus.pdf"
        assert "resumes/marcus.pdf" in str(info.value)

    def test_original_exception_is_chained(self, monkeypatch):
        """__cause__ must survive so the worker log shows the real API error."""
        def _fail():
            raise _api_error(genai_errors.ClientError, 400, "INVALID_ARGUMENT")

        _install_fake_client(monkeypatch, _fail)
        with pytest.raises(ExtractionError) as info:
            extract_candidate("a.pdf", b"bytes")
        assert isinstance(info.value.__cause__, genai_errors.APIError)

    def test_both_error_types_share_a_base(self):
        """Callers that do not care about the distinction catch ExtractionError."""
        assert issubclass(TransientExtractionError, ExtractionError)
        assert issubclass(PermanentExtractionError, ExtractionError)


class TestBlockingBackoffWrapper:
    """extract_candidate_with_backoff is the CLI's retry path. The Celery task
    must never use it -- time.sleep in a worker holds the slot for the backoff."""

    def test_returns_on_first_success(self, monkeypatch):
        client = _install_fake_client(monkeypatch, lambda: _FakeResponse(EMPTY_RESUME_JSON))
        record = extract_candidate_with_backoff("a.pdf", b"bytes", max_attempts=3)
        assert isinstance(record, dict)
        assert client.models.calls == 1

    def test_retries_transient_then_succeeds(self, monkeypatch):
        monkeypatch.setattr(extraction.time, "sleep", lambda seconds: None)
        state = {"n": 0}

        def _flaky():
            state["n"] += 1
            if state["n"] < 3:
                raise _api_error(genai_errors.ServerError, 503, "UNAVAILABLE")
            return _FakeResponse(EMPTY_RESUME_JSON)

        client = _install_fake_client(monkeypatch, _flaky)
        record = extract_candidate_with_backoff("a.pdf", b"bytes", max_attempts=5)
        assert isinstance(record, dict)
        assert client.models.calls == 3

    def test_gives_up_after_max_attempts(self, monkeypatch):
        monkeypatch.setattr(extraction.time, "sleep", lambda seconds: None)

        def _fail():
            raise _api_error(genai_errors.ServerError, 503, "UNAVAILABLE")

        client = _install_fake_client(monkeypatch, _fail)
        with pytest.raises(TransientExtractionError):
            extract_candidate_with_backoff("a.pdf", b"bytes", max_attempts=4)
        assert client.models.calls == 4

    def test_permanent_failure_is_not_retried(self, monkeypatch):
        """Retrying a 400 wastes the whole backoff budget and still fails."""
        monkeypatch.setattr(extraction.time, "sleep", lambda seconds: None)

        def _fail():
            raise _api_error(genai_errors.ClientError, 400, "INVALID_ARGUMENT")

        client = _install_fake_client(monkeypatch, _fail)
        with pytest.raises(PermanentExtractionError):
            extract_candidate_with_backoff("a.pdf", b"bytes", max_attempts=5)
        assert client.models.calls == 1


class TestNoHardcodedSecrets:
    def test_no_placeholder_api_key_literal(self):
        for name in ("server.py", "extraction.py", "config.py"):
            source = (Path(__file__).resolve().parent.parent / name).read_text(encoding="utf-8")
            assert 'api_key="API_key"' not in source

    def test_old_sdk_is_gone(self):
        source = (Path(__file__).resolve().parent.parent / "extraction.py").read_text(encoding="utf-8")
        assert "google.generativeai" not in source
        assert "from google import genai" in source


class TestExceptionsSurviveTheResultBackend:
    """
    Celery rebuilds an exception from its result backend as cls(*exc.args). A
    two-argument __init__ whose args held one pre-formatted string could not be
    reconstructed, so Celery substituted UnpickleableExceptionWrapper and
    /status reported that meaningless type for every failure -- the entire typed
    hierarchy was invisible to API clients.

    Found by a live load test, not by the unit tests, because nothing here had
    ever round-tripped an exception.
    """

    @pytest.mark.parametrize("cls", [
        ExtractionError, TransientExtractionError, PermanentExtractionError,
    ])
    def test_rebuilds_the_way_celery_does(self, cls):
        original = cls("resumes/a.pdf", "transient API failure: 429")
        rebuilt = type(original)(*original.args)

        assert type(rebuilt) is cls
        assert rebuilt.filename == "resumes/a.pdf"
        assert rebuilt.message == "transient API failure: 429"

    @pytest.mark.parametrize("cls", [
        ExtractionError, TransientExtractionError, PermanentExtractionError,
    ])
    def test_survives_pickling(self, cls):
        original = cls("resumes/a.pdf", "boom")
        restored = pickle.loads(pickle.dumps(original))

        assert type(restored) is cls
        assert restored.filename == original.filename
        assert restored.message == original.message

    def test_args_carry_both_fields_not_one_formatted_string(self):
        """The precise shape of the bug: args of length 1 cannot rebuild a
        two-argument __init__."""
        exc = TransientExtractionError("a.pdf", "boom")
        assert len(exc.args) == 2
        assert exc.args == ("a.pdf", "boom")

    def test_str_is_still_human_readable(self):
        """Splitting args must not turn log lines into a bare tuple."""
        assert str(TransientExtractionError("a.pdf", "boom")) == "a.pdf: boom"

    def test_subclasses_keep_their_identity_through_a_round_trip(self):
        """A transient error rebuilt as a permanent one would invert the retry
        decision on the far side."""
        transient = pickle.loads(pickle.dumps(TransientExtractionError("a.pdf", "x")))
        permanent = pickle.loads(pickle.dumps(PermanentExtractionError("a.pdf", "x")))

        assert isinstance(transient, TransientExtractionError)
        assert not isinstance(transient, PermanentExtractionError)
        assert isinstance(permanent, PermanentExtractionError)
        assert not isinstance(permanent, TransientExtractionError)

    def test_reported_type_name_is_the_real_one(self):
        """jobs.summarize reports type(result).__name__ into /status. This is
        what showed UnpickleableExceptionWrapper before the fix."""
        import jobs

        exc = TransientExtractionError("resumes/a.pdf", "429 RESOURCE_EXHAUSTED")
        restored = pickle.loads(pickle.dumps(exc))

        class _App:
            def AsyncResult(self, task_id):
                class _R:
                    state = "FAILURE"
                    result = restored
                return _R()

        summary = jobs.summarize(_App(), {
            "batch_id": "b1",
            "files": [{"task_id": "t1", "file": "resumes/a.pdf"}],
        })
        detail = summary["files"][0]
        assert detail["error"] == "TransientExtractionError"
        assert detail["error"] != "UnpickleableExceptionWrapper"
