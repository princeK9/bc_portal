"""Phone normalization and the idempotent write path."""

import pytest

import db
from db import (
    SQL_PLAIN_INSERT,
    SQL_UPSERT_ON_BATCH_FILE,
    SQL_UPSERT_ON_PHONE,
    normalize_phone,
    upsert_candidate,
)


class _FakeCursor:
    """Records the SQL and params it was handed, executes nothing."""

    def __init__(self):
        self.statements = []

    def execute(self, sql, params=None):
        self.statements.append((sql, params))

    @property
    def _last_write(self):
        """Last statement that carried params, i.e. the actual INSERT.
        SAVEPOINT / RELEASE / ROLLBACK carry none and must be skipped."""
        return next(s for s in reversed(self.statements) if s[1] is not None)

    @property
    def last_sql(self):
        return self._last_write[0]

    @property
    def last_params(self):
        return self._last_write[1]


def _record(**overrides):
    base = {
        "name": "Marcus Devlin",
        "raw_title": "Electrician",
        "experience_years": 9,
        "skills": ["Electrician"],
        "phone": "(918) 246-8135",
        "email": "marcus@example.com",
        "embedding": [0.0] * 384,
        "source_file": "marcus.pdf",
    }
    base.update(overrides)
    return base


class TestNormalizePhone:
    @pytest.mark.parametrize("raw", [
        "(918) 246-8135",
        "918-246-8135",
        "918.246.8135",
        "9182468135",
        "+1 918 246 8135",
        "+1 (918) 246-8135",
        "  918 246 8135  ",
    ])
    def test_equivalent_formats_collapse_to_one_key(self, raw):
        """The whole point: the same human must produce the same key however
        their number was typed, or dedup silently does nothing."""
        assert normalize_phone(raw) == "9182468135"

    def test_country_code_is_stripped(self):
        assert normalize_phone("+44 918 246 8135") == normalize_phone("9182468135")

    @pytest.mark.parametrize("raw", [None, "", "   ", "246-8135", "12345", "ext. 42", "abc"])
    def test_unusable_numbers_return_none(self, raw):
        """A short fragment must not become a key -- it would collide unrelated
        candidates onto one row, which is worse than not deduplicating."""
        assert normalize_phone(raw) is None

    def test_distinct_numbers_stay_distinct(self):
        assert normalize_phone("918-246-8135") != normalize_phone("918-246-8136")

    def test_result_is_always_ten_digits(self):
        for raw in ["+1 918 246 8135", "00 44 7911 123456", "9182468135"]:
            key = normalize_phone(raw)
            assert key is not None and len(key) == 10 and key.isdigit()


class TestUpsertStrategy:
    def test_record_with_phone_upserts_on_phone(self):
        cursor = _FakeCursor()
        strategy = upsert_candidate(cursor, _record(), batch_id="job1")
        assert strategy == "phone"
        assert cursor.last_sql is SQL_UPSERT_ON_PHONE
        assert "ON CONFLICT (phone_normalized)" in cursor.last_sql

    def test_record_without_phone_falls_back_to_batch_and_file(self):
        """Retry safety for the ~5% of resumes with no extractable phone."""
        cursor = _FakeCursor()
        strategy = upsert_candidate(cursor, _record(phone=None), batch_id="job1")
        assert strategy == "batch_file"
        assert cursor.last_sql is SQL_UPSERT_ON_BATCH_FILE

    def test_unusable_phone_is_treated_as_no_phone(self):
        cursor = _FakeCursor()
        strategy = upsert_candidate(cursor, _record(phone="ext 42"), batch_id="job1")
        assert strategy == "batch_file"

    def test_no_phone_and_no_batch_id_is_a_plain_insert(self):
        """The CLI path has no idempotency key available and says so."""
        cursor = _FakeCursor()
        strategy = upsert_candidate(cursor, _record(phone=None), batch_id=None)
        assert strategy == "plain"
        assert cursor.last_sql is SQL_PLAIN_INSERT
        assert "ON CONFLICT" not in cursor.last_sql

    def test_normalized_phone_is_persisted_not_just_computed(self):
        cursor = _FakeCursor()
        upsert_candidate(cursor, _record(), batch_id="job1")
        assert "9182468135" in cursor.last_params

    def test_batch_id_is_persisted(self):
        cursor = _FakeCursor()
        upsert_candidate(cursor, _record(), batch_id="job-xyz")
        assert "job-xyz" in cursor.last_params

    def test_embedding_is_stringified_for_pgvector(self):
        cursor = _FakeCursor()
        upsert_candidate(cursor, _record(), batch_id="job1")
        embedding_param = [p for p in cursor.last_params if isinstance(p, str) and p.startswith("[")]
        assert embedding_param, "embedding must be passed as a bracketed string"

    def test_param_count_matches_placeholders(self):
        cursor = _FakeCursor()
        upsert_candidate(cursor, _record(), batch_id="job1")
        assert len(cursor.last_params) == len(db.INSERT_COLUMNS)


class TestUniquenessRulesAreDisjoint:
    """The two partial unique indexes must never both apply to one row, or a
    single ON CONFLICT target would be insufficient and inserts would raise."""

    def test_phone_index_predicate_is_not_null(self):
        assert "WHERE phone_normalized IS NOT NULL" in SQL_UPSERT_ON_PHONE

    def test_batch_file_index_predicate_is_the_exact_complement(self):
        assert "phone_normalized IS NULL" in SQL_UPSERT_ON_BATCH_FILE

    def test_upsert_updates_rather_than_ignores(self):
        """DO NOTHING would leave a stale row after a re-extraction; a retry
        should refresh the record, not silently keep the older version."""
        for sql in (SQL_UPSERT_ON_PHONE, SQL_UPSERT_ON_BATCH_FILE):
            assert "DO UPDATE SET" in sql
            assert "DO NOTHING" not in sql


class TestPlaceholderPhonesAreNotIdentities:
    """
    Phone is the *primary* dedup signal, so a number many unrelated people share
    silently merges them and destroys the losers' records.

    Not hypothetical: a production query found five distinct candidates --
    Hiro Huang, Aya Nakamura, Ming Davis, David Anderson, Fred Hill -- all
    carrying (555) 555-5555 from resume templates. Under naive phone dedup they
    would have collapsed into one row and four people's data would be gone.
    """

    @pytest.mark.parametrize("raw", [
        "(555) 555-5555", "555-555-5555", "5555555555", "+1 555 555 5555",
    ])
    def test_literal_five_five_five_is_rejected(self, raw):
        assert normalize_phone(raw) is None

    @pytest.mark.parametrize("raw", [
        "(918) 555-0100", "(918) 555-0142", "(918) 555-0199",
        "(212) 555-0150", "+1 409 555-0198",
    ])
    def test_reserved_fiction_range_is_rejected(self, raw):
        """555-0100..555-0199 is reserved for fiction under every area code, so
        it turns up across unrelated sample and template resumes."""
        assert normalize_phone(raw) is None

    @pytest.mark.parametrize("raw", [
        "0000000000", "1111111111", "9999999999", "(777) 777-7777",
    ])
    def test_repeated_digit_numbers_are_rejected(self, raw):
        assert normalize_phone(raw) is None

    @pytest.mark.parametrize("raw", [
        "(918) 555-0200", "(918) 555-0099", "(918) 555-2142", "(918) 246-8135",
    ])
    def test_real_numbers_just_outside_the_range_still_work(self, raw):
        """The blocklist must not swallow legitimate numbers that merely look
        adjacent, or dedup quietly stops working for real candidates."""
        assert normalize_phone(raw) is not None

    def test_two_people_sharing_a_placeholder_do_not_merge(self):
        """The exact production scenario, as a regression guard."""
        hiro = _record(name="Hiro Huang", phone="(555)555-5555",
                       email="hiro@example.com", source_file="001.png")
        aya = _record(name="Aya Nakamura", phone="(555) 555-5555",
                      email="aya@example.com", source_file="002.png")

        assert normalize_phone(hiro["phone"]) is None
        assert normalize_phone(aya["phone"]) is None

        cursor = _FakeCursor()
        assert upsert_candidate(cursor, hiro, batch_id="batch-A") == "batch_file"
        first_params = cursor.last_params
        assert upsert_candidate(cursor, aya, batch_id="batch-A") == "batch_file"
        second_params = cursor.last_params

        # Different source_file means different key -> two distinct rows.
        assert first_params != second_params
        assert "001.png" in first_params and "002.png" in second_params

    def test_placeholder_still_persists_the_raw_phone(self):
        """Only the dedup key is suppressed. The number a recruiter would dial
        is still stored -- we are declining to treat it as an identity, not
        discarding it."""
        cursor = _FakeCursor()
        upsert_candidate(cursor, _record(phone="(555)555-5555"), batch_id="b1")
        assert "(555)555-5555" in cursor.last_params
        assert None in cursor.last_params  # phone_normalized suppressed
