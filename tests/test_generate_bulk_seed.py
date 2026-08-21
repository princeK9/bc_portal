"""Synthetic seed generator: labels and vectors must match the real system."""

import json

import generate_bulk_seed
from extraction import FALLBACK_SKILL, TAXONOMY, build_embedding_text, normalize_skills
from generate_bulk_seed import TRADES, random_candidate


class TestTaxonomyIsShared:
    """The generator used to carry its own trade list, so seeded records were
    labelled with skills the real taxonomy does not contain and search filters
    could never match them."""

    def test_trades_are_drawn_from_the_real_taxonomy(self):
        assert set(TRADES).issubset(set(TAXONOMY))

    def test_fallback_bucket_is_not_used_as_a_job_title(self):
        assert FALLBACK_SKILL not in TRADES

    def test_generated_skills_survive_normalization_unchanged(self):
        """Every generated label must already be canonical -- if normalize_skills
        rewrites them, the seed does not represent real extracted records."""
        for index in range(200):
            record = random_candidate(index)
            assert normalize_skills(record["skills"]) == record["skills"]

    def test_title_is_always_a_taxonomy_member(self):
        for index in range(200):
            assert random_candidate(index)["raw_title"] in TAXONOMY

    def test_title_is_included_in_skills(self):
        for index in range(100):
            record = random_candidate(index)
            assert record["raw_title"] in record["skills"]


class TestRecordShape:
    def test_has_every_column_the_db_expects(self):
        record = random_candidate(0)
        for field in ("name", "raw_title", "experience_years", "phone",
                      "email", "skills", "source_file"):
            assert field in record

    def test_experience_years_in_plausible_range(self):
        for index in range(200):
            assert 0 <= random_candidate(index)["experience_years"] <= 25

    def test_some_records_have_no_phone(self):
        """The real corpus has resumes where phone extraction fails; the seed
        must reproduce that or it will hide null-handling bugs."""
        phones = [random_candidate(i)["phone"] for i in range(500)]
        assert any(phone is None for phone in phones)
        assert any(phone is not None for phone in phones)

    def test_emails_are_unique_per_index(self):
        emails = {random_candidate(i)["email"] for i in range(500)}
        assert len(emails) == 500


class TestEmbeddingFormatMatchesProduction:
    def test_uses_the_shared_embedding_text_helper(self):
        assert generate_bulk_seed.build_embedding_text is build_embedding_text

    def test_generated_text_matches_canonical_format(self):
        record = random_candidate(0)
        expected = (
            f"Title: {record['raw_title']}. "
            f"Experience: {record['experience_years']} years. "
            f"Skills: {', '.join(record['skills'])}."
        )
        assert build_embedding_text(record) == expected


class TestOutputFile:
    def test_written_records_are_valid_jsonl(self, tmp_path):
        out_path = tmp_path / "seed.jsonl"
        generate_bulk_seed.generate(n_records=5, out_path=out_path)

        lines = out_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 5
        for line in lines:
            record = json.loads(line)
            assert len(record["embedding"]) == 384
            assert record["raw_title"] in TAXONOMY
