"""
Load data/candidates.json into the candidates table.

Run after scripts/pipeline.py has produced the JSON. Embeddings are populated
separately by scripts/vectorize.py, so rows land here with a NULL embedding.
"""

import json
import logging
import sys
from pathlib import Path

import psycopg2

# config.py lives at the repo root; running this file directly only puts
# scripts/ on sys.path, so add the root explicitly.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config import CANDIDATES_JSON, DB_CONFIG, setup_logging  # noqa: E402

logger = logging.getLogger(__name__)


def migrate_json_to_db() -> None:
    """
    Upsert every record in candidates.json, keyed on email.

    Uses ON CONFLICT rather than a check-then-insert so re-running the migration
    is idempotent and safe against concurrent writers.
    """
    if not CANDIDATES_JSON.exists():
        logger.error("Source file not found at: %s", CANDIDATES_JSON)
        return

    with CANDIDATES_JSON.open(encoding="utf-8") as handle:
        candidates = json.load(handle)

    logger.info("Migrating %d records from %s", len(candidates), CANDIDATES_JSON)

    conn = None
    cursor = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        insert_query = """
            INSERT INTO candidates (name, raw_title, experience_years, phone, email, skills, source_file)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (email) DO UPDATE SET
                name = EXCLUDED.name,
                raw_title = EXCLUDED.raw_title,
                experience_years = EXCLUDED.experience_years,
                phone = EXCLUDED.phone,
                skills = EXCLUDED.skills,
                source_file = EXCLUDED.source_file;
        """

        for record in candidates:
            cursor.execute(insert_query, (
                record.get("name"),
                record.get("raw_title"),
                record.get("experience_years"),
                record.get("phone"),
                record.get("email"),
                record.get("skills"),  # psycopg2 maps Python lists to SQL arrays
                record.get("source_file"),
            ))

        conn.commit()
        logger.info("%d records committed.", len(candidates))

    except Exception:
        logger.exception("Transaction failed; rolling back.")
        if conn is not None:
            conn.rollback()

    finally:
        if cursor is not None:
            cursor.close()
        if conn is not None:
            conn.close()


def main() -> None:
    """CLI entry point."""
    setup_logging()
    migrate_json_to_db()


if __name__ == "__main__":
    main()
