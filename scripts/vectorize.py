"""
Backfill embeddings for candidates that do not have one yet.

Run after scripts/migrate.py. Only touches rows WHERE embedding IS NULL, so it
is safe to re-run and cheap to resume after an interruption.

The embedding text and the encoder both come from extraction.py rather than
being rebuilt here. An earlier version reimplemented both, and that duplication
is precisely what let the two write paths drift into different vector spaces.
"""

import logging
import sys
from pathlib import Path

import psycopg2

# config.py and extraction.py live at the repo root; running this file directly
# only puts scripts/ on sys.path, so add the root explicitly.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config import DB_CONFIG, setup_logging  # noqa: E402
from extraction import build_embedding_text, get_embedder  # noqa: E402

logger = logging.getLogger(__name__)


def generate_and_store_vectors() -> None:
    """
    Encode unvectorized candidates with MiniLM and store the result.

    Rows are read, encoded and updated one at a time rather than batched, so an
    interruption leaves the already-written vectors intact and the next run
    picks up exactly the remaining NULLs.
    """
    model = get_embedder()

    conn = None
    cursor = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT id, raw_title, experience_years, skills FROM candidates WHERE embedding IS NULL"
        )
        rows = cursor.fetchall()

        if not rows:
            logger.info("All candidates are already vectorized.")
            return

        logger.info("Found %d candidates needing vectors. Processing...", len(rows))

        for record_id, raw_title, exp_years, skills in rows:
            # Same helper the upload path uses, so vectors stay comparable.
            semantic_text = build_embedding_text({
                "raw_title": raw_title,
                "experience_years": exp_years,
                "skills": skills,
            })

            # pgvector accepts vectors as a bracketed string list, e.g. "[0.1, 0.2, ...]"
            vector_string = str(model.encode(semantic_text).tolist())

            cursor.execute(
                "UPDATE candidates SET embedding = %s WHERE id = %s",
                (vector_string, record_id),
            )
            logger.debug("Stored vector for candidate ID %s", record_id)

        conn.commit()
        logger.info("%d vectors stored.", len(rows))

    except Exception:
        logger.exception("Vectorization failed; rolling back.")
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
    generate_and_store_vectors()


if __name__ == "__main__":
    main()
