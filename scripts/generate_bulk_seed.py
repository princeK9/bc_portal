"""
Generate synthetic candidate records with real MiniLM embeddings, for local
load-testing or seeding a test database.

Does NOT call Gemini and does NOT write to Supabase -- output is a local JSONL
file only.

Trades and skills are drawn from extraction.TAXONOMY, and the embedding text
comes from extraction.build_embedding_text. An earlier version kept its own
trade list and built its own embedding string, which meant seeded records
carried skill labels the real system does not recognise and vectors in a
different format than production rows -- so any load test run against them
would have measured the wrong thing.
"""

import json
import logging
import random
import sys
import time
from pathlib import Path

# config.py and extraction.py live at the repo root; running this file directly
# only puts scripts/ on sys.path, so add the root explicitly.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config import SAMPLE_DATA_DIR, setup_logging  # noqa: E402
from extraction import (  # noqa: E402
    FALLBACK_SKILL,
    TAXONOMY,
    build_embedding_text,
    get_embedder,
)

logger = logging.getLogger(__name__)

random.seed(7)

N_RECORDS = 100_000
OUT_PATH = SAMPLE_DATA_DIR / "bulk_seed.jsonl"
BATCH_SIZE = 256

# "Other" is the fallback bucket, not a trade anyone lists as their title.
TRADES = [label for label in TAXONOMY if label != FALLBACK_SKILL]

FIRST_NAMES = [
    "James", "Maria", "Robert", "Linda", "Michael", "Patricia", "David", "Jennifer", "Carlos", "Aisha",
    "Wei", "Fatima", "John", "Sofia", "Trevor", "Priya", "Marcus", "Angela", "Devon", "Kimberly",
    "Wesley", "Renata", "Curtis", "Latoya", "Bryan", "Julio", "Nadia", "Omar", "Grace", "Hiroshi",
]
LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez", "Hernandez",
    "Lopez", "Wilson", "Anderson", "Thomas", "Nakamura", "Souza", "Devlin", "Petrova", "Coburn", "Ramaswamy",
    "O'Sullivan", "Restrepo", "Osei", "Fenwick", "Hackett", "Yamada", "Marchetti",
]
CITIES = [
    "Tulsa, OK", "Beaumont, TX", "Charlotte, NC", "Reno, NV", "Fresno, CA", "Columbus, OH", "Portland, ME",
    "Pittsburgh, PA", "Boise, ID", "Atlanta, GA", "Grand Rapids, MI", "Newark, NJ", "San Antonio, TX",
    "Kansas City, MO", "Sacramento, CA", "Louisville, KY", "Omaha, NE", "Tucson, AZ", "Albuquerque, NM",
]


def random_candidate(index: int) -> dict:
    """
    Build one synthetic candidate.

    Skills always include the candidate's own trade plus a few adjacent
    categories, which mirrors what normalize_skills() emits for a real resume:
    every label is a member of the taxonomy, never free text.
    """
    trade = random.choice(TRADES)
    others = [label for label in TRADES if label != trade]
    skills = [trade] + random.sample(others, k=random.randint(0, 3))

    first = random.choice(FIRST_NAMES)
    last = random.choice(LAST_NAMES)
    # ~5% of records have no phone, matching the real corpus where phone
    # extraction fails on some scanned resumes.
    has_phone = random.random() > 0.05

    return {
        "name": f"{first} {last}",
        "raw_title": trade,
        "experience_years": random.randint(0, 25),
        "phone": f"({random.randint(200, 999)}) 555-{random.randint(1000, 9999)}" if has_phone else None,
        "email": f"{first.lower()}.{last.lower().replace(chr(39), '')}{index}@example.com",
        "skills": skills,
        "source_file": f"synthetic_{index}.pdf",
        "city": random.choice(CITIES),
    }


def generate(n_records: int = N_RECORDS, out_path: Path = OUT_PATH) -> None:
    """Generate n_records candidates with embeddings and stream them to JSONL."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = get_embedder()

    logger.info("Generating %d synthetic candidate records...", n_records)
    start = time.perf_counter()

    buffered_records: list[dict] = []
    buffered_texts: list[str] = []
    written = 0

    with out_path.open("w", encoding="utf-8") as handle:
        for index in range(n_records):
            record = random_candidate(index)
            buffered_records.append(record)
            # Same helper the upload and backfill paths use.
            buffered_texts.append(build_embedding_text(record))

            if len(buffered_records) >= BATCH_SIZE or index == n_records - 1:
                embeddings = model.encode(buffered_texts, show_progress_bar=False)
                for buffered_record, embedding in zip(buffered_records, embeddings):
                    buffered_record["embedding"] = embedding.tolist()
                    handle.write(json.dumps(buffered_record) + "\n")

                written += len(buffered_records)
                buffered_records.clear()
                buffered_texts.clear()

                if written % 10_000 < BATCH_SIZE:
                    logger.info(
                        "%d/%d records written (%.1fs elapsed)",
                        written, n_records, time.perf_counter() - start,
                    )

    elapsed = time.perf_counter() - start
    size_bytes = out_path.stat().st_size
    logger.info("Wrote %d records to %s", written, out_path)
    logger.info("Total time: %.1fs (%.1f min)", elapsed, elapsed / 60)
    logger.info("File size: %d bytes (%.1f MB)", size_bytes, size_bytes / (1024 * 1024))


def main() -> None:
    """CLI entry point."""
    setup_logging()
    generate()


if __name__ == "__main__":
    main()
