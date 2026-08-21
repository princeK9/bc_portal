"""
Apply a SQL migration, refusing to proceed if existing data would violate it.

A UNIQUE index creation against a live table either succeeds or fails loudly on
the first duplicate, with no indication of how many others exist. This script
runs the conflict check first and prints every offending group, so the decision
about how to merge duplicates is made deliberately rather than discovered
halfway through a failed DDL.

Usage:
    python scripts/apply_migration.py migrations/001_phone_dedup.sql [--force]
"""

import argparse
import logging
import sys
from pathlib import Path

import psycopg2

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config import DB_CONFIG, setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

# Groups of existing rows that would collide under candidates_phone_normalized_uniq.
DUPLICATE_PHONE_CHECK = """
    SELECT RIGHT(regexp_replace(phone, '[^0-9]', '', 'g'), 10) AS key,
           COUNT(*) AS n,
           array_agg(id ORDER BY id) AS ids
    FROM candidates
    WHERE phone IS NOT NULL
      AND LENGTH(regexp_replace(phone, '[^0-9]', '', 'g')) >= 10
      -- Must apply the same placeholder exclusions the migration's backfill
      -- does, or this reports collisions on numbers that will never be indexed.
      AND RIGHT(regexp_replace(phone, '[^0-9]', '', 'g'), 10) !~ '^(.)\\1{9}$'
      AND RIGHT(regexp_replace(phone, '[^0-9]', '', 'g'), 10) <> '5555555555'
      AND RIGHT(regexp_replace(phone, '[^0-9]', '', 'g'), 10) !~ '55501[0-9][0-9]$'
    GROUP BY 1
    HAVING COUNT(*) > 1
    ORDER BY n DESC;
"""


def find_conflicts(cursor) -> list:
    """Return duplicate phone groups that would break the unique index."""
    cursor.execute(DUPLICATE_PHONE_CHECK)
    return cursor.fetchall()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sql_file", type=Path)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Apply even if duplicate phone numbers exist (the DDL will still fail).",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Report conflicts and exit without applying anything.",
    )
    args = parser.parse_args()

    setup_logging()

    sql_path = args.sql_file if args.sql_file.is_absolute() else REPO_ROOT / args.sql_file
    if not sql_path.exists():
        logger.error("Migration file not found: %s", sql_path)
        sys.exit(1)

    conn = psycopg2.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cursor:
            conflicts = find_conflicts(cursor)

        if conflicts:
            logger.error(
                "%d phone number(s) appear on more than one row. The unique index "
                "cannot be created until these are merged:", len(conflicts)
            )
            for key, count, ids in conflicts[:20]:
                logger.error("  ...%s -> %d rows (ids: %s)", key[-4:], count, list(ids))
            if len(conflicts) > 20:
                logger.error("  ... and %d more", len(conflicts) - 20)
            if not args.force:
                logger.error("Refusing to apply. Re-run with --force to try anyway.")
                sys.exit(2)
        else:
            logger.info("No duplicate phone numbers found; index is safe to create.")

        if args.check_only:
            logger.info("--check-only set; nothing applied.")
            return

        logger.info("Applying %s", sql_path.name)
        with conn.cursor() as cursor:
            cursor.execute(sql_path.read_text(encoding="utf-8"))
        conn.commit()
        logger.info("Migration applied.")

    except Exception:
        conn.rollback()
        logger.exception("Migration failed; rolled back.")
        sys.exit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
