"""
Database access and the idempotent candidate write.

All writes to the candidates table go through upsert_candidate(). That matters
more than usual here: with acks_late=True a Celery task can be redelivered after
it has already written a row, so a plain INSERT would duplicate a candidate
every time a worker died mid-task. The upsert makes the write safe to repeat.

Three uniqueness rules back it, in priority order:

  1. phone_normalized -- primary deduplication. The same person appearing in
     two different resumes. Phone is the reliably-maintained identifier in this
     labour market; email frequently is not.

  2. email -- secondary. `candidates_email_key` already existed on the table
     before any of this was designed. A collision there used to crash the task
     with a UniqueViolation that retried to exhaustion, because ON CONFLICT
     accepts only one target and that target was phone. Handled explicitly now.

  3. (batch_id, source_file) -- retry safety only, for rows with no usable
     phone. Not identity: it exists so a redelivered task overwrites its own
     earlier row rather than adding a second.

Rules 1 and 3 are enforced by partial unique indexes with deliberately disjoint
predicates (phone_normalized IS NOT NULL vs IS NULL), so no row can violate
both. Rule 2 is a pre-existing full unique constraint and can fire alongside
either, which is why the email fallback is a savepoint retry rather than a
fourth ON CONFLICT target. See migrations/001_phone_dedup.sql.
"""

from __future__ import annotations

import logging
import re
from contextlib import contextmanager
from typing import Optional

import psycopg2
from psycopg2 import errors as pg_errors

from config import DB_CONFIG

logger = logging.getLogger(__name__)

# North American numbers are 10 significant digits; anything shorter is a
# fragment (an extension, a partial OCR read) and is not a usable identity.
PHONE_SIGNIFICANT_DIGITS = 10

# The unique constraint that already existed on the table.
EMAIL_CONSTRAINT = "candidates_email_key"

INSERT_COLUMNS = (
    "name", "raw_title", "experience_years", "skills",
    "phone", "phone_normalized", "email", "embedding", "source_file", "batch_id",
)

_UPDATE_CLAUSE = """
    name = EXCLUDED.name,
    raw_title = EXCLUDED.raw_title,
    experience_years = EXCLUDED.experience_years,
    skills = EXCLUDED.skills,
    phone = EXCLUDED.phone,
    phone_normalized = EXCLUDED.phone_normalized,
    email = EXCLUDED.email,
    embedding = EXCLUDED.embedding,
    source_file = EXCLUDED.source_file,
    batch_id = EXCLUDED.batch_id
"""

_BASE_INSERT = """
    INSERT INTO candidates
        (name, raw_title, experience_years, skills,
         phone, phone_normalized, email, embedding, source_file, batch_id)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector, %s, %s)
"""

# Same person, seen again. Merge onto the existing row.
SQL_UPSERT_ON_PHONE = _BASE_INSERT + f"""
    ON CONFLICT (phone_normalized) WHERE phone_normalized IS NOT NULL
    DO UPDATE SET {_UPDATE_CLAUSE}
"""

# No usable phone, so we cannot dedupe by identity -- but we can still make a
# redelivered task overwrite its own previous row instead of adding a second.
SQL_UPSERT_ON_BATCH_FILE = _BASE_INSERT + f"""
    ON CONFLICT (batch_id, source_file)
        WHERE phone_normalized IS NULL AND batch_id IS NOT NULL
    DO UPDATE SET {_UPDATE_CLAUSE}
"""

# Fallback when the pre-existing email constraint fires.
SQL_UPSERT_ON_EMAIL = _BASE_INSERT + f"""
    ON CONFLICT (email) DO UPDATE SET {_UPDATE_CLAUSE}
"""

# CLI path: no batch id, no phone. Nothing to key on, so this insert is not
# idempotent. Callers running without a batch id accept that.
SQL_PLAIN_INSERT = _BASE_INSERT

_SAVEPOINT = "upsert_candidate"


def is_placeholder_phone(digits: str) -> bool:
    """
    True for numbers that are not a real person's identity.

    This is a correctness guard, not tidiness. Phone is the *primary* dedup
    signal, so a number that many unrelated people share silently merges them
    into one row and destroys the losers' data. That is not hypothetical: a
    production query found five distinct candidates -- Hiro Huang, Aya Nakamura,
    Ming Davis, David Anderson and Fred Hill -- all carrying (555) 555-5555 from
    resume templates. Under phone dedup they would have collapsed to a single
    row, with four people's records overwritten.

    Blocked:
      * 555-0100 to 555-0199, the range reserved for fiction, under any area code
      * the literal 5555555555
      * any number that is a single repeated digit (0000000000, 1111111111, ...)
    """
    if len(set(digits)) == 1:
        return True
    if digits == "5555555555":
        return True

    # Subscriber number is the last 7 digits: exchange (3) + line (4).
    subscriber = digits[-7:]
    if subscriber.startswith("555"):
        line = subscriber[3:]
        if line.isdigit() and 100 <= int(line) <= 199:
            return True
    return False


def normalize_phone(raw: Optional[str]) -> Optional[str]:
    """
    Reduce a phone number to a comparable identity: its last 10 digits.

    Strips formatting and country code so that "+1 (918) 246-8135",
    "918-246-8135" and "9182468135" all collapse to the same key.

    Returns None when the number cannot serve as an identity: too few digits to
    be real, or a known placeholder (see is_placeholder_phone). Callers treat
    None as "no usable phone" and fall through to the (batch_id, source_file)
    key, which gives retry safety without ever claiming two people are one.
    """
    if not raw:
        return None
    digits = re.sub(r"\D", "", raw)
    if len(digits) < PHONE_SIGNIFICANT_DIGITS:
        return None

    digits = digits[-PHONE_SIGNIFICANT_DIGITS:]
    if is_placeholder_phone(digits):
        logger.debug("Ignoring placeholder phone %s for dedup purposes.", digits)
        return None
    return digits


@contextmanager
def connection():
    """Context-managed psycopg2 connection that always closes."""
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        yield conn
    finally:
        conn.close()


def _build_params(record: dict, phone_normalized: Optional[str],
                  batch_id: Optional[str]) -> tuple:
    return (
        record.get("name"),
        record.get("raw_title"),
        record.get("experience_years"),
        record.get("skills"),
        record.get("phone"),
        phone_normalized,
        record.get("email"),
        str(record["embedding"]),
        record.get("source_file"),
        batch_id,
    )


def upsert_candidate(cursor, record: dict, batch_id: Optional[str] = None) -> str:
    """
    Write one candidate idempotently. Returns the conflict strategy used.

    Takes a cursor rather than opening its own connection so a caller can batch
    several writes in one transaction, and so the Celery task controls commit
    boundaries.

    On conflict the row is *updated*, not skipped: a candidate re-applying with
    a newer resume must end up with their new title, experience and skills, not
    silently keep the stale ones. Most recent upload wins.

    The email fallback uses a savepoint because ON CONFLICT accepts exactly one
    target. A row can satisfy the phone rule and still collide on the
    pre-existing email constraint, and without the savepoint that error would
    abort the surrounding transaction rather than being recoverable.
    """
    phone_normalized = normalize_phone(record.get("phone"))
    params = _build_params(record, phone_normalized, batch_id)

    if phone_normalized:
        primary_sql, strategy = SQL_UPSERT_ON_PHONE, "phone"
    elif batch_id:
        primary_sql, strategy = SQL_UPSERT_ON_BATCH_FILE, "batch_file"
    else:
        logger.warning(
            "Inserting %s with neither phone nor batch id; write is not idempotent.",
            record.get("source_file"),
        )
        primary_sql, strategy = SQL_PLAIN_INSERT, "plain"

    cursor.execute(f"SAVEPOINT {_SAVEPOINT}")
    try:
        cursor.execute(primary_sql, params)
    except pg_errors.UniqueViolation as exc:
        constraint = getattr(getattr(exc, "diag", None), "constraint_name", None)
        if constraint != EMAIL_CONSTRAINT or not record.get("email"):
            # Not the email constraint, or nothing to key on. Let it propagate;
            # the task will retry and, if it is genuinely stuck, fail loudly.
            cursor.execute(f"ROLLBACK TO SAVEPOINT {_SAVEPOINT}")
            raise

        # A different row already owns this email. Update that row instead of
        # crashing -- previously this raised and retried to exhaustion.
        logger.info(
            "Email collision on %s for %s; updating the existing record.",
            record.get("email"), record.get("source_file"),
        )
        cursor.execute(f"ROLLBACK TO SAVEPOINT {_SAVEPOINT}")
        cursor.execute(SQL_UPSERT_ON_EMAIL, params)
        strategy = "email"

    cursor.execute(f"RELEASE SAVEPOINT {_SAVEPOINT}")
    return strategy
