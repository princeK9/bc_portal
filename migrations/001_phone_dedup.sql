-- 001_phone_dedup.sql
--
-- Adds the columns and uniqueness rules that make candidate writes idempotent.
-- Required before Celery: with acks_late=True a task can be redelivered after
-- it has already written its row, and a plain INSERT would duplicate.
--
-- Apply with:  python scripts/apply_migration.py migrations/001_phone_dedup.sql
-- The script refuses to create the indexes if existing data would violate them,
-- and prints the offending rows instead.

BEGIN;

-- Comparable form of the phone number: last 10 digits, formatting stripped.
ALTER TABLE candidates ADD COLUMN IF NOT EXISTS phone_normalized text;

-- Which upload batch produced the row. NULL for rows written by the CLI path.
ALTER TABLE candidates ADD COLUMN IF NOT EXISTS batch_id text;

-- Backfill existing rows. RIGHT(...) mirrors normalize_phone() in db.py:
-- strip non-digits, keep the last 10, and ignore anything too short to be real.
-- Placeholder numbers are deliberately left NULL. They are not identities:
-- five distinct production candidates all carry (555) 555-5555 from resume
-- templates, and treating that as a dedup key would merge them into one row.
-- These three predicates mirror db.is_placeholder_phone() exactly; if one side
-- changes the other must too, or the backfill and the application disagree.
UPDATE candidates
SET phone_normalized = RIGHT(regexp_replace(phone, '[^0-9]', '', 'g'), 10)
WHERE phone IS NOT NULL
  AND phone_normalized IS NULL
  AND LENGTH(regexp_replace(phone, '[^0-9]', '', 'g')) >= 10
  -- not a single repeated digit (0000000000, 1111111111, ...)
  AND RIGHT(regexp_replace(phone, '[^0-9]', '', 'g'), 10) !~ '^(.)\1{9}$'
  -- not the literal 5555555555
  AND RIGHT(regexp_replace(phone, '[^0-9]', '', 'g'), 10) <> '5555555555'
  -- not the 555-0100..555-0199 range reserved for fiction, any area code
  AND RIGHT(regexp_replace(phone, '[^0-9]', '', 'g'), 10) !~ '55501[0-9][0-9]$';

-- Rule 1 -- deduplication: one row per real person, keyed on phone.
-- Partial, so the many rows with no extractable phone do not all collide.
CREATE UNIQUE INDEX IF NOT EXISTS candidates_phone_normalized_uniq
    ON candidates (phone_normalized)
    WHERE phone_normalized IS NOT NULL;

-- Rule 2 -- retry safety for rows rule 1 cannot cover. A redelivered task
-- overwrites its own earlier row instead of adding a second one.
-- The predicate is the exact complement of rule 1's, so the two indexes are
-- disjoint and no row can violate both.
CREATE UNIQUE INDEX IF NOT EXISTS candidates_batch_file_uniq
    ON candidates (batch_id, source_file)
    WHERE phone_normalized IS NULL AND batch_id IS NOT NULL;

-- Supports the /search filter and any dedup reporting.
CREATE INDEX IF NOT EXISTS candidates_batch_id_idx ON candidates (batch_id);

COMMIT;
