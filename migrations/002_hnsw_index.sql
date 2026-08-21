-- 002_hnsw_index.sql
--
-- Approximate-nearest-neighbour index on the embedding column, so /search stops
-- scanning every row. Verified against pgvector 0.8.6 on PostgreSQL 16:
-- `hnsw` is a registered access method and `vector_cosine_ops` is a real
-- operator class there. Both were checked in pg_am / pg_opclass before this
-- file was written rather than assumed from documentation.
--
-- Apply with:  python scripts/apply_migration.py migrations/002_hnsw_index.sql
--
-- Why a separate migration rather than editing 000_schema.sql: 000 only ever
-- runs when a brand-new database is bootstrapped, so an index added there would
-- never reach an existing deployment. Production already exists and needs this
-- applied incrementally, which is exactly what the 001 pattern is for. 000 is
-- kept in sync for fresh test databases by mounting this file alongside it in
-- docker-compose.test.yml.

BEGIN;

-- vector_cosine_ops must match the operator used by the query. server.py orders
-- by `embedding <=> query`, and <=> is cosine distance; an l2 or ip opclass
-- would build a perfectly valid index that the planner then ignores, which
-- looks like "the index did nothing" rather than like a mismatch.
--
-- Left at pgvector's defaults (m=16, ef_construction=64). Measured at 99,387
-- rows those give a 143MB index, a ~78s single-threaded build, and a drop in
-- server-side execution from 66.9ms to 0.98ms. Tuning m/ef_construction trades
-- build time and index size for recall; there is no recall problem to solve
-- yet, so the knobs stay untouched rather than being set to look deliberate.
--
-- Not CONCURRENTLY: that cannot run inside a transaction block, and the table
-- it is being applied to is small. On a large live table, run it separately as
-- CREATE INDEX CONCURRENTLY to avoid blocking writes for the build duration.
CREATE INDEX IF NOT EXISTS candidates_embedding_hnsw_idx
    ON candidates USING hnsw (embedding vector_cosine_ops);

COMMIT;

-- Build note, learned the hard way: a parallel index build allocates a shared
-- memory segment, and Docker's default 64MB /dev/shm is far too small for it.
-- The first attempt died with
--   "could not resize shared memory segment ... No space left on device".
-- Fixed by giving the postgres service shm_size: 1gb in docker-compose.test.yml.
-- Without that, either raise shm_size or build with
--   SET max_parallel_maintenance_workers = 0;
-- which succeeds but is materially slower.
