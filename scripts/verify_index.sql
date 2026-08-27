-- Verification queries for the CareDesk document/chunk index.
--
-- Run after `scripts/index_corpus.py` to sanity-check what actually landed
-- in Postgres. Read-only; safe to run at any time.
--
--   psql "$DATABASE_URL" -f scripts/verify_index.sql

-- 1. Row counts per table
SELECT 'documents' AS table_name, count(*) AS row_count FROM documents
UNION ALL
SELECT 'chunks', count(*) FROM chunks;

-- 2. Chunks per source_type
SELECT source_type, count(*) AS chunk_count
FROM chunks
GROUP BY source_type
ORDER BY source_type;

-- 3. Chunks per persona_visibility
SELECT persona_visibility, count(*) AS chunk_count
FROM chunks
GROUP BY persona_visibility
ORDER BY persona_visibility;

-- 4. Any document with zero chunks (should be empty -- every indexed
--    document should have produced at least one chunk)
SELECT documents.doc_id, documents.title, documents.source_type
FROM documents
LEFT JOIN chunks ON chunks.doc_id = documents.doc_id
WHERE chunks.chunk_id IS NULL;

-- 5. Any chunk with a null or zero-norm embedding (should be empty --
--    a zero-norm vector indicates a failed/placeholder embedding, since a
--    real embedding is never exactly the zero vector)
SELECT chunk_id, doc_id
FROM chunks
WHERE embedding IS NULL OR vector_norm(embedding) = 0;

-- 6. Sample nearest-neighbour query
--    Uses an arbitrary existing chunk's own embedding as the fixed query
--    vector, so this is runnable without retrieval/embedding code (out of
--    scope for this commit) and without hand-writing a 1536-dimension
--    literal. The top result should be the chunk itself at distance 0 --
--    a basic sanity check that cosine ordering and the HNSW index work.
WITH query_vector AS (
    SELECT embedding FROM chunks ORDER BY chunk_id LIMIT 1
)
SELECT
    chunks.chunk_id,
    chunks.doc_id,
    chunks.embedding <=> query_vector.embedding AS cosine_distance
FROM chunks, query_vector
ORDER BY chunks.embedding <=> query_vector.embedding
LIMIT 5;
