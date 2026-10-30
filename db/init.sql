CREATE EXTENSION IF NOT EXISTS vector;

-- One row per paper (metadata used for filters)
CREATE TABLE IF NOT EXISTS papers (
    arxiv_id   TEXT PRIMARY KEY,
    title      TEXT NOT NULL,
    authors    TEXT,
    abstract   TEXT,
    category   TEXT,
    published  DATE,
    pdf_url    TEXT
);

-- One row per chunk of text (what we actually search)
CREATE TABLE IF NOT EXISTS chunks (
    id          SERIAL PRIMARY KEY,
    arxiv_id    TEXT NOT NULL REFERENCES papers(arxiv_id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content     TEXT NOT NULL,
    embedding   vector(384),
    content_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
);

-- Vector index (semantic search) and full-text index (keyword search)
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx
    ON chunks USING gin (content_tsv);
CREATE INDEX IF NOT EXISTS papers_category_idx ON papers (category);
CREATE INDEX IF NOT EXISTS papers_published_idx ON papers (published);