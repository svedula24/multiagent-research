CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS customer_reviews (
    id          SERIAL PRIMARY KEY,
    content     TEXT NOT NULL,
    rating      INTEGER CHECK (rating BETWEEN 1 AND 5),
    category    VARCHAR(50),
    created_at  TIMESTAMP DEFAULT NOW(),
    embedding   VECTOR(1536)
);

CREATE INDEX IF NOT EXISTS customer_reviews_embedding_idx
    ON customer_reviews USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

CREATE TABLE IF NOT EXISTS sales_reports (
    id                  SERIAL PRIMARY KEY,
    period              VARCHAR(20) NOT NULL,
    revenue             DECIMAL(12,2),
    churn_rate          DECIMAL(5,2),
    new_customers       INTEGER,
    lost_deals_reason   TEXT,
    top_feature         VARCHAR(100),
    nps_score           INTEGER CHECK (nps_score BETWEEN 0 AND 100),
    segment             VARCHAR(50),
    created_at          TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS research_runs (
    id              SERIAL PRIMARY KEY,
    query           TEXT NOT NULL,
    competitors     TEXT[] NOT NULL,
    status          VARCHAR(30) DEFAULT 'running',
    rejection_count INTEGER DEFAULT 0,
    created_at      TIMESTAMP DEFAULT NOW(),
    updated_at      TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS worker_outputs (
    id              SERIAL PRIMARY KEY,
    run_id          INTEGER REFERENCES research_runs(id) ON DELETE CASCADE,
    source          VARCHAR(20) NOT NULL,
    status          VARCHAR(20) NOT NULL,
    findings        JSONB,
    confidence      DECIMAL(4,3),
    error           TEXT,
    retrieved_at    TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS draft_recommendations (
    id              SERIAL PRIMARY KEY,
    run_id          INTEGER REFERENCES research_runs(id) ON DELETE CASCADE,
    summary         TEXT NOT NULL,
    findings_by_source JSONB,
    confidence      DECIMAL(4,3),
    status          VARCHAR(30) DEFAULT 'pending_approval',
    rejection_count INTEGER DEFAULT 0,
    created_at      TIMESTAMP DEFAULT NOW(),
    updated_at      TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS final_recommendations (
    id              SERIAL PRIMARY KEY,
    run_id          INTEGER REFERENCES research_runs(id) ON DELETE CASCADE,
    content         TEXT NOT NULL,
    confidence      DECIMAL(4,3),
    approved_at     TIMESTAMP DEFAULT NOW()
);
