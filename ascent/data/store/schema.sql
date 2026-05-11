-- ascent/data/store/schema.sql
-- TimescaleDB schema for Ascent Capital real-time infrastructure.
-- Run via: psql $TIMESCALEDB_URL -f schema.sql

-- 1-minute price bars
CREATE TABLE IF NOT EXISTS prices_1m (
    time        TIMESTAMPTZ     NOT NULL,
    symbol      TEXT            NOT NULL,
    open        DOUBLE PRECISION,
    high        DOUBLE PRECISION,
    low         DOUBLE PRECISION,
    close       DOUBLE PRECISION,
    volume      BIGINT
);
SELECT create_hypertable('prices_1m', 'time', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 week');
CREATE UNIQUE INDEX IF NOT EXISTS prices_1m_symbol_time_idx ON prices_1m (symbol, time DESC);
SELECT add_compression_policy('prices_1m', INTERVAL '7 days', if_not_exists => TRUE);
SELECT add_retention_policy('prices_1m', INTERVAL '90 days', if_not_exists => TRUE);

-- Daily adjusted price bars
CREATE TABLE IF NOT EXISTS prices_1d (
    time        TIMESTAMPTZ     NOT NULL,
    symbol      TEXT            NOT NULL,
    open        DOUBLE PRECISION,
    high        DOUBLE PRECISION,
    low         DOUBLE PRECISION,
    close       DOUBLE PRECISION,
    volume      BIGINT,
    adj_close   DOUBLE PRECISION
);
SELECT create_hypertable('prices_1d', 'time', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 month');
CREATE UNIQUE INDEX IF NOT EXISTS prices_1d_symbol_time_idx ON prices_1d (symbol, time DESC);

-- Alpha scores by sleeve
CREATE TABLE IF NOT EXISTS factor_scores (
    time        TIMESTAMPTZ     NOT NULL,
    agent_id    TEXT            NOT NULL,
    symbol      TEXT            NOT NULL,
    sleeve      TEXT            NOT NULL,
    score       DOUBLE PRECISION
);
SELECT create_hypertable('factor_scores', 'time', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 month');

-- Portfolio state (weights per agent)
CREATE TABLE IF NOT EXISTS portfolio_state (
    time        TIMESTAMPTZ     NOT NULL,
    agent_id    TEXT            NOT NULL,
    symbol      TEXT            NOT NULL,
    weight      DOUBLE PRECISION,
    alpha_score DOUBLE PRECISION
);
SELECT create_hypertable('portfolio_state', 'time', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 month');

-- NAV series
CREATE TABLE IF NOT EXISTS nav_series (
    time         TIMESTAMPTZ     NOT NULL,
    agent_id     TEXT            NOT NULL,
    nav          DOUBLE PRECISION,
    daily_return DOUBLE PRECISION,
    spy_return   DOUBLE PRECISION
);
SELECT create_hypertable('nav_series', 'time', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 month');
CREATE UNIQUE INDEX IF NOT EXISTS nav_series_agent_time_idx ON nav_series (agent_id, time DESC);

-- Factor exposures
CREATE TABLE IF NOT EXISTS factor_exposures (
    time                TIMESTAMPTZ     NOT NULL,
    factor              TEXT            NOT NULL,
    exposure            DOUBLE PRECISION,
    variance_explained  DOUBLE PRECISION
);
SELECT create_hypertable('factor_exposures', 'time', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 month');

-- Continuous aggregate: daily OHLCV from 1-minute bars
CREATE MATERIALIZED VIEW IF NOT EXISTS prices_1d_from_1m
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', time) AS bucket,
    symbol,
    FIRST(open,  time) AS open,
    MAX(high)          AS high,
    MIN(low)           AS low,
    LAST(close,  time) AS close,
    SUM(volume)        AS volume
FROM prices_1m
GROUP BY bucket, symbol
WITH NO DATA;
