-- ============================================================
-- trading_simulator_schema.sql
-- Clean, idempotent schema for trading_workflow (MD 1)
-- ============================================================

-- ============================================================
-- 1. Master Symbols (union of all users' interests)
-- ============================================================
CREATE TABLE IF NOT EXISTS active_symbols (
    symbol VARCHAR(10) PRIMARY KEY,
    added_by_users TEXT[] NOT NULL,
    last_updated TIMESTAMP NOT NULL,
    is_active BOOLEAN DEFAULT TRUE
);

CREATE INDEX IF NOT EXISTS idx_active_symbols_users
ON active_symbols USING GIN (added_by_users);

-- ============================================================
-- 2. User Profiles
-- ============================================================
CREATE TABLE IF NOT EXISTS user_profiles (
    user_id VARCHAR(50) PRIMARY KEY,
    budget DECIMAL(12,2) NOT NULL,
    risk_tolerance VARCHAR(20) NOT NULL,
    trading_style VARCHAR(20),
    interested_symbols TEXT[] NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- ============================================================
-- 3. Market Data (shared across all users)
-- ============================================================
CREATE TABLE IF NOT EXISTS market_data (
    id SERIAL PRIMARY KEY,
    run_timestamp TIMESTAMP NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    price DECIMAL(10,4) NOT NULL,
    volume BIGINT,
    open_price DECIMAL(10,4),
    high_price DECIMAL(10,4),
    low_price DECIMAL(10,4),
    close_price DECIMAL(10,4),
    market_cap BIGINT,
    pe_ratio DECIMAL(6,2),
    data_source VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW(),
    CONSTRAINT fk_market_data_symbol
        FOREIGN KEY (symbol)
        REFERENCES active_symbols(symbol)
);

CREATE INDEX IF NOT EXISTS idx_market_data_run_symbol
ON market_data (run_timestamp, symbol);

-- ============================================================
-- 4. Technical Analysis Results (shared across users)
-- ============================================================
CREATE TABLE IF NOT EXISTS technical_analysis (
    id SERIAL PRIMARY KEY,
    run_timestamp TIMESTAMP NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    rsi DECIMAL(5,2),
    macd_value DECIMAL(8,4),
    macd_signal DECIMAL(8,4),
    macd_histogram DECIMAL(8,4),
    bb_upper DECIMAL(10,4),
    bb_middle DECIMAL(10,4),
    bb_lower DECIMAL(10,4),
    signal VARCHAR(10),
    confidence DECIMAL(3,2),
    created_at TIMESTAMP DEFAULT NOW(),
    CONSTRAINT fk_technical_analysis_symbol
        FOREIGN KEY (symbol)
        REFERENCES active_symbols(symbol)
);

CREATE INDEX IF NOT EXISTS idx_technical_analysis_run_symbol
ON technical_analysis (run_timestamp, symbol);

-- ============================================================
-- 5. Sentiment Analysis (shared across users)
-- ============================================================
CREATE TABLE IF NOT EXISTS sentiment_analysis (
    id SERIAL PRIMARY KEY,
    run_timestamp TIMESTAMP NOT NULL,
    symbol VARCHAR(10),
    sentiment_score DECIMAL(4,3),
    sentiment_label VARCHAR(20),
    confidence DECIMAL(3,2),
    article_count INTEGER,
    data_source VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW(),
    CONSTRAINT fk_sentiment_analysis_symbol
        FOREIGN KEY (symbol)
        REFERENCES active_symbols(symbol)
);

CREATE INDEX IF NOT EXISTS idx_sentiment_analysis_run_symbol
ON sentiment_analysis (run_timestamp, symbol);

-- ============================================================
-- 6. Trading Decisions (personalized per user)
-- ============================================================
CREATE TABLE IF NOT EXISTS trading_decisions (
    id SERIAL PRIMARY KEY,
    run_timestamp TIMESTAMP NOT NULL,
    user_id VARCHAR(50) NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    action VARCHAR(10),
    recommended_quantity INTEGER,
    recommended_price DECIMAL(10,4),
    confidence DECIMAL(3,2),
    reasoning TEXT,
    budget_allocated DECIMAL(12,2),
    executed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW(),
    CONSTRAINT fk_trading_decisions_user
        FOREIGN KEY (user_id)
        REFERENCES user_profiles(user_id),
    CONSTRAINT fk_trading_decisions_symbol
        FOREIGN KEY (symbol)
        REFERENCES active_symbols(symbol)
);

CREATE INDEX IF NOT EXISTS idx_trading_decisions_user_run
ON trading_decisions (user_id, run_timestamp);

-- ============================================================
-- 7. DAG Execution Tracking
-- ============================================================
CREATE TABLE IF NOT EXISTS dag_runs (
    run_timestamp TIMESTAMP PRIMARY KEY,
    dag_status VARCHAR(20),
    symbols_processed INTEGER,
    users_served INTEGER,
    execution_time_ms INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);

-- ============================================================
-- 8. TTL Cleanup Function (NOT auto-executed)
-- ============================================================
-- NOTE:
-- This function must be invoked explicitly by a scheduled job
-- or Airflow DAG. It is NOT triggered automatically.
CREATE OR REPLACE FUNCTION cleanup_expired_data()
RETURNS void AS $$
BEGIN
    DELETE FROM market_data
     WHERE created_at < NOW() - INTERVAL '30 days';

    DELETE FROM technical_analysis
     WHERE created_at < NOW() - INTERVAL '30 days';

    DELETE FROM sentiment_analysis
     WHERE created_at < NOW() - INTERVAL '7 days';

    DELETE FROM trading_decisions
     WHERE created_at < NOW() - INTERVAL '90 days';

    DELETE FROM dag_runs
     WHERE created_at < NOW() - INTERVAL '365 days';
END;
$$ LANGUAGE plpgsql;
