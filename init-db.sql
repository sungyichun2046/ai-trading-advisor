-- Initialize trading advisor database
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    is_active BOOLEAN DEFAULT true,
    risk_tolerance DECIMAL(3,2) DEFAULT 0.02, -- 2% default risk per trade
    max_portfolio_risk DECIMAL(3,2) DEFAULT 0.20, -- 20% max portfolio risk
    daily_loss_limit DECIMAL(3,2) DEFAULT 0.06, -- 6% daily loss limit
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Portfolios table
CREATE TABLE IF NOT EXISTS portfolios (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    total_value DECIMAL(12,2) DEFAULT 0.00,
    cash_balance DECIMAL(12,2) DEFAULT 0.00,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Positions table
CREATE TABLE IF NOT EXISTS positions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    portfolio_id UUID REFERENCES portfolios(id) ON DELETE CASCADE,
    symbol VARCHAR(20) NOT NULL,
    quantity DECIMAL(12,4) NOT NULL,
    avg_cost DECIMAL(10,4) NOT NULL,
    current_price DECIMAL(10,4),
    market_value DECIMAL(12,2),
    unrealized_pnl DECIMAL(12,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Market data table
CREATE TABLE IF NOT EXISTS market_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(20) NOT NULL,
    price DECIMAL(10,4) NOT NULL,
    volume BIGINT,
    timestamp TIMESTAMP NOT NULL,
    source VARCHAR(50),
    execution_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- News data table
CREATE TABLE IF NOT EXISTS news_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title TEXT NOT NULL,
    content TEXT,
    sentiment DECIMAL(3,2),
    timestamp TIMESTAMP NOT NULL,
    execution_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Analysis results table
CREATE TABLE IF NOT EXISTS analysis_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(20) NOT NULL,
    analysis_type VARCHAR(50) NOT NULL,
    results JSONB NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    execution_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Trading recommendations table
CREATE TABLE IF NOT EXISTS recommendations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(20) NOT NULL,
    action VARCHAR(10) NOT NULL,
    confidence DECIMAL(3,2) NOT NULL,
    position_size DECIMAL(10,2),
    target_price DECIMAL(10,4),
    stop_loss DECIMAL(10,4),
    reasoning TEXT,
    risk_level VARCHAR(20) NOT NULL,
    timeframe VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP,
    execution_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_portfolios_user_id ON portfolios(user_id);
CREATE INDEX IF NOT EXISTS idx_positions_portfolio_id ON positions(portfolio_id);
CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol);
CREATE INDEX IF NOT EXISTS idx_market_data_symbol_timestamp ON market_data(symbol, timestamp);
CREATE INDEX IF NOT EXISTS idx_market_data_execution_date ON market_data(execution_date);
CREATE INDEX IF NOT EXISTS idx_news_data_execution_date ON news_data(execution_date);
CREATE INDEX IF NOT EXISTS idx_news_data_timestamp ON news_data(timestamp);
CREATE INDEX IF NOT EXISTS idx_analysis_results_symbol_type ON analysis_results(symbol, analysis_type);
CREATE INDEX IF NOT EXISTS idx_analysis_results_execution_date ON analysis_results(execution_date);
CREATE INDEX IF NOT EXISTS idx_recommendations_symbol_created ON recommendations(symbol, created_at);
CREATE INDEX IF NOT EXISTS idx_recommendations_execution_date ON recommendations(execution_date);