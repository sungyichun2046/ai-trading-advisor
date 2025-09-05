-- Simple database initialization for AI Trading Advisor
-- Focus on core tables needed for trigger-and-wait

-- Market data table
CREATE TABLE IF NOT EXISTS market_data (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    price DECIMAL(10,4) NOT NULL,
    volume BIGINT,
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    source VARCHAR(50),
    execution_date DATE DEFAULT CURRENT_DATE,
    created_at TIMESTAMP DEFAULT NOW()
);

-- News data table
CREATE TABLE IF NOT EXISTS news_data (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT,
    sentiment DECIMAL(3,2), -- -1.00 to 1.00
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    execution_date DATE DEFAULT CURRENT_DATE,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Analysis results table
CREATE TABLE IF NOT EXISTS analysis_results (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    analysis_type VARCHAR(50) NOT NULL, -- technical, fundamental, sentiment, risk
    results JSONB NOT NULL,
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    execution_date DATE DEFAULT CURRENT_DATE,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Trading recommendations table
CREATE TABLE IF NOT EXISTS recommendations (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    action VARCHAR(10) NOT NULL, -- BUY, SELL, HOLD
    confidence DECIMAL(3,2) NOT NULL, -- 0.00 to 1.00
    position_size DECIMAL(10,2),
    target_price DECIMAL(10,4),
    stop_loss DECIMAL(10,4),
    reasoning TEXT,
    risk_level VARCHAR(20) NOT NULL, -- LOW, MEDIUM, HIGH
    timeframe VARCHAR(20) DEFAULT 'INTRADAY', -- INTRADAY, SHORT_TERM, LONG_TERM
    timestamp TIMESTAMP DEFAULT NOW(),
    execution_date DATE DEFAULT CURRENT_DATE,
    created_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP
);

-- Create basic indexes
CREATE INDEX IF NOT EXISTS idx_market_data_symbol_timestamp ON market_data(symbol, timestamp);
CREATE INDEX IF NOT EXISTS idx_market_data_execution_date ON market_data(execution_date);
CREATE INDEX IF NOT EXISTS idx_news_data_execution_date ON news_data(execution_date);
CREATE INDEX IF NOT EXISTS idx_news_data_timestamp ON news_data(timestamp);
CREATE INDEX IF NOT EXISTS idx_analysis_results_symbol_type ON analysis_results(symbol, analysis_type);
CREATE INDEX IF NOT EXISTS idx_analysis_results_execution_date ON analysis_results(execution_date);
CREATE INDEX IF NOT EXISTS idx_recommendations_symbol_created ON recommendations(symbol, created_at);
CREATE INDEX IF NOT EXISTS idx_recommendations_execution_date ON recommendations(execution_date);

-- Insert sample data immediately
INSERT INTO market_data (symbol, price, volume, timestamp, execution_date) VALUES
('AAPL', 175.50, 1000000, NOW(), CURRENT_DATE),
('GOOGL', 2750.30, 500000, NOW(), CURRENT_DATE),
('MSFT', 380.75, 750000, NOW(), CURRENT_DATE),
('TSLA', 220.45, 2000000, NOW(), CURRENT_DATE),
('AMZN', 145.80, 800000, NOW(), CURRENT_DATE)
ON CONFLICT DO NOTHING;

INSERT INTO news_data (title, content, sentiment, timestamp, execution_date) VALUES
('Apple Q4 Earnings Beat', 'Strong quarterly results exceed expectations', 0.75, NOW(), CURRENT_DATE),
('Google AI Breakthrough', 'Major advancement in machine learning', 0.60, NOW(), CURRENT_DATE),
('Microsoft Cloud Growth', 'Azure continues market expansion', 0.55, NOW(), CURRENT_DATE),
('Tesla Production Target', 'Manufacturing goals successfully met', 0.45, NOW(), CURRENT_DATE),
('Amazon Prime Success', 'Record-breaking sales numbers achieved', 0.65, NOW(), CURRENT_DATE)
ON CONFLICT DO NOTHING;

INSERT INTO analysis_results (symbol, analysis_type, results, timestamp, execution_date) VALUES
('AAPL', 'technical', '{"rsi": 65.5, "macd": "bullish", "recommendation": "buy"}', NOW(), CURRENT_DATE),
('GOOGL', 'technical', '{"rsi": 58.2, "macd": "neutral", "recommendation": "hold"}', NOW(), CURRENT_DATE),
('MSFT', 'fundamental', '{"pe_ratio": 28.5, "revenue_growth": 0.15, "recommendation": "buy"}', NOW(), CURRENT_DATE),
('TSLA', 'sentiment', '{"sentiment": 0.45, "news_count": 15, "recommendation": "neutral"}', NOW(), CURRENT_DATE),
('AMZN', 'risk', '{"volatility": 0.28, "beta": 1.15, "risk_score": "medium"}', NOW(), CURRENT_DATE)
ON CONFLICT DO NOTHING;

INSERT INTO recommendations (symbol, action, confidence, position_size, risk_level, timestamp, execution_date) VALUES
('AAPL', 'BUY', 0.85, 1000.00, 'MEDIUM', NOW(), CURRENT_DATE),
('GOOGL', 'HOLD', 0.60, 0.00, 'LOW', NOW(), CURRENT_DATE),
('MSFT', 'BUY', 0.75, 1500.00, 'MEDIUM', NOW(), CURRENT_DATE),
('TSLA', 'SELL', 0.70, 500.00, 'HIGH', NOW(), CURRENT_DATE),
('AMZN', 'HOLD', 0.55, 0.00, 'LOW', NOW(), CURRENT_DATE)
ON CONFLICT DO NOTHING;