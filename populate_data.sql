-- Populate sample data for AI Trading Advisor
-- This script inserts sample data into all tables

-- Clear existing data (optional)
TRUNCATE TABLE market_data, news_data, analysis_results, recommendations CASCADE;

-- Insert sample market data
INSERT INTO market_data (symbol, price, volume, timestamp, execution_date) VALUES
('AAPL', 175.50, 1000000, NOW(), CURRENT_DATE),
('GOOGL', 2750.30, 500000, NOW(), CURRENT_DATE),
('MSFT', 380.75, 750000, NOW(), CURRENT_DATE),
('TSLA', 220.45, 2000000, NOW(), CURRENT_DATE),
('AMZN', 145.80, 800000, NOW(), CURRENT_DATE);

-- Insert sample news data
INSERT INTO news_data (title, content, sentiment, timestamp, execution_date) VALUES
('Apple Reports Strong Q4 Earnings', 'Apple Inc. reported strong quarterly earnings with revenue beating expectations...', 0.75, NOW(), CURRENT_DATE),
('Google AI Breakthrough Announced', 'Google announces major breakthrough in artificial intelligence research...', 0.60, NOW(), CURRENT_DATE),
('Microsoft Cloud Growth Continues', 'Microsoft Azure continues to show strong growth in cloud market share...', 0.55, NOW(), CURRENT_DATE),
('Tesla Production Targets Met', 'Tesla successfully meets quarterly production targets for Model Y...', 0.45, NOW(), CURRENT_DATE),
('Amazon Prime Day Success', 'Amazon Prime Day breaks sales records with billions in sales...', 0.65, NOW(), CURRENT_DATE);

-- Insert sample analysis results
INSERT INTO analysis_results (symbol, analysis_type, results, timestamp, execution_date) VALUES
('AAPL', 'technical', '{"rsi": 65.5, "macd": "bullish", "sma_20": 172.30, "recommendation": "buy"}', NOW(), CURRENT_DATE),
('GOOGL', 'technical', '{"rsi": 58.2, "macd": "neutral", "sma_20": 2745.80, "recommendation": "hold"}', NOW(), CURRENT_DATE),
('MSFT', 'fundamental', '{"pe_ratio": 28.5, "revenue_growth": 0.15, "debt_ratio": 0.25, "recommendation": "buy"}', NOW(), CURRENT_DATE),
('TSLA', 'sentiment', '{"overall_sentiment": 0.45, "news_count": 15, "social_sentiment": 0.35, "recommendation": "neutral"}', NOW(), CURRENT_DATE),
('AMZN', 'risk', '{"volatility": 0.28, "beta": 1.15, "var_95": 0.045, "risk_score": "medium"}', NOW(), CURRENT_DATE);

-- Insert sample recommendations
INSERT INTO recommendations (symbol, action, confidence, position_size, risk_level, timestamp, execution_date) VALUES
('AAPL', 'BUY', 0.85, 1000.00, 'MEDIUM', NOW(), CURRENT_DATE),
('GOOGL', 'HOLD', 0.60, 0.00, 'LOW', NOW(), CURRENT_DATE),
('MSFT', 'BUY', 0.75, 1500.00, 'MEDIUM', NOW(), CURRENT_DATE),
('TSLA', 'SELL', 0.70, 500.00, 'HIGH', NOW(), CURRENT_DATE),
('AMZN', 'HOLD', 0.55, 0.00, 'LOW', NOW(), CURRENT_DATE);

-- Display confirmation
SELECT 
    'market_data' as table_name, 
    COUNT(*) as record_count 
FROM market_data
UNION ALL
SELECT 
    'news_data' as table_name, 
    COUNT(*) as record_count 
FROM news_data
UNION ALL
SELECT 
    'analysis_results' as table_name, 
    COUNT(*) as record_count 
FROM analysis_results
UNION ALL
SELECT 
    'recommendations' as table_name, 
    COUNT(*) as record_count 
FROM recommendations
ORDER BY table_name;