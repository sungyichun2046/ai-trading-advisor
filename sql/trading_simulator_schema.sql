-- =====================================================================
-- Multi-Profile Trading Simulator Database Schema
-- =====================================================================
-- Database: Supabase PostgreSQL (500MB free tier)
-- Purpose: Support multiple trading profiles with different styles
-- Features: User profiles, portfolios, trades, market data, performance tracking
-- Storage: Optimized for 500MB limit with calculated projections

-- =====================================================================
-- 1. USER_PROFILES TABLE
-- =====================================================================
-- Purpose: Store user profiles with trading preferences and styles
-- Estimated Size: ~1KB per profile, supports 10,000+ profiles
CREATE TABLE user_profiles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Basic Profile Information
    email VARCHAR(255) UNIQUE NOT NULL,
    display_name VARCHAR(100) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_login TIMESTAMP WITH TIME ZONE,
    
    -- Trading Style Configuration
    trading_style VARCHAR(20) NOT NULL CHECK (trading_style IN ('swing', 'long_term')),
    risk_tolerance VARCHAR(20) NOT NULL CHECK (risk_tolerance IN ('conservative', 'moderate', 'aggressive')),
    investment_horizon VARCHAR(20) NOT NULL CHECK (investment_horizon IN ('short', 'medium', 'long')),
    
    -- Profile Preferences
    preferred_sectors TEXT[], -- Array of sector preferences
    max_position_size DECIMAL(5,2) DEFAULT 10.00 CHECK (max_position_size > 0 AND max_position_size <= 100), -- Percentage
    auto_rebalance BOOLEAN DEFAULT false,
    notification_preferences JSONB DEFAULT '{}',
    
    -- Profile Status
    is_active BOOLEAN DEFAULT true,
    subscription_tier VARCHAR(20) DEFAULT 'free' CHECK (subscription_tier IN ('free', 'premium', 'professional'))
);

-- Indexes for user_profiles
CREATE INDEX idx_user_profiles_email ON user_profiles(email);
CREATE INDEX idx_user_profiles_trading_style ON user_profiles(trading_style);
CREATE INDEX idx_user_profiles_created_at ON user_profiles(created_at);
CREATE INDEX idx_user_profiles_active ON user_profiles(is_active);

-- =====================================================================
-- 2. PORTFOLIOS TABLE
-- =====================================================================
-- Purpose: Track portfolio balances and allocations for each user
-- Estimated Size: ~500 bytes per portfolio, supports 20,000+ portfolios
CREATE TABLE portfolios (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES user_profiles(id) ON DELETE CASCADE,
    
    -- Portfolio Identification
    portfolio_name VARCHAR(100) NOT NULL,
    portfolio_type VARCHAR(20) NOT NULL CHECK (portfolio_type IN ('simulation', 'paper', 'live')),
    
    -- Portfolio Balances
    initial_balance DECIMAL(15,2) NOT NULL CHECK (initial_balance > 0),
    current_cash DECIMAL(15,2) NOT NULL DEFAULT 0,
    total_value DECIMAL(15,2) NOT NULL DEFAULT 0,
    unrealized_pnl DECIMAL(15,2) DEFAULT 0,
    realized_pnl DECIMAL(15,2) DEFAULT 0,
    
    -- Portfolio Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_rebalance TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT true,
    
    -- Performance Tracking
    performance_benchmark VARCHAR(10) DEFAULT 'SPY', -- Benchmark symbol
    
    UNIQUE(user_id, portfolio_name)
);

-- Indexes for portfolios
CREATE INDEX idx_portfolios_user_id ON portfolios(user_id);
CREATE INDEX idx_portfolios_type ON portfolios(portfolio_type);
CREATE INDEX idx_portfolios_active ON portfolios(is_active);
CREATE INDEX idx_portfolios_updated_at ON portfolios(updated_at);

-- =====================================================================
-- 3. TRADES TABLE
-- =====================================================================
-- Purpose: Record all trade executions with detailed information
-- Estimated Size: ~300 bytes per trade, supports 300,000+ trades
CREATE TABLE trades (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    portfolio_id UUID NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
    
    -- Trade Identification
    symbol VARCHAR(10) NOT NULL,
    trade_type VARCHAR(10) NOT NULL CHECK (trade_type IN ('BUY', 'SELL')),
    order_type VARCHAR(20) NOT NULL CHECK (order_type IN ('market', 'limit', 'stop', 'stop_limit')),
    
    -- Trade Quantities and Prices
    quantity DECIMAL(15,6) NOT NULL CHECK (quantity > 0),
    price DECIMAL(10,4) NOT NULL CHECK (price > 0),
    total_amount DECIMAL(15,2) NOT NULL,
    commission DECIMAL(8,2) DEFAULT 0,
    
    -- Trade Execution Details
    executed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    strategy_used VARCHAR(50),
    confidence_score DECIMAL(3,2) CHECK (confidence_score >= 0 AND confidence_score <= 1),
    
    -- Trade Analysis
    entry_reason TEXT,
    exit_reason TEXT,
    holding_period INTERVAL, -- Calculated for sell trades
    
    -- Position Tracking
    position_size_pct DECIMAL(5,2), -- Percentage of portfolio
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for trades
CREATE INDEX idx_trades_portfolio_id ON trades(portfolio_id);
CREATE INDEX idx_trades_symbol ON trades(symbol);
CREATE INDEX idx_trades_executed_at ON trades(executed_at);
CREATE INDEX idx_trades_trade_type ON trades(trade_type);
CREATE INDEX idx_trades_strategy ON trades(strategy_used);

-- =====================================================================
-- 4. MARKET_DATA TABLE
-- =====================================================================
-- Purpose: Store historical and real-time market data
-- Estimated Size: ~200 bytes per record, supports 500,000+ data points
-- Note: Optimized with data retention policies
CREATE TABLE market_data (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Market Data Identification
    symbol VARCHAR(10) NOT NULL,
    data_type VARCHAR(20) NOT NULL CHECK (data_type IN ('price', 'volume', 'fundamental', 'sentiment')),
    timeframe VARCHAR(10) NOT NULL CHECK (timeframe IN ('1min', '5min', '1hour', '1day', '1week')),
    
    -- Price Data (OHLCV)
    open_price DECIMAL(10,4),
    high_price DECIMAL(10,4),
    low_price DECIMAL(10,4),
    close_price DECIMAL(10,4),
    volume BIGINT,
    
    -- Additional Metrics
    market_cap DECIMAL(15,2),
    pe_ratio DECIMAL(8,2),
    sentiment_score DECIMAL(3,2),
    
    -- Technical Indicators (stored as JSONB for flexibility)
    technical_indicators JSONB DEFAULT '{}',
    
    -- Temporal Information
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Unique constraint to prevent duplicates
    UNIQUE(symbol, data_type, timeframe, timestamp)
);

-- Indexes for market_data
CREATE INDEX idx_market_data_symbol ON market_data(symbol);
CREATE INDEX idx_market_data_timestamp ON market_data(timestamp);
CREATE INDEX idx_market_data_symbol_timestamp ON market_data(symbol, timestamp);
CREATE INDEX idx_market_data_type ON market_data(data_type);
CREATE INDEX idx_market_data_timeframe ON market_data(timeframe);

-- =====================================================================
-- 5. PERFORMANCE_METRICS TABLE
-- =====================================================================
-- Purpose: Track portfolio and trade performance metrics
-- Estimated Size: ~400 bytes per record, supports 250,000+ metrics
CREATE TABLE performance_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    portfolio_id UUID NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
    
    -- Metric Identification
    metric_type VARCHAR(30) NOT NULL CHECK (metric_type IN (
        'daily_return', 'total_return', 'sharpe_ratio', 'max_drawdown', 
        'win_rate', 'avg_trade_return', 'volatility', 'beta'
    )),
    metric_period VARCHAR(20) NOT NULL CHECK (metric_period IN ('daily', 'weekly', 'monthly', 'quarterly', 'yearly', 'all_time')),
    
    -- Metric Values
    metric_value DECIMAL(15,6) NOT NULL,
    benchmark_value DECIMAL(15,6),
    alpha DECIMAL(8,4), -- Excess return over benchmark
    
    -- Performance Context
    trade_count INTEGER DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    losing_trades INTEGER DEFAULT 0,
    
    -- Temporal Information
    calculation_date DATE NOT NULL,
    period_start DATE,
    period_end DATE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Unique constraint
    UNIQUE(portfolio_id, metric_type, metric_period, calculation_date)
);

-- Indexes for performance_metrics
CREATE INDEX idx_performance_metrics_portfolio_id ON performance_metrics(portfolio_id);
CREATE INDEX idx_performance_metrics_type ON performance_metrics(metric_type);
CREATE INDEX idx_performance_metrics_date ON performance_metrics(calculation_date);
CREATE INDEX idx_performance_metrics_period ON performance_metrics(metric_period);

-- =====================================================================
-- 6. NOTIFICATION_SETTINGS TABLE
-- =====================================================================
-- Purpose: Manage user notification preferences and delivery
-- Estimated Size: ~300 bytes per record, supports 100,000+ notifications
CREATE TABLE notification_settings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES user_profiles(id) ON DELETE CASCADE,
    
    -- Notification Type Configuration
    notification_type VARCHAR(30) NOT NULL CHECK (notification_type IN (
        'trade_execution', 'portfolio_alert', 'performance_update', 
        'market_news', 'rebalance_suggestion', 'risk_warning'
    )),
    
    -- Delivery Preferences
    email_enabled BOOLEAN DEFAULT true,
    sms_enabled BOOLEAN DEFAULT false,
    push_enabled BOOLEAN DEFAULT true,
    in_app_enabled BOOLEAN DEFAULT true,
    
    -- Notification Triggers
    trigger_conditions JSONB DEFAULT '{}', -- Customizable trigger conditions
    frequency VARCHAR(20) DEFAULT 'immediate' CHECK (frequency IN ('immediate', 'hourly', 'daily', 'weekly')),
    
    -- Notification Content
    custom_message TEXT,
    priority_level VARCHAR(10) DEFAULT 'medium' CHECK (priority_level IN ('low', 'medium', 'high', 'critical')),
    
    -- Status and Metadata
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(user_id, notification_type)
);

-- Indexes for notification_settings
CREATE INDEX idx_notification_settings_user_id ON notification_settings(user_id);
CREATE INDEX idx_notification_settings_type ON notification_settings(notification_type);
CREATE INDEX idx_notification_settings_active ON notification_settings(is_active);
CREATE INDEX idx_notification_settings_priority ON notification_settings(priority_level);

-- =====================================================================
-- 7. TRIGGERS FOR AUTOMATIC TIMESTAMP UPDATES
-- =====================================================================

-- Function to update timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for automatic timestamp updates
CREATE TRIGGER update_user_profiles_updated_at BEFORE UPDATE ON user_profiles FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_portfolios_updated_at BEFORE UPDATE ON portfolios FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_notification_settings_updated_at BEFORE UPDATE ON notification_settings FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- =====================================================================
-- 8. STORAGE CALCULATIONS AND OPTIMIZATION
-- =====================================================================

-- Storage Analysis (for 500MB Supabase free tier):
-- 1. user_profiles: 1KB × 10,000 users = 10MB
-- 2. portfolios: 500B × 20,000 portfolios = 10MB  
-- 3. trades: 300B × 300,000 trades = 90MB
-- 4. market_data: 200B × 500,000 records = 100MB (with retention policy)
-- 5. performance_metrics: 400B × 250,000 records = 100MB
-- 6. notification_settings: 300B × 100,000 records = 30MB
-- 7. Indexes and overhead: ~40MB
-- Total Estimated Usage: ~380MB (76% of 500MB limit)

-- Data Retention Policy for Market Data (to stay within limits)
CREATE OR REPLACE FUNCTION cleanup_old_market_data()
RETURNS void AS $$
BEGIN
    -- Delete market data older than 2 years for minute/hour data
    DELETE FROM market_data 
    WHERE timeframe IN ('1min', '5min', '1hour') 
    AND timestamp < NOW() - INTERVAL '2 years';
    
    -- Delete daily data older than 5 years
    DELETE FROM market_data 
    WHERE timeframe = '1day' 
    AND timestamp < NOW() - INTERVAL '5 years';
    
    -- Keep weekly data indefinitely (small volume)
END;
$$ LANGUAGE plpgsql;

-- =====================================================================
-- 9. PERFORMANCE OPTIMIZATION VIEWS
-- =====================================================================

-- View: Portfolio Summary with Performance
CREATE VIEW portfolio_summary AS
SELECT 
    p.id,
    p.portfolio_name,
    u.display_name as user_name,
    u.trading_style,
    p.current_cash,
    p.total_value,
    p.unrealized_pnl,
    p.realized_pnl,
    ROUND(((p.total_value - p.initial_balance) / p.initial_balance * 100), 2) as total_return_pct,
    (SELECT COUNT(*) FROM trades t WHERE t.portfolio_id = p.id) as total_trades,
    p.last_rebalance,
    p.updated_at
FROM portfolios p
JOIN user_profiles u ON p.user_id = u.id
WHERE p.is_active = true;

-- View: Recent Trade Activity
CREATE VIEW recent_trades AS
SELECT 
    t.id,
    p.portfolio_name,
    u.display_name as user_name,
    t.symbol,
    t.trade_type,
    t.quantity,
    t.price,
    t.total_amount,
    t.strategy_used,
    t.confidence_score,
    t.executed_at
FROM trades t
JOIN portfolios p ON t.portfolio_id = p.id
JOIN user_profiles u ON p.user_id = u.id
ORDER BY t.executed_at DESC
LIMIT 1000;

-- View: Top Performing Portfolios
CREATE VIEW top_performers AS
SELECT 
    p.id,
    p.portfolio_name,
    u.display_name as user_name,
    u.trading_style,
    p.total_value,
    ROUND(((p.total_value - p.initial_balance) / p.initial_balance * 100), 2) as return_pct,
    pm.sharpe_ratio,
    pm.max_drawdown,
    pm.win_rate
FROM portfolios p
JOIN user_profiles u ON p.user_id = u.id
LEFT JOIN (
    SELECT DISTINCT ON (portfolio_id)
        portfolio_id,
        metric_value as sharpe_ratio
    FROM performance_metrics
    WHERE metric_type = 'sharpe_ratio' AND metric_period = 'all_time'
    ORDER BY portfolio_id, calculation_date DESC
) pm_sharpe ON p.id = pm_sharpe.portfolio_id
LEFT JOIN (
    SELECT DISTINCT ON (portfolio_id)
        portfolio_id,
        metric_value as max_drawdown
    FROM performance_metrics
    WHERE metric_type = 'max_drawdown' AND metric_period = 'all_time'
    ORDER BY portfolio_id, calculation_date DESC
) pm_dd ON p.id = pm_dd.portfolio_id
LEFT JOIN (
    SELECT DISTINCT ON (portfolio_id)
        portfolio_id,
        metric_value as win_rate
    FROM performance_metrics
    WHERE metric_type = 'win_rate' AND metric_period = 'all_time'
    ORDER BY portfolio_id, calculation_date DESC
) pm ON p.id = pm.portfolio_id
WHERE p.is_active = true
ORDER BY return_pct DESC;

-- =====================================================================
-- 10. SAMPLE DATA VALIDATION FUNCTIONS
-- =====================================================================

-- Function to validate portfolio consistency
CREATE OR REPLACE FUNCTION validate_portfolio_balance(portfolio_uuid UUID)
RETURNS BOOLEAN AS $$
DECLARE
    calculated_total DECIMAL(15,2);
    stored_total DECIMAL(15,2);
BEGIN
    -- Calculate total value from cash + positions
    SELECT 
        p.current_cash + COALESCE(SUM(
            CASE WHEN t.trade_type = 'BUY' THEN t.total_amount ELSE -t.total_amount END
        ), 0) INTO calculated_total
    FROM portfolios p
    LEFT JOIN trades t ON p.id = t.portfolio_id
    WHERE p.id = portfolio_uuid
    GROUP BY p.id, p.current_cash;
    
    -- Get stored total value
    SELECT total_value INTO stored_total
    FROM portfolios
    WHERE id = portfolio_uuid;
    
    -- Return true if difference is within $0.01
    RETURN ABS(calculated_total - stored_total) < 0.01;
END;
$$ LANGUAGE plpgsql;

-- Function to calculate win rate for a portfolio
CREATE OR REPLACE FUNCTION calculate_win_rate(portfolio_uuid UUID)
RETURNS DECIMAL AS $$
DECLARE
    total_trades INTEGER;
    winning_trades INTEGER;
BEGIN
    SELECT COUNT(*) INTO total_trades
    FROM trades
    WHERE portfolio_id = portfolio_uuid AND trade_type = 'SELL';
    
    IF total_trades = 0 THEN
        RETURN 0;
    END IF;
    
    -- This is simplified - in reality you'd match buy/sell pairs
    SELECT COUNT(*) INTO winning_trades
    FROM trades
    WHERE portfolio_id = portfolio_uuid 
    AND trade_type = 'SELL' 
    AND total_amount > 0; -- Simplified profit check
    
    RETURN ROUND(winning_trades::DECIMAL / total_trades * 100, 2);
END;
$$ LANGUAGE plpgsql;

-- =====================================================================
-- 11. SECURITY AND ROW LEVEL SECURITY (RLS)
-- =====================================================================

-- Enable RLS on sensitive tables
ALTER TABLE user_profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE portfolios ENABLE ROW LEVEL SECURITY;
ALTER TABLE trades ENABLE ROW LEVEL SECURITY;
ALTER TABLE notification_settings ENABLE ROW LEVEL SECURITY;

-- RLS Policies (assuming Supabase auth.uid() function)
-- Users can only see their own data

-- User profiles - users can only access their own profile
CREATE POLICY "Users can view own profile" ON user_profiles
    FOR ALL USING (auth.uid()::TEXT = id::TEXT);

-- Portfolios - users can only access their own portfolios
CREATE POLICY "Users can view own portfolios" ON portfolios
    FOR ALL USING (auth.uid()::TEXT IN (
        SELECT id::TEXT FROM user_profiles WHERE id = user_id
    ));

-- Trades - users can only access trades for their own portfolios
CREATE POLICY "Users can view own trades" ON trades
    FOR ALL USING (auth.uid()::TEXT IN (
        SELECT up.id::TEXT 
        FROM user_profiles up 
        JOIN portfolios p ON up.id = p.user_id 
        WHERE p.id = portfolio_id
    ));

-- Notification settings - users can only access their own notifications
CREATE POLICY "Users can view own notifications" ON notification_settings
    FOR ALL USING (auth.uid()::TEXT IN (
        SELECT id::TEXT FROM user_profiles WHERE id = user_id
    ));

-- =====================================================================
-- 12. PERFORMANCE INDEXES FOR COMMON QUERIES
-- =====================================================================

-- Composite indexes for common query patterns
CREATE INDEX idx_trades_portfolio_symbol_date ON trades(portfolio_id, symbol, executed_at);
CREATE INDEX idx_market_data_symbol_type_timestamp ON market_data(symbol, data_type, timestamp DESC);
CREATE INDEX idx_performance_metrics_portfolio_type_date ON performance_metrics(portfolio_id, metric_type, calculation_date DESC);

-- Partial indexes for active records only
CREATE INDEX idx_active_portfolios_user ON portfolios(user_id, updated_at) WHERE is_active = true;
CREATE INDEX idx_active_users_login ON user_profiles(last_login) WHERE is_active = true;

-- =====================================================================
-- END OF SCHEMA
-- =====================================================================

-- Schema Summary:
-- ✅ 6 main tables with proper relationships
-- ✅ Comprehensive indexes for performance
-- ✅ Storage optimized for 500MB Supabase limit
-- ✅ Row Level Security for data protection
-- ✅ Performance views for common queries
-- ✅ Data validation functions
-- ✅ Automatic timestamp management
-- ✅ Data retention policies
-- ✅ Support for multiple trading styles (swing vs long-term)
-- ✅ Flexible notification system
-- ✅ Portfolio performance tracking
-- ✅ Comprehensive trade history

-- Estimated Storage: ~380MB (76% of 500MB free tier)
-- Capacity: 10K users, 20K portfolios, 300K trades, 500K market data points
-- Performance: Optimized indexes for sub-second query response