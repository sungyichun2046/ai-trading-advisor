# Multi-Profile Trading Simulator Database Setup

Complete database design and implementation for a multi-profile trading simulator with Supabase PostgreSQL.

## 🎯 Project Overview

This database system supports:
- **Multiple user profiles** with different trading styles (swing vs long-term)
- **Portfolio management** with real-time tracking
- **Trade execution history** with detailed analytics
- **Market data storage** with optimized retention
- **Performance metrics** and reporting
- **Email notifications** and alerts
- **Storage optimization** for Supabase 500MB free tier

## 📋 Database Schema

### Core Tables

1. **`user_profiles`** - User accounts with trading preferences
2. **`portfolios`** - Multiple portfolios per user
3. **`trades`** - Complete trade execution history
4. **`market_data`** - Historical and real-time market data
5. **`performance_metrics`** - Portfolio performance tracking
6. **`notification_settings`** - User notification preferences

### Key Features

- ✅ **Row Level Security (RLS)** for data protection
- ✅ **Foreign key constraints** with CASCADE deletes
- ✅ **Performance indexes** for fast queries
- ✅ **Data validation** with CHECK constraints
- ✅ **Automatic timestamps** with triggers
- ✅ **Storage monitoring** and cleanup functions

## 🚀 Quick Setup

### 1. Install Dependencies

```bash
pip install psycopg2-binary asyncpg supabase
```

### 2. Configure Environment

Create `.env` file:

```env
SUPABASE_URL=postgresql://postgres:[PASSWORD]@db.your-project-ref.supabase.co:5432/postgres
DATABASE_URL=postgresql://postgres:[PASSWORD]@db.your-project-ref.supabase.co:5432/postgres
ENVIRONMENT=development
```

### 3. Apply Schema

```sql
-- Execute sql/trading_simulator_schema.sql in Supabase SQL Editor
\i sql/trading_simulator_schema.sql
```

### 4. Test Setup

```bash
python config/supabase_config.py
python tests/test_trading_database_schema.py
```

## 📊 Storage Calculations

**Optimized for Supabase 500MB Free Tier:**

| Table | Records | Size/Record | Total Size | % of Limit |
|-------|---------|-------------|------------|------------|
| user_profiles | 10,000 | 1KB | 10MB | 2% |
| portfolios | 20,000 | 512B | 10MB | 2% |
| trades | 300,000 | 300B | 90MB | 18% |
| market_data | 500,000 | 200B | 100MB | 20% |
| performance_metrics | 250,000 | 400B | 100MB | 20% |
| notification_settings | 100,000 | 300B | 30MB | 6% |
| **Indexes & Overhead** | - | - | **40MB** | **8%** |
| **TOTAL** | - | - | **380MB** | **76%** |

**Remaining capacity: 120MB (24%)**

## 🔧 Usage Examples

### User Management

```python
from core.trading_database_manager import get_trading_db_manager, UserProfile, TradingStyle

db = get_trading_db_manager()

# Create user profile
profile = UserProfile(
    email="trader@example.com",
    display_name="Active Trader",
    trading_style=TradingStyle.SWING,
    max_position_size=Decimal('15.0')
)
user_id = db.create_user_profile(profile)

# Get user profile
profile = db.get_user_profile(user_id)
```

### Portfolio Management

```python
from core.trading_database_manager import Portfolio, PortfolioType

# Create portfolio
portfolio = Portfolio(
    user_id=user_id,
    portfolio_name="Growth Portfolio",
    portfolio_type=PortfolioType.SIMULATION,
    initial_balance=Decimal('100000.00')
)
portfolio_id = db.create_portfolio(portfolio)

# Get user portfolios
portfolios = db.get_user_portfolios(user_id)
```

### Trade Recording

```python
from core.trading_database_manager import Trade, TradeType, OrderType

# Record trade
trade = Trade(
    portfolio_id=portfolio_id,
    symbol="AAPL",
    trade_type=TradeType.BUY,
    order_type=OrderType.MARKET,
    quantity=Decimal('100'),
    price=Decimal('150.00'),
    total_amount=Decimal('15000.00')
)
trade_id = db.record_trade(trade)

# Get portfolio trades
trades = db.get_portfolio_trades(portfolio_id, symbol="AAPL")
```

### Performance Analysis

```python
# Calculate portfolio performance
performance = db.calculate_portfolio_performance(portfolio_id)
print(f"Total Return: {performance['total_return_pct']:.2f}%")
print(f"Total Trades: {performance['total_trades']}")
```

## 🔍 Monitoring & Validation

### Storage Validation

```bash
python scripts/validate_storage_performance.py
```

### Performance Benchmarks

```bash
python scripts/validate_storage_performance.py --load-test --load-profiles 100
```

### Health Check

```python
from config.supabase_config import get_supabase_manager

manager = get_supabase_manager()
health = manager.health_check()
print(f"Status: {health['status']}")
print(f"Storage: {health['storage_usage']['usage_pct']}%")
```

## 📈 Performance Optimizations

### Indexes

```sql
-- Key performance indexes
CREATE INDEX idx_trades_portfolio_symbol_date ON trades(portfolio_id, symbol, executed_at);
CREATE INDEX idx_market_data_symbol_timestamp ON market_data(symbol, timestamp DESC);
CREATE INDEX idx_performance_metrics_portfolio_type_date ON performance_metrics(portfolio_id, metric_type, calculation_date DESC);
```

### Query Patterns

```sql
-- Optimized portfolio summary
SELECT p.*, u.display_name, u.trading_style
FROM portfolios p
JOIN user_profiles u ON p.user_id = u.id
WHERE p.is_active = true
ORDER BY p.updated_at DESC;

-- Recent trade activity
SELECT t.*, p.portfolio_name
FROM trades t
JOIN portfolios p ON t.portfolio_id = p.id
WHERE t.executed_at >= NOW() - INTERVAL '7 days'
ORDER BY t.executed_at DESC;
```

### Data Cleanup

```sql
-- Automated cleanup for storage optimization
SELECT cleanup_old_market_data();
```

## 🔒 Security Features

### Row Level Security

```sql
-- Users can only access their own data
ALTER TABLE user_profiles ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Users can view own profile" ON user_profiles
    FOR ALL USING (auth.uid()::TEXT = id::TEXT);
```

### Data Validation

```sql
-- Enum constraints
CHECK (trading_style IN ('swing', 'long_term'))
CHECK (risk_tolerance IN ('conservative', 'moderate', 'aggressive'))

-- Numeric constraints  
CHECK (initial_balance > 0)
CHECK (quantity > 0)
CHECK (max_position_size > 0 AND max_position_size <= 100)
```

## 🧪 Testing

### Run Full Test Suite

```bash
python tests/test_trading_database_schema.py
```

### Test Categories

- ✅ **Schema Validation** - Table structure and constraints
- ✅ **Data Integrity** - Foreign keys and cascades
- ✅ **Business Logic** - Validation and calculations
- ✅ **Performance** - Query speed and optimization
- ✅ **Edge Cases** - Error handling and extreme values

### Load Testing

```bash
# Test with 100 profiles, 200 portfolios, 1000 trades
python scripts/validate_storage_performance.py --load-test --load-profiles 100 --cleanup
```

## 📚 File Structure

```
├── sql/
│   └── trading_simulator_schema.sql          # Complete database schema
├── config/
│   ├── supabase_config.py                   # Database configuration
│   └── supabase_setup.md                    # Setup instructions
├── src/core/
│   └── trading_database_manager.py          # High-level database API
├── scripts/
│   └── validate_storage_performance.py      # Validation and testing
├── tests/
│   └── test_trading_database_schema.py      # Comprehensive tests
└── README_DATABASE_SETUP.md                 # This file
```

## 🎯 Production Readiness

### Deployment Checklist

- [ ] ✅ Schema applied to production database
- [ ] ✅ Environment variables configured
- [ ] ✅ RLS policies enabled and tested
- [ ] ✅ Performance indexes created
- [ ] ✅ Data validation working
- [ ] ✅ Storage monitoring active
- [ ] ✅ Cleanup jobs scheduled
- [ ] ✅ Backup strategy verified
- [ ] ✅ Connection pooling optimized
- [ ] ✅ Error handling implemented

### Scaling Considerations

For production use beyond 500MB:

1. **Upgrade Supabase Plan** - Pro plan provides more storage
2. **Implement Data Archiving** - Move old data to cold storage
3. **Optimize Indexes** - Remove unused indexes
4. **Partition Large Tables** - Split by date or user
5. **Consider Read Replicas** - For analytics workloads

## 🛠️ Troubleshooting

### Common Issues

1. **Connection Errors**
   ```bash
   # Check connectivity
   ping db.your-project-ref.supabase.co
   ```

2. **Storage Limit**
   ```python
   # Check and optimize storage
   manager.optimize_storage()
   ```

3. **Slow Queries**
   ```sql
   -- Analyze query performance
   EXPLAIN ANALYZE SELECT * FROM trades WHERE portfolio_id = 'uuid';
   ```

### Support Resources

- 📖 [Supabase Documentation](https://supabase.com/docs)
- 🐘 [PostgreSQL Docs](https://www.postgresql.org/docs/)
- 🐍 [psycopg2 Documentation](https://www.psycopg.org/docs/)

---

**Database designed and optimized for production use with Supabase PostgreSQL. Ready for multi-profile trading simulator deployment!** 🚀