# Supabase Setup Guide for Multi-Profile Trading Simulator

This guide will walk you through setting up the Supabase PostgreSQL database for the multi-profile trading simulator.

## Prerequisites

- Supabase account (free tier provides 500MB PostgreSQL database)
- Python 3.8+ with required dependencies
- Basic understanding of PostgreSQL

## Step 1: Create Supabase Project

1. **Sign up for Supabase** (if you haven't already):
   - Go to [https://supabase.com](https://supabase.com)
   - Sign up with GitHub, Google, or email

2. **Create a new project**:
   - Click "New project"
   - Choose your organization
   - Enter project name: `trading-simulator`
   - Enter database password (save this!)
   - Select region closest to your users
   - Click "Create new project"

3. **Wait for setup** (1-2 minutes):
   - Supabase will provision your PostgreSQL database
   - You'll get a dashboard with database URL and API keys

## Step 2: Get Database Connection Details

1. **Navigate to Settings > Database**
2. **Copy the connection details**:
   ```
   Host: db.your-project-ref.supabase.co
   Database name: postgres
   Port: 5432
   Username: postgres
   Password: [your-password]
   ```

3. **Copy the connection string**:
   ```
   postgresql://postgres:[YOUR-PASSWORD]@db.your-project-ref.supabase.co:5432/postgres
   ```

## Step 3: Install Required Dependencies

```bash
# Install Python dependencies
pip install -r requirements-dev.txt

# Or install specific packages:
pip install psycopg2-binary asyncpg supabase
```

## Step 4: Configure Environment Variables

Create a `.env` file in your project root:

```env
# Supabase Configuration
SUPABASE_URL=postgresql://postgres:[YOUR-PASSWORD]@db.your-project-ref.supabase.co:5432/postgres
SUPABASE_SERVICE_KEY=your-service-role-key-here
SUPABASE_ANON_KEY=your-anon-key-here

# Alternative: Use DATABASE_URL
DATABASE_URL=postgresql://postgres:[YOUR-PASSWORD]@db.your-project-ref.supabase.co:5432/postgres

# Environment Configuration
ENVIRONMENT=development
DB_CONNECTION_TIMEOUT=30
DB_COMMAND_TIMEOUT=60
SUPABASE_SSL_MODE=require

# Trading Simulator Configuration
ENABLE_REAL_TRADING=false
DEFAULT_PORTFOLIO_BALANCE=100000.00
```

**Security Notes:**
- Never commit `.env` files to version control
- Add `.env` to your `.gitignore` file
- Use environment variables in production

## Step 5: Run Database Schema Setup

1. **Test connection**:
   ```bash
   python config/supabase_config.py
   ```

2. **Apply database schema**:
   ```bash
   # Connect to your Supabase database
   psql "postgresql://postgres:[YOUR-PASSWORD]@db.your-project-ref.supabase.co:5432/postgres"
   
   # Or use Supabase SQL Editor in the dashboard
   ```

3. **Execute schema file**:
   ```sql
   -- Copy and paste the contents of sql/trading_simulator_schema.sql
   -- Or upload the file through Supabase dashboard > SQL Editor
   ```

## Step 6: Configure Row Level Security (RLS)

The schema includes RLS policies, but you need to enable authentication:

1. **Go to Authentication > Settings**
2. **Configure your authentication provider** (email, OAuth, etc.)
3. **Enable RLS** (already done in schema)
4. **Test RLS policies**:
   ```sql
   -- Test as authenticated user
   SELECT * FROM user_profiles;
   
   -- Should only return data for the authenticated user
   ```

## Step 7: Set Up Storage Monitoring

1. **Enable monitoring** in Supabase dashboard
2. **Set up alerts** for storage usage:
   ```sql
   -- Query to check current storage usage
   SELECT 
       pg_size_pretty(pg_database_size(current_database())) as size,
       pg_database_size(current_database()) / (1024*1024) as size_mb
   ```

3. **Configure cleanup job** (run weekly):
   ```sql
   -- Manual cleanup (or set up as scheduled function)
   SELECT cleanup_old_market_data();
   ```

## Step 8: Test Database Operations

Run the test script to verify everything is working:

```bash
python -c "
from config.supabase_config import get_supabase_manager

# Test connection
manager = get_supabase_manager()
health = manager.health_check()
print('Health status:', health['status'])
print('Storage usage:', health.get('storage_usage', {}))

# Test table creation
tables = manager.get_table_sizes()
print('Tables created:', list(tables.keys()))
"
```

## Step 9: Configure Application

Update your application configuration to use Supabase:

```python
# In your main application
from config.supabase_config import get_supabase_manager

# Get database manager
db_manager = get_supabase_manager()

# Use in your application
with db_manager.get_connection() as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM user_profiles LIMIT 5")
    users = cursor.fetchall()
```

## Storage Optimization Tips

### 1. Monitor Storage Usage
```python
from config.supabase_config import get_storage_status

status = get_storage_status()
print(f"Storage: {status['storage']['usage_pct']}% used")
```

### 2. Set Up Automated Cleanup
```sql
-- Create scheduled cleanup (if supported)
SELECT cron.schedule('cleanup-market-data', '0 2 * * 0', 'SELECT cleanup_old_market_data()');
```

### 3. Optimize Queries
- Use indexes effectively
- Limit large result sets
- Archive old data regularly

### 4. Storage Breakdown (500MB free tier)
- **User profiles**: ~10MB (10,000 users)
- **Portfolios**: ~10MB (20,000 portfolios) 
- **Trades**: ~90MB (300,000 trades)
- **Market data**: ~100MB (with cleanup)
- **Performance metrics**: ~100MB
- **Notifications**: ~30MB
- **Indexes & overhead**: ~40MB
- **Total**: ~380MB (76% of limit)

## Troubleshooting

### Common Issues

1. **Connection timeout**:
   ```bash
   # Check if Supabase is accessible
   ping db.your-project-ref.supabase.co
   
   # Verify SSL settings
   psql "postgresql://postgres:password@host:5432/postgres?sslmode=require"
   ```

2. **Permission errors**:
   ```sql
   -- Check RLS policies
   SELECT * FROM pg_policies WHERE tablename = 'user_profiles';
   
   -- Verify user authentication
   SELECT auth.uid();
   ```

3. **Storage limit exceeded**:
   ```python
   # Run storage optimization
   from config.supabase_config import get_supabase_manager
   
   manager = get_supabase_manager()
   results = manager.optimize_storage()
   print(f"Freed {results['space_freed_mb']}MB")
   ```

4. **Slow queries**:
   ```sql
   -- Check index usage
   EXPLAIN ANALYZE SELECT * FROM trades WHERE portfolio_id = 'uuid';
   
   -- Monitor slow queries
   SELECT query, mean_time, calls 
   FROM pg_stat_statements 
   ORDER BY mean_time DESC 
   LIMIT 10;
   ```

### Performance Optimization

1. **Enable query performance monitoring**:
   ```sql
   -- Enable pg_stat_statements (if available)
   CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
   ```

2. **Monitor connection pool**:
   ```python
   health = manager.health_check()
   print(health['connection_pool_status'])
   ```

3. **Use prepared statements** for frequent queries:
   ```python
   cursor.execute("PREPARE get_user AS SELECT * FROM user_profiles WHERE id = $1")
   cursor.execute("EXECUTE get_user (%s)", [user_id])
   ```

## Production Checklist

- [ ] Database schema applied successfully
- [ ] RLS policies enabled and tested
- [ ] Environment variables configured
- [ ] SSL connection verified
- [ ] Storage monitoring set up
- [ ] Backup strategy configured (Supabase handles this)
- [ ] Connection pooling optimized
- [ ] Query performance tested
- [ ] Error handling implemented
- [ ] Cleanup jobs scheduled

## Security Best Practices

1. **Never expose service role key** in client-side code
2. **Use anon key** for client authentication
3. **Implement proper RLS policies** for all tables
4. **Validate all inputs** before database operations
5. **Use prepared statements** to prevent SQL injection
6. **Monitor database access logs**
7. **Regularly review and update permissions**

## Support and Resources

- **Supabase Documentation**: https://supabase.com/docs
- **PostgreSQL Documentation**: https://www.postgresql.org/docs/
- **Python psycopg2 Documentation**: https://www.psycopg.org/docs/
- **Trading Simulator GitHub Issues**: [Your repository URL]

## Next Steps

After completing the Supabase setup:

1. **Implement database manager** (`src/core/trading_database_manager.py`)
2. **Create API endpoints** for portfolio management
3. **Build user interface** for profile management
4. **Integrate with trading engine**
5. **Set up monitoring and alerts**
6. **Deploy to production**

---

**Note**: This setup uses Supabase free tier (500MB). For production use with larger datasets, consider upgrading to Supabase Pro or implementing data archiving strategies.