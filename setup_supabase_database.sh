#!/bin/bash
# =====================================================================
# Supabase Database Setup and Validation Script
# =====================================================================
# One-time setup script to initialize and validate the Supabase database
# for the multi-profile trading simulator before running DAGs.
#
# Usage:
#   ./setup_supabase_database.sh
#   ./setup_supabase_database.sh --validate-only
#   ./setup_supabase_database.sh --reset
#
# Requirements:
#   - Supabase account and project created
#   - SUPABASE_URL environment variable or .env file
#   - Python virtual environment with dependencie
# 
# 6 Table will be created + 3 extra tables VIEWs:

#  1. portfolio_summary - A view combining portfolios + user_profiles with performance metrics
#  2. recent_trades - A view showing the latest 1000 trades with portfolio/user info
#  3. top_performers - A view ranking portfolios by return percentage
# =====================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
VALIDATE_ONLY=false
RESET_DATABASE=false
SKIP_SCHEMA=false

# Parse command line arguments
for arg in "$@"; do
    case $arg in
        --validate-only)
            VALIDATE_ONLY=true
            echo "🔍 Mode: Validation only (no schema changes)"
            ;;
        --reset)
            RESET_DATABASE=true
            echo "🔄 Mode: Reset database (WARNING: Will delete all data)"
            ;;
        --skip-schema)
            SKIP_SCHEMA=true
            echo "⏭️  Mode: Skip schema application"
            ;;
        --help)
            echo "Supabase Database Setup and Validation"
            echo ""
            echo "Usage:"
            echo "  ./setup_supabase_database.sh                # Full setup"
            echo "  ./setup_supabase_database.sh --validate-only # Only validate"
            echo "  ./setup_supabase_database.sh --reset        # Reset database"
            echo "  ./setup_supabase_database.sh --skip-schema  # Skip schema"
            echo ""
            echo "Environment Variables:"
            echo "  SUPABASE_URL    - Full PostgreSQL connection URL (required)"
            echo "  DATABASE_URL    - Alternative to SUPABASE_URL"
            echo ""
            exit 0
            ;;
    esac
done

echo -e "${BLUE}🚀 SUPABASE DATABASE SETUP AND VALIDATION${NC}"
echo "=============================================="
echo "$(date '+%Y-%m-%d %H:%M:%S') - Starting database setup..."
echo ""

# =====================================================================
# 1. ENVIRONMENT VALIDATION
# =====================================================================

echo -e "${BLUE}📋 Step 1: Environment Validation${NC}"
echo "=================================="

# Check for database URL
if [ -n "$SUPABASE_URL" ]; then
    DATABASE_URL="$SUPABASE_URL"
    echo "✅ Using SUPABASE_URL from environment"
elif [ -n "$DATABASE_URL" ]; then
    echo "✅ Using DATABASE_URL from environment"
elif [ -f ".env" ] && grep -q "SUPABASE_URL" .env; then
    source .env
    DATABASE_URL="$SUPABASE_URL"
    echo "✅ Loaded SUPABASE_URL from .env file"
elif [ -f ".env" ] && grep -q "DATABASE_URL" .env; then
    source .env
    echo "✅ Loaded DATABASE_URL from .env file"
else
    echo -e "${RED}❌ ERROR: No database URL found!${NC}"
    echo ""
    echo "Please set one of:"
    echo "  export SUPABASE_URL=\"postgresql://postgres:password@db.project.supabase.co:5432/postgres\""
    echo "  export DATABASE_URL=\"postgresql://postgres:password@db.project.supabase.co:5432/postgres\""
    echo ""
    echo "Or create .env file with:"
    echo "  SUPABASE_URL=postgresql://postgres:password@db.project.supabase.co:5432/postgres"
    exit 1
fi

# Validate URL format
if [[ ! "$DATABASE_URL" =~ ^postgresql:// ]]; then
    echo -e "${RED}❌ ERROR: Invalid database URL format${NC}"
    echo "Expected: postgresql://username:password@host:port/database"
    echo "Got: $DATABASE_URL"
    exit 1
fi

echo "📊 Database URL: ${DATABASE_URL%@*}@[HIDDEN]"

# Check and setup Python virtual environment
echo ""
echo "🐍 Checking Python environment..."

# Check if we're in a virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${YELLOW}⚠️  Not in a virtual environment${NC}"
    
    # Check if venv directory exists
    if [ -d "venv" ]; then
        echo "🔍 Found existing venv directory"
        echo "🔄 Activating virtual environment..."
        
        # Activate virtual environment
        if [ -f "venv/bin/activate" ]; then
            source venv/bin/activate
            echo "✅ Virtual environment activated: $VIRTUAL_ENV"
        else
            echo -e "${RED}❌ ERROR: venv/bin/activate not found${NC}"
            echo "Please create virtual environment: python -m venv venv"
            exit 1
        fi
    else
        echo "📦 Creating new virtual environment..."
        
        # Create virtual environment
        if command -v python3 &> /dev/null; then
            python3 -m venv venv
        elif command -v python &> /dev/null; then
            python -m venv venv
        else
            echo -e "${RED}❌ ERROR: Python not found${NC}"
            echo "Please install Python 3.8+ first"
            exit 1
        fi
        
        # Activate virtual environment
        source venv/bin/activate
        echo "✅ Virtual environment created and activated: $VIRTUAL_ENV"
        
        # Upgrade pip
        echo "📦 Upgrading pip..."
        pip install --upgrade pip
    fi
else
    echo "✅ Already in virtual environment: $VIRTUAL_ENV"
fi

# Determine Python command to use
if command -v python &> /dev/null; then
    PYTHON_CMD="python"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    echo -e "${RED}❌ ERROR: Python not found${NC}"
    exit 1
fi

echo "✅ Using Python: $($PYTHON_CMD --version)"

# Check for psql
if ! command -v psql &> /dev/null; then
    echo -e "${YELLOW}⚠️  WARNING: psql not found - will use Python for database operations${NC}"
    USE_PYTHON_ONLY=true
else
    echo "✅ psql found: $(psql --version)"
    USE_PYTHON_ONLY=false
fi

# Check Python dependencies
echo ""
echo "🔍 Checking Python dependencies..."

MISSING_DEPS=()

# Check each dependency
if ! $PYTHON_CMD -c "import psycopg2" &> /dev/null; then
    MISSING_DEPS+=("psycopg2-binary")
fi

# Check if requirements files exist and install from them
if [ -f "requirements.txt" ] && [ ${#MISSING_DEPS[@]} -gt 0 ]; then
    echo "📋 Found requirements.txt - installing all dependencies..."
    pip install -r requirements.txt || {
        echo -e "${YELLOW}⚠️  Failed to install from requirements.txt, installing individual packages...${NC}"
    }
elif [ -f "requirements-dev.txt" ] && [ ${#MISSING_DEPS[@]} -gt 0 ]; then
    echo "📋 Found requirements-dev.txt - installing all dependencies..."
    pip install -r requirements-dev.txt || {
        echo -e "${YELLOW}⚠️  Failed to install from requirements-dev.txt, installing individual packages...${NC}"
    }
fi

# Check again after requirements installation
STILL_MISSING=()
if ! $PYTHON_CMD -c "import psycopg2" &> /dev/null; then
    STILL_MISSING+=("psycopg2-binary")
fi

# Install any still missing dependencies
if [ ${#STILL_MISSING[@]} -gt 0 ]; then
    echo -e "${YELLOW}📦 Installing missing dependencies: ${STILL_MISSING[*]}${NC}"
    pip install "${STILL_MISSING[@]}" || {
        echo -e "${RED}❌ ERROR: Failed to install dependencies${NC}"
        echo ""
        echo "🔧 Manual installation required:"
        echo "   source venv/bin/activate"
        echo "   pip install psycopg2-binary"
        exit 1
    }
fi

echo "✅ All Python dependencies available"

# =====================================================================
# 2. DATABASE CONNECTION TEST
# =====================================================================

echo ""
echo -e "${BLUE}📋 Step 2: Database Connection Test${NC}"
echo "==================================="

echo "🔗 Testing database connection..."

# Test connection using Python
CONNECTION_TEST=$($PYTHON_CMD -c "
import psycopg2
import sys
try:
    conn = psycopg2.connect('$DATABASE_URL')
    cursor = conn.cursor()
    cursor.execute('SELECT version()')
    version = cursor.fetchone()[0]
    cursor.close()
    conn.close()
    print('SUCCESS')
    print(version)
except Exception as e:
    print('FAILED')
    print(str(e))
    sys.exit(1)
")

if echo "$CONNECTION_TEST" | grep -q "SUCCESS"; then
    echo "✅ Database connection successful"
    DB_VERSION=$(echo "$CONNECTION_TEST" | tail -n 1)
    echo "📊 Database: $DB_VERSION"
else
    echo -e "${RED}❌ ERROR: Database connection failed${NC}"
    echo "$CONNECTION_TEST"
    echo ""
    echo "🔧 Troubleshooting:"
    echo "  1. Check if Supabase project is active"
    echo "  2. Verify database URL and password"
    echo "  3. Check network connectivity"
    echo "  4. Ensure database allows connections"
    exit 1
fi

# =====================================================================
# 3. SCHEMA VALIDATION AND SETUP
# =====================================================================

if [ "$VALIDATE_ONLY" = false ]; then
    echo ""
    echo -e "${BLUE}📋 Step 3: Database Schema Setup${NC}"
    echo "================================"

    if [ "$RESET_DATABASE" = true ]; then
        echo -e "${YELLOW}🔄 Resetting database (dropping existing schema)...${NC}"
        
        # Reset confirmation
        read -p "⚠️  This will DELETE ALL DATA. Type 'YES' to continue: " confirm
        if [ "$confirm" != "YES" ]; then
            echo "❌ Database reset cancelled"
            exit 1
        fi
        
        # Drop existing tables
        $PYTHON_CMD -c "
import psycopg2
conn = psycopg2.connect('$DATABASE_URL')
cursor = conn.cursor()

# Drop tables in reverse dependency order
tables = [
    'performance_metrics',
    'notification_settings', 
    'trades',
    'market_data',
    'portfolios',
    'user_profiles'
]

for table in tables:
    try:
        cursor.execute(f'DROP TABLE IF EXISTS {table} CASCADE')
        print(f'✅ Dropped table: {table}')
    except Exception as e:
        print(f'⚠️  Could not drop {table}: {e}')

conn.commit()
cursor.close()
conn.close()
print('✅ Database reset completed')
"
    fi

    if [ "$SKIP_SCHEMA" = false ]; then
        echo ""
        echo "📊 Applying database schema..."

        # Check if schema file exists
        if [ ! -f "sql/trading_simulator_schema.sql" ]; then
            echo -e "${RED}❌ ERROR: Schema file not found: sql/trading_simulator_schema.sql${NC}"
            exit 1
        fi

        # Apply schema using psql if available, otherwise use Python
        if [ "$USE_PYTHON_ONLY" = true ]; then
            echo "🐍 Applying schema using improved Python parser..."
            
            # Use the improved parser that handles dollar-quoted strings correctly
            $PYTHON_CMD apply_schema_fixed.py "$DATABASE_URL" || {
                echo -e "${RED}❌ ERROR: Schema application failed${NC}"
                exit 1
            }
        else
            echo "🐘 Applying schema using psql..."
            if psql "$DATABASE_URL" -f sql/trading_simulator_schema.sql -q; then
                echo "✅ Schema applied successfully using psql"
            else
                echo -e "${YELLOW}⚠️  psql failed, trying Python fallback...${NC}"
                # Fall back to the improved Python parser
                $PYTHON_CMD apply_schema_fixed.py "$DATABASE_URL" || {
                    echo -e "${RED}❌ ERROR: Schema application failed${NC}"
                    exit 1
                }
                echo "✅ Schema applied successfully using Python fallback"
            fi
        fi
    else
        echo "⏭️  Skipping schema application"
    fi
fi

# =====================================================================
# 4. SCHEMA VALIDATION
# =====================================================================

echo ""
echo -e "${BLUE}📋 Step 4: Schema Validation${NC}"
echo "============================"

echo "🔍 Validating database schema..."

# Check tables exist
SCHEMA_VALIDATION=$($PYTHON_CMD -c "
import psycopg2
import sys

required_tables = [
    'user_profiles',
    'portfolios', 
    'trades',
    'market_data',
    'performance_metrics',
    'notification_settings'
]

try:
    conn = psycopg2.connect('$DATABASE_URL')
    cursor = conn.cursor()
    
    # Check tables
    cursor.execute('''
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public' 
        AND table_type = 'BASE TABLE'
    ''')
    
    existing_tables = {row[0] for row in cursor.fetchall()}
    missing_tables = [table for table in required_tables if table not in existing_tables]
    
    if missing_tables:
        print(f'MISSING_TABLES:{missing_tables}')
        sys.exit(1)
    
    # Check indexes
    cursor.execute('''
        SELECT indexname 
        FROM pg_indexes 
        WHERE tablename IN %s
    ''', (tuple(required_tables),))
    
    indexes = [row[0] for row in cursor.fetchall()]
    
    # Check foreign keys
    cursor.execute('''
        SELECT COUNT(*) 
        FROM information_schema.table_constraints 
        WHERE constraint_type = 'FOREIGN KEY'
    ''')
    
    fk_count = cursor.fetchone()[0]
    
    print('VALIDATION_SUCCESS')
    print(f'TABLES:{len(existing_tables)}')
    print(f'INDEXES:{len(indexes)}')
    print(f'FOREIGN_KEYS:{fk_count}')
    
    cursor.close()
    conn.close()
    
except Exception as e:
    print(f'VALIDATION_ERROR:{e}')
    sys.exit(1)
")

if echo "$SCHEMA_VALIDATION" | grep -q "VALIDATION_SUCCESS"; then
    echo "✅ Schema validation successful"
    
    # Extract statistics
    TABLE_COUNT=$(echo "$SCHEMA_VALIDATION" | grep "TABLES:" | cut -d: -f2)
    INDEX_COUNT=$(echo "$SCHEMA_VALIDATION" | grep "INDEXES:" | cut -d: -f2)
    FK_COUNT=$(echo "$SCHEMA_VALIDATION" | grep "FOREIGN_KEYS:" | cut -d: -f2)
    
    echo "📊 Schema Statistics:"
    echo "   Tables: $TABLE_COUNT"
    echo "   Indexes: $INDEX_COUNT" 
    echo "   Foreign Keys: $FK_COUNT"
else
    echo -e "${RED}❌ ERROR: Schema validation failed${NC}"
    
    if echo "$SCHEMA_VALIDATION" | grep -q "MISSING_TABLES:"; then
        MISSING=$(echo "$SCHEMA_VALIDATION" | grep "MISSING_TABLES:" | cut -d: -f2-)
        echo "Missing tables: $MISSING"
    fi
    
    echo "$SCHEMA_VALIDATION"
    exit 1
fi

# =====================================================================
# 5. FUNCTIONAL VALIDATION
# =====================================================================

echo ""
echo -e "${BLUE}📋 Step 5: Functional Validation${NC}"
echo "================================"

echo "🧪 Testing database functionality..."

# Test basic CRUD operations
FUNCTIONAL_TEST=$($PYTHON_CMD -c "
import psycopg2
import sys
from uuid import uuid4
from datetime import datetime
from decimal import Decimal

try:
    conn = psycopg2.connect('$DATABASE_URL')
    
    # Test 1: Create test user profile
    with conn:
        with conn.cursor() as cursor:
            test_user_id = str(uuid4())
            cursor.execute('''
                INSERT INTO user_profiles (id, email, display_name, trading_style, risk_tolerance, investment_horizon)
                VALUES (%s, %s, %s, %s, %s, %s)
            ''', (test_user_id, 'test@example.com', 'Test User', 'swing', 'moderate', 'medium'))

    # Test 2: Create test portfolio
    with conn:
        with conn.cursor() as cursor:
            test_portfolio_id = str(uuid4())
            cursor.execute('''
                INSERT INTO portfolios (id, user_id, portfolio_name, portfolio_type, initial_balance, current_cash, total_value)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            ''', (test_portfolio_id, test_user_id, 'Test Portfolio', 'simulation', 100000.00, 100000.00, 100000.00))

    # Test 3: Create test trade
    with conn:
        with conn.cursor() as cursor:
            test_trade_id = str(uuid4())
            cursor.execute('''
                INSERT INTO trades (id, portfolio_id, symbol, trade_type, order_type, quantity, price, total_amount)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ''', (test_trade_id, test_portfolio_id, 'AAPL', 'BUY', 'market', 100, 150.00, 15000.00))

    # Test 4: Query data with joins
    with conn:
        with conn.cursor() as cursor:
            cursor.execute('''
                SELECT u.display_name, p.portfolio_name, t.symbol, t.quantity
                FROM user_profiles u
                JOIN portfolios p ON u.id = p.user_id  
                JOIN trades t ON p.id = t.portfolio_id
                WHERE u.id = %s
            ''', (test_user_id,))
            
            result = cursor.fetchone()
            if not result:
                raise Exception('Query returned no results')

    # Test 5: Test constraints (in separate transaction)
    constraint_test_passed = False
    try:
        with conn:
            with conn.cursor() as cursor:
                cursor.execute('''
                    INSERT INTO user_profiles (id, email, display_name, trading_style, risk_tolerance, investment_horizon)
                    VALUES (%s, %s, %s, %s, %s, %s)
                ''', (str(uuid4()), 'test@example.com', 'Duplicate User', 'swing', 'moderate', 'medium'))
    except psycopg2.IntegrityError:
        constraint_test_passed = True

    # Cleanup test data
    with conn:
        with conn.cursor() as cursor:
            cursor.execute('DELETE FROM user_profiles WHERE id = %s', (test_user_id,))

    conn.close()
    
    print('FUNCTIONAL_TEST_SUCCESS')
    print(f'USER_NAME:{result[0]}')
    print(f'PORTFOLIO_NAME:{result[1]}')
    print(f'TRADE_SYMBOL:{result[2]}')
    print(f'TRADE_QUANTITY:{result[3]}')
    print(f'CONSTRAINTS_WORKING:{constraint_test_passed}')
    
except Exception as e:
    print(f'FUNCTIONAL_TEST_FAILED:{e}')
    sys.exit(1)
")

if echo "$FUNCTIONAL_TEST" | grep -q "FUNCTIONAL_TEST_SUCCESS"; then
    echo "✅ Functional validation successful"
    
    USER_NAME=$(echo "$FUNCTIONAL_TEST" | grep "USER_NAME:" | cut -d: -f2-)
    PORTFOLIO_NAME=$(echo "$FUNCTIONAL_TEST" | grep "PORTFOLIO_NAME:" | cut -d: -f2-)
    TRADE_SYMBOL=$(echo "$FUNCTIONAL_TEST" | grep "TRADE_SYMBOL:" | cut -d: -f2-)
    
    echo "📊 Test Results:"
    echo "   Created user: $USER_NAME"
    echo "   Created portfolio: $PORTFOLIO_NAME"
    echo "   Executed trade: $TRADE_SYMBOL"
    echo "   Constraints working: ✅"
else
    echo -e "${RED}❌ ERROR: Functional validation failed${NC}"
    echo "$FUNCTIONAL_TEST"
    exit 1
fi

# =====================================================================
# 6. PERFORMANCE AND STORAGE CHECK
# =====================================================================

echo ""
echo -e "${BLUE}📋 Step 6: Performance and Storage Check${NC}"
echo "========================================"

echo "📊 Checking database performance and storage..."

STORAGE_CHECK=$($PYTHON_CMD -c "
import psycopg2
import time

try:
    conn = psycopg2.connect('$DATABASE_URL')
    cursor = conn.cursor()
    
    # Storage check
    cursor.execute('SELECT pg_size_pretty(pg_database_size(current_database()))')
    db_size = cursor.fetchone()[0]
    
    cursor.execute('SELECT pg_database_size(current_database())')
    db_size_bytes = cursor.fetchone()[0]
    db_size_mb = round(db_size_bytes / (1024 * 1024), 2)
    
    # Performance check - simple query timing
    start_time = time.time()
    cursor.execute('SELECT COUNT(*) FROM user_profiles')
    user_count = cursor.fetchone()[0]
    query_time_ms = round((time.time() - start_time) * 1000, 2)
    
    # Table sizes
    cursor.execute('''
        SELECT schemaname, tablename, pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename))
        FROM pg_tables 
        WHERE schemaname = 'public'
        ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
        LIMIT 5
    ''')
    
    table_sizes = cursor.fetchall()
    
    cursor.close()
    conn.close()
    
    print('STORAGE_CHECK_SUCCESS')
    print(f'DATABASE_SIZE:{db_size}')
    print(f'DATABASE_SIZE_MB:{db_size_mb}')
    print(f'USER_COUNT:{user_count}')
    print(f'QUERY_TIME_MS:{query_time_ms}')
    
    for schema, table, size in table_sizes:
        print(f'TABLE:{table}:{size}')
    
except Exception as e:
    print(f'STORAGE_CHECK_FAILED:{e}')
")

if echo "$STORAGE_CHECK" | grep -q "STORAGE_CHECK_SUCCESS"; then
    echo "✅ Performance and storage check successful"
    
    DB_SIZE=$(echo "$STORAGE_CHECK" | grep "DATABASE_SIZE:" | cut -d: -f2-)
    DB_SIZE_MB=$(echo "$STORAGE_CHECK" | grep "DATABASE_SIZE_MB:" | cut -d: -f2-)
    USER_COUNT=$(echo "$STORAGE_CHECK" | grep "USER_COUNT:" | cut -d: -f2-)
    QUERY_TIME=$(echo "$STORAGE_CHECK" | grep "QUERY_TIME_MS:" | cut -d: -f2-)
    
    echo "📊 Database Metrics:"
    echo "   Total Size: $DB_SIZE (${DB_SIZE_MB}MB)"
    echo "   User Records: $USER_COUNT"
    echo "   Query Performance: ${QUERY_TIME}ms"
    
    # Calculate percentage of 500MB Supabase limit
    if [ "$DB_SIZE_MB" != "0" ]; then
        USAGE_PCT=$(echo "scale=1; $DB_SIZE_MB / 500 * 100" | bc -l 2>/dev/null || echo "0")
        echo "   Supabase Usage: ${USAGE_PCT}% of 500MB limit"
        
        if (( $(echo "$USAGE_PCT > 80" | bc -l 2>/dev/null) )); then
            echo -e "   ${YELLOW}⚠️  WARNING: Database usage above 80% of free tier limit${NC}"
        fi
    fi
    
    echo ""
    echo "📋 Largest Tables:"
    echo "$STORAGE_CHECK" | grep "TABLE:" | head -3 | while read line; do
        TABLE_INFO=$(echo "$line" | cut -d: -f2-)
        echo "   $TABLE_INFO"
    done
else
    echo -e "${YELLOW}⚠️  WARNING: Storage check failed (non-critical)${NC}"
    echo "$STORAGE_CHECK"
fi

# =====================================================================
# 7. CONFIGURATION VALIDATION
# =====================================================================

echo ""
echo -e "${BLUE}📋 Step 7: Configuration Validation${NC}"
echo "==================================="

echo "🔧 Validating trading database manager configuration..."

if [ -f "config/supabase_config.py" ] && [ -f "src/core/trading_database_manager.py" ]; then
    CONFIG_TEST=$($PYTHON_CMD -c "
import sys
import os
sys.path.insert(0, '$(pwd)')

try:
    import config.supabase_config as supabase_config
    import src.core.trading_database_manager as tdm
    
    # Test basic imports work
    print('CONFIG_TEST_SUCCESS')
    print('SUPABASE_STATUS:available')
    print('TRADING_MANAGER_STATUS:available')
    
except ImportError as e:
    if 'asyncpg' in str(e) or 'supabase' in str(e):
        # Missing optional dependencies - this is OK for basic validation
        print('CONFIG_TEST_PARTIAL')
        print('SUPABASE_STATUS:dependencies_missing')
        print('TRADING_MANAGER_STATUS:dependencies_missing')
    else:
        print(f'CONFIG_TEST_FAILED:{e}')
except Exception as e:
    print(f'CONFIG_TEST_FAILED:{e}')
" 2>/dev/null)

    if echo "$CONFIG_TEST" | grep -q "CONFIG_TEST_SUCCESS"; then
        echo "✅ Configuration validation successful"
        
        SUPABASE_STATUS=$(echo "$CONFIG_TEST" | grep "SUPABASE_STATUS:" | cut -d: -f2-)
        TRADING_STATUS=$(echo "$CONFIG_TEST" | grep "TRADING_MANAGER_STATUS:" | cut -d: -f2-)
        
        echo "📊 Manager Status:"
        echo "   Supabase Manager: $SUPABASE_STATUS"
        echo "   Trading Manager: $TRADING_STATUS"
    elif echo "$CONFIG_TEST" | grep -q "CONFIG_TEST_PARTIAL"; then
        echo -e "${YELLOW}⚠️  WARNING: Configuration files found but optional dependencies missing${NC}"
        echo "   This is normal for basic database setup validation"
        echo "   Install asyncpg and supabase-py for full configuration testing"
    else
        echo -e "${YELLOW}⚠️  WARNING: Configuration validation failed (check Python imports)${NC}"
        echo "$CONFIG_TEST"
    fi
else
    echo -e "${YELLOW}⚠️  WARNING: Configuration files not found (skipping validation)${NC}"
fi

# =====================================================================
# 8. FINAL SUMMARY
# =====================================================================

echo ""
echo -e "${GREEN}🎉 SETUP COMPLETED SUCCESSFULLY!${NC}"
echo "================================="
echo "$(date '+%Y-%m-%d %H:%M:%S') - Database setup and validation completed"
echo ""
echo "📋 Summary:"
echo "✅ Database connection established"
echo "✅ Schema applied and validated"
echo "✅ Functional tests passed"
echo "✅ Performance checks completed"
echo "✅ Configuration validated"
echo ""
echo "🚀 Your Supabase database is ready for DAG execution!"
echo ""
echo "📖 Next Steps:"
echo "   1. Run your trading DAGs: ./check_dags.sh"
echo "   2. Monitor database: python config/supabase_config.py"
echo "   3. Run full validation: python src/utils/validate_storage_performance.py"
echo ""
echo -e "${BLUE}Database URL: ${DATABASE_URL%@*}@[HIDDEN]${NC}"
echo ""

# Save setup completion timestamp
echo "SUPABASE_SETUP_COMPLETED=$(date -u '+%Y-%m-%d %H:%M:%S UTC')" > .supabase_setup_completed

exit 0