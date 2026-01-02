# 🚀 Quick Start: Supabase Database Setup

**One-command setup** for the multi-profile trading simulator database.

## 📋 Prerequisites (5 minutes)

### 1. **Create Supabase Account** (Free)
```bash
# Go to: https://supabase.com
# Sign up (free account)
# Create new project: "trading-simulator"
# Remember your password!
```

### 2. **Get Your Database URL**
In Supabase Dashboard:
1. Go to **Settings** → **Database**
2. Find **Connection info** section
3. Copy the connection details

Your URL format will be:
```bash
postgresql://postgres:YOUR_PASSWORD@db.YOUR_PROJECT_REF.supabase.co:5432/postgres
```

**Real examples:**
```bash
postgresql://postgres:SecurePass123@db.abcdefghijklmn.supabase.co:5432/postgres
postgresql://postgres:MyPassword@db.xyzuvwrstqpkjh.supabase.co:5432/postgres
```

## ⚡ One-Command Setup

### **Option 1: Environment Variable**
```bash
# Set your Supabase URL (replace with your actual URL)
export SUPABASE_URL="postgresql://postgres:YOUR_PASSWORD@db.YOUR_PROJECT_REF.supabase.co:5432/postgres"

# Run setup (handles everything automatically)
./setup_supabase_database.sh
```

### **Option 2: .env File**
```bash
# Create .env file with your URL
echo 'SUPABASE_URL="postgresql://postgres:YOUR_PASSWORD@db.YOUR_PROJECT_REF.supabase.co:5432/postgres"' > .env

# Run setup
./setup_supabase_database.sh
```

## ✅ What the Script Does Automatically

### **🐍 Python Environment:**
- ✅ **Creates virtual environment** if not exists (`venv/`)
- ✅ **Activates virtual environment** automatically
- ✅ **Installs dependencies** from `requirements-dev.txt` or individual packages
- ✅ **Handles missing psycopg2** and other dependencies

### **🗄️ Database Setup:**
- ✅ **Tests connection** to Supabase
- ✅ **Applies complete schema** (6 tables, indexes, constraints)
- ✅ **Validates table structure** and relationships
- ✅ **Tests CRUD operations** with sample data
- ✅ **Checks performance** and storage usage

### **📊 Validation:**
- ✅ **Schema validation** (tables, indexes, foreign keys)
- ✅ **Functional testing** (create users, portfolios, trades)
- ✅ **Performance benchmarks** (query speed)
- ✅ **Storage monitoring** (500MB Supabase limit)
- ✅ **Configuration validation** (Python imports)

## 📺 Expected Output

```bash
🚀 SUPABASE DATABASE SETUP AND VALIDATION
==============================================

🐍 Checking Python environment...
📦 Creating new virtual environment...
✅ Virtual environment created and activated: /path/to/venv
📋 Found requirements-dev.txt - installing all dependencies...
✅ All Python dependencies available

📋 Step 1: Environment Validation
✅ Using SUPABASE_URL from environment
✅ Using Python: Python 3.11.0

📋 Step 2: Database Connection Test
✅ Database connection successful
📊 Database: PostgreSQL 15.0

📋 Step 3: Database Schema Setup
📊 Applying database schema...
✅ Schema applied successfully

📋 Step 4: Schema Validation
✅ Schema validation successful
📊 Schema Statistics:
   Tables: 6
   Indexes: 12
   Foreign Keys: 4

📋 Step 5: Functional Validation
✅ Functional validation successful
📊 Test Results:
   Created user: Test User
   Created portfolio: Test Portfolio
   Executed trade: AAPL

📋 Step 6: Performance and Storage Check
✅ Performance and storage check successful
📊 Database Metrics:
   Total Size: 8.2 MB
   Query Performance: 45ms
   Supabase Usage: 1.6% of 500MB limit

🎉 SETUP COMPLETED SUCCESSFULLY!
=================================

🚀 Your Supabase database is ready for DAG execution!

📖 Next Steps:
   1. Run your trading DAGs: ./check_dags.sh
   2. Monitor database: python config/supabase_config.py
```

## 🛠️ Advanced Usage

### **Validate Existing Setup:**
```bash
./setup_supabase_database.sh --validate-only
```

### **Reset Database (⚠️ Deletes all data):**
```bash
./setup_supabase_database.sh --reset
```

### **Skip Schema Application:**
```bash
./setup_supabase_database.sh --skip-schema
```

## 🔧 Manual Steps (if needed)

### **Create Virtual Environment Manually:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt
```

### **Install Dependencies Manually:**
```bash
pip install psycopg2-binary
```

### **Test Connection Manually:**
```bash
python -c "
import psycopg2
conn = psycopg2.connect('$SUPABASE_URL')
print('✅ Connection successful!')
conn.close()
"
```

## ❌ Troubleshooting

### **"Python not found"**
```bash
# Install Python 3.8+
sudo apt update && sudo apt install python3 python3-venv python3-pip

# Or on macOS:
brew install python3
```

### **"psycopg2 installation failed"**
```bash
# Install system dependencies
sudo apt install libpq-dev python3-dev  # Ubuntu/Debian
brew install postgresql                  # macOS

# Then retry:
pip install psycopg2-binary
```

### **"Database connection failed"**
```bash
# Check URL format:
echo $SUPABASE_URL

# Should look like:
# postgresql://postgres:password@db.project.supabase.co:5432/postgres

# Test with psql:
psql "$SUPABASE_URL" -c "SELECT version();"
```

### **"Schema application failed"**
```bash
# Check if Supabase project is active
# Verify you have database write permissions
# Try manual schema application in Supabase SQL Editor
```

## 🚀 Complete Workflow

```bash
# 1. Get Supabase URL from dashboard
export SUPABASE_URL="postgresql://postgres:your-password@db.your-project.supabase.co:5432/postgres"

# 2. One-command setup (handles everything)
./setup_supabase_database.sh

# 3. Run your trading system
./check_dags.sh

# 4. Monitor database
python config/supabase_config.py
```

**That's it!** The script handles Python virtual environments, dependencies, database setup, and validation automatically. 🎉

## 📚 Files Created

After successful setup, you'll have:
- ✅ `venv/` - Python virtual environment with dependencies
- ✅ `.supabase_setup_completed` - Setup completion timestamp
- ✅ Database with 6 tables ready for trading data
- ✅ All indexes and constraints applied
- ✅ Validation tests passed

**Ready for production trading simulator use!** 🚀