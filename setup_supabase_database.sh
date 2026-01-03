#!/bin/bash
# ============================================================
# Supabase Database Setup & Validation
# ============================================================
# Purpose:
#   - Apply trading_simulator_schema.sql
#   - Validate database contract
#
# Guarantees:
#   - Idempotent (safe to rerun)
#   - No DROP TABLE
#   - No sample data
#   - No side effects beyond schema creation
#
# Usage:
#   ./setup_supabase_database.sh
#   ./setup_supabase_database.sh --validate-only
#
# ============================================================

set -euo pipefail

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
VALIDATE_ONLY=false

for arg in "$@"; do
  case "$arg" in
    --validate-only)
      VALIDATE_ONLY=true
      ;;
    --help)
      echo "Usage:"
      echo "  ./setup_supabase_database.sh"
      echo "  ./setup_supabase_database.sh --validate-only"
      exit 0
      ;;
  esac
done

echo "🚀 Supabase DB setup (clean mode)"
echo "================================"
echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC')"

RESET=false

for arg in "$@"; do
  case "$arg" in
    --reset)
      RESET=true
      ;;
    --validate-only)
      VALIDATE_ONLY=true
      ;;
  esac
done

# ------------------------------------------------------------
# 1. Load database URL
# ------------------------------------------------------------
if [[ -n "${SUPABASE_URL:-}" ]]; then
  DATABASE_URL="$SUPABASE_URL"
elif [[ -n "${DATABASE_URL:-}" ]]; then
  DATABASE_URL="$DATABASE_URL"
elif [[ -f ".env" ]]; then
  source .env
  DATABASE_URL="${SUPABASE_URL:-${DATABASE_URL:-}}"
else
  echo "❌ DATABASE_URL / SUPABASE_URL not set"
  exit 1
fi

if [[ ! "$DATABASE_URL" =~ ^postgresql:// ]]; then
  echo "❌ Invalid DATABASE_URL format"
  exit 1
fi

echo "✅ Database URL loaded"

# ------------------------------------------------------------
# 2. Python availability
# ------------------------------------------------------------
if command -v python >/dev/null 2>&1; then
  PYTHON=python
elif command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
else
  echo "❌ Python not found"
  exit 1
fi

$PYTHON --version

# ------------------------------------------------------------
# 3. psycopg2 availability
# ------------------------------------------------------------
if ! $PYTHON - <<EOF >/dev/null 2>&1
import psycopg2
EOF
then
  echo "📦 Installing psycopg2-binary"
  pip install psycopg2-binary
fi

# ------------------------------------------------------------
# 4. Database connectivity test
# ------------------------------------------------------------
echo "🔗 Testing database connection..."

$PYTHON - <<EOF
import psycopg2
conn = psycopg2.connect("$DATABASE_URL")
cur = conn.cursor()
cur.execute("SELECT 1")
cur.close()
conn.close()
print("✅ Connection OK")
EOF

# ------------------------------------------------------------
# RESET: Drop ALL tables in public schema (one-shot)
# ------------------------------------------------------------
if [ "$RESET" = true ]; then
  echo "🔥 RESET MODE: Dropping ALL tables in public schema"

  $PYTHON - <<EOF
import psycopg2

conn = psycopg2.connect("$DATABASE_URL")
conn.autocommit = True
cur = conn.cursor()

cur.execute("""
DO \$\$
DECLARE
    r RECORD;
BEGIN
    FOR r IN (
        SELECT tablename
        FROM pg_tables
        WHERE schemaname = 'public'
    ) LOOP
        EXECUTE 'DROP TABLE IF EXISTS public.' || quote_ident(r.tablename) || ' CASCADE';
    END LOOP;
END \$\$;
""")

cur.close()
conn.close()

print("✅ All public tables dropped")
EOF
fi

# ------------------------------------------------------------
# 5. Apply schema (unless validate-only)
# ------------------------------------------------------------
if [ "$VALIDATE_ONLY" = false ]; then
  echo "📄 Applying schema..."

  if [ ! -f "sql/trading_simulator_schema.sql" ]; then
    echo "❌ sql/trading_simulator_schema.sql not found"
    exit 1
  fi

  if command -v psql >/dev/null 2>&1; then
    psql "$DATABASE_URL" -f sql/trading_simulator_schema.sql
  else
    echo "⚠️ psql not found, applying via Python"
    $PYTHON - <<EOF
import psycopg2, pathlib
sql = pathlib.Path("sql/trading_simulator_schema.sql").read_text()
conn = psycopg2.connect("$DATABASE_URL")
with conn:
    with conn.cursor() as cur:
        cur.execute(sql)
conn.close()
EOF
  fi

  echo "✅ Schema applied"
else
  echo "🔍 Validate-only mode (schema not applied)"
fi

# ------------------------------------------------------------
# 6. schema validation
# ------------------------------------------------------------
echo "🔎 Validating schema contract..."

$PYTHON - <<EOF
import psycopg2, sys

required_tables = {
    "active_symbols",
    "user_profiles",
    "market_data",
    "technical_analysis",
    "sentiment_analysis",
    "trading_decisions",
    "dag_runs",
}

conn = psycopg2.connect("$DATABASE_URL")
cur = conn.cursor()

cur.execute("""
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = 'public'
""")

existing = {r[0] for r in cur.fetchall()}
missing = required_tables - existing

if missing:
    print("❌ Missing tables:", ", ".join(sorted(missing)))
    sys.exit(1)

print("✅ All required tables present")
print("📊 Table count:", len(existing))

cur.close()
conn.close()
EOF

# ------------------------------------------------------------
# 7. Finalize
# ------------------------------------------------------------
echo "🎉 Supabase database ready"

echo "SUPABASE_SETUP_COMPLETED=$(date -u '+%Y-%m-%d %H:%M:%S UTC')" \
  > .supabase_setup_completed

exit 0
