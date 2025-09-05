#!/bin/bash

echo "Running final fix for AI Trading Advisor..."
echo "Checking and triggering fundamental_pipeline and volatility_monitoring DAGs..."

# Check if Airflow is running
if ! docker ps | grep -q airflow-scheduler; then
    echo "❌ Airflow scheduler container is not running. Start it with 'make airflow-start'"
    exit 1
fi

echo "✅ Airflow is running"

# Check DAG status
echo "Checking DAG status..."
docker exec ai-trading-advisor-airflow-scheduler-1 airflow dags list | grep -E "(fundamental|volatility)" || echo "DAGs not found"

# Try to unpause DAGs first (if they're paused, they can't be triggered)
echo "Unpausing DAGs..."
docker exec ai-trading-advisor-airflow-scheduler-1 airflow dags unpause fundamental_pipeline 2>/dev/null && echo "✅ fundamental_pipeline unpaused" || echo "⚠️ Could not unpause fundamental_pipeline"
docker exec ai-trading-advisor-airflow-scheduler-1 airflow dags unpause volatility_monitoring 2>/dev/null && echo "✅ volatility_monitoring unpaused" || echo "⚠️ Could not unpause volatility_monitoring"

# Trigger DAGs
echo "Triggering DAGs..."
docker exec ai-trading-advisor-airflow-scheduler-1 airflow dags trigger fundamental_pipeline 2>/dev/null && echo "✅ fundamental_pipeline triggered" || echo "❌ Failed to trigger fundamental_pipeline"
docker exec ai-trading-advisor-airflow-scheduler-1 airflow dags trigger volatility_monitoring 2>/dev/null && echo "✅ volatility_monitoring triggered" || echo "❌ Failed to trigger volatility_monitoring"

echo ""
echo "✨ Final fix completed!"
echo "📊 You can check DAG runs at: http://localhost:8080 (admin/admin)"
echo "🔍 Use 'make airflow-dags-list' to check DAG status"