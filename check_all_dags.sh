#!/bin/bash
# AI Trading Advisor - Complete DAG Status Checker
# Run all DAGs once and check their completion status

set -e

echo "🚀 AI TRADING ADVISOR - DAG EXECUTION CHECKER"
echo "============================================="
echo ""

# Check if Docker Compose is running
if ! docker compose ps airflow-scheduler | grep -q "Up"; then
    echo "❌ Airflow scheduler is not running. Starting services..."
    docker-compose up -d airflow-scheduler airflow-webserver
    echo "⏳ Waiting for Airflow to initialize..."
    sleep 15
fi

echo "📋 Step 1: Listing all available DAGs"
echo "======================================"
docker compose exec airflow-scheduler airflow dags list

echo ""
echo "🔄 Step 2: Triggering all active DAGs"
echo "===================================="

# Get list of all DAGs (both active and paused) - we'll unpause them all
ALL_DAGS=$(docker compose exec airflow-scheduler airflow dags list | tail -n +3 | head -n -1 | awk '{print $1}')

echo "🔧 Unpausing all DAGs first..."
for dag in $ALL_DAGS; do
    docker compose exec airflow-scheduler airflow dags unpause $dag > /dev/null 2>&1
done

# Get list of active DAGs (not paused) after unpausing
ACTIVE_DAGS=$(docker compose exec airflow-scheduler airflow dags list | grep "False" | awk '{print $1}' | grep -v "dag_id")

echo "Active DAGs found:"
for dag in $ACTIVE_DAGS; do
    echo "  - $dag"
done

echo ""
echo "Triggering DAG runs..."
for dag in $ACTIVE_DAGS; do
    echo "🎯 Triggering: $dag"
    docker compose exec airflow-scheduler airflow dags trigger $dag > /dev/null 2>&1
    sleep 2
done

echo ""
echo "⏳ Step 3: Waiting for DAG completion (60 seconds)"
echo "================================================="
sleep 60

echo ""
echo "🔍 Step 4: Pre-flight Checks (Import Errors & Historical Success)"
echo "================================================================="

# Check for import errors first
echo "🔎 Checking for DAG import errors..."
import_errors=$(docker compose exec airflow-scheduler airflow dags list-import-errors 2>/dev/null | tail -n +3 | head -n -1)
if [[ -n "$import_errors" && "$import_errors" != *"No DAG import errors"* ]]; then
    echo "⚠️  IMPORT ERRORS DETECTED:"
    echo "$import_errors"
    echo ""
else
    echo "✅ No import errors detected"
fi

echo ""
echo "📊 Step 5: Checking Real DAG Execution Status"
echo "============================================="

# Enhanced function to check REAL DAG status (webserver-accurate)
check_dag_status() {
    local dag=$1
    echo -n "📊 $dag: "
    
    # 1. Check for import errors first
    dag_import_error=$(echo "$import_errors" | grep "$dag" || echo "")
    if [[ -n "$dag_import_error" ]]; then
        echo "❌ IMPORT ERROR"
        return 1
    fi
    
    # 2. Get the most recent DAG run ID
    recent_run=$(docker compose exec airflow-scheduler bash -c "airflow dags list-runs -d $dag -o plain" | tail -n +2 | head -1 | awk '{print $2}')
    
    if [[ -z "$recent_run" || "$recent_run" == "No data found" ]]; then
        echo "⚠️  NEVER EXECUTED"
        return 3
    fi
    
    # 3. Check task-level success for the recent run
    task_states=$(docker compose exec airflow-scheduler bash -c "airflow tasks states-for-dag-run $dag $recent_run" | tail -n +3 | grep -v "^=" | grep -v "time=" | grep "| success\|| failed\|| running")
    
    if [[ -z "$task_states" ]]; then
        echo "⚠️  NO TASK DATA"
        return 3
    fi
    
    # Count task states
    total_tasks=$(echo "$task_states" | wc -l)
    success_tasks=$(echo "$task_states" | grep -c "success" || echo "0")
    failed_tasks=$(echo "$task_states" | grep -c "failed" || echo "0")
    running_tasks=$(echo "$task_states" | grep -c "running" || echo "0")
    
    # 4. Determine real status based on task analysis
    if [[ $failed_tasks -gt 0 ]]; then
        echo "❌ FAILED ($failed_tasks/$total_tasks tasks failed)"
        return 1
    elif [[ $running_tasks -gt 0 ]]; then
        echo "🔄 RUNNING ($running_tasks/$total_tasks tasks running)"
        return 2
    elif [[ $success_tasks -eq $total_tasks && $total_tasks -gt 0 ]]; then
        echo "✅ SUCCESS ($success_tasks/$total_tasks tasks completed)"
        return 0
    else
        echo "⚠️  INCOMPLETE ($success_tasks/$total_tasks tasks successful)"
        return 3
    fi
}

# Check each active DAG with enhanced validation
successful=0
failed=0
running=0
unknown=0
import_error_count=0
never_executed_count=0

echo ""
echo "🔍 Detailed DAG Analysis:"
echo "========================"

for dag in $ACTIVE_DAGS; do
    check_dag_status $dag
    status=$?
    case $status in
        0) ((successful++)) ;;
        1) 
            ((failed++))
            # Check if it's an import error specifically
            if echo "$import_errors" | grep -q "$dag"; then
                ((import_error_count++))
            fi
            ;;
        2) ((running++)) ;;
        3) 
            ((unknown++))
            # Check if it's a never-executed DAG
            recent_run=$(docker compose exec airflow-scheduler bash -c "airflow dags list-runs -d $dag -o plain" | tail -n +2 | head -1 | awk '{print $2}')
            if [[ -z "$recent_run" || "$recent_run" == "No data found" ]]; then
                ((never_executed_count++))
            fi
            ;;
    esac
done

echo ""
echo "🏁 WEBSERVER-ACCURATE FINAL REPORT"
echo "=================================="
echo "📊 Total DAGs Analyzed: $((successful + failed + running + unknown))"
echo "✅ Fully Successful: $successful"
echo "❌ Failed/Errored: $failed"
echo "🔄 Currently Running: $running"
echo "⚠️  Incomplete/Unknown: $unknown"

if [[ $import_error_count -gt 0 ]]; then
    echo "🚨 Import Errors: $import_error_count (CRITICAL - needs immediate fix)"
fi

if [[ $never_executed_count -gt 0 ]]; then
    echo "⚠️  Never Executed: $never_executed_count (needs investigation)"
fi

echo ""
echo "🎯 RECOMMENDATION:"
if [[ $import_error_count -gt 0 ]]; then
    echo "❗ CRITICAL: Fix import errors first - these DAGs will never work"
    echo "   Run: docker compose exec airflow-scheduler airflow dags list-import-errors"
    exit 1
elif [[ $failed -gt 0 ]]; then
    echo "⚠️  Some DAGs have execution failures"
    echo "   Check Airflow webserver: http://localhost:8080"
    echo "   Or check logs: make airflow-logs"
    exit 1
elif [[ $never_executed_count -gt 0 ]]; then
    echo "⚠️  Some DAGs have never executed successfully"
    echo "   Manual investigation needed: http://localhost:8080"
    exit 2
elif [[ $running -gt 0 ]]; then
    echo "⏳ Some DAGs still running - wait and re-run this script"
    exit 2
elif [[ $successful -eq $((successful + failed + running + unknown)) && $successful -gt 0 ]]; then
    echo "🎉 ALL DAGS WORKING PERFECTLY!"
    echo "   ✅ $successful/$successful DAGs executing successfully"
    echo "   🌐 Status matches Airflow webserver"
    exit 0
else
    echo "❓ Unexpected state detected - manual investigation needed"
    echo "   Visit Airflow webserver: http://localhost:8080"
    exit 3
fi