#!/bin/bash
# Webserver-Accurate DAG Status Report (No Triggering)

echo "🌐 WEBSERVER-ACCURATE DAG STATUS REPORT"
echo "======================================"

# Get list of all DAGs
ALL_DAGS=$(docker compose exec airflow-scheduler airflow dags list | tail -n +3 | head -n -1 | awk '{print $1}')

echo "🔍 Checking for import errors..."
import_errors=$(docker compose exec airflow-scheduler airflow dags list-import-errors 2>/dev/null | tail -n +3 | head -n -1)
if [[ -n "$import_errors" && "$import_errors" != *"No DAG import errors"* ]]; then
    echo "🚨 IMPORT ERRORS DETECTED:"
    echo "$import_errors"
else
    echo "✅ No import errors detected"
fi

echo ""
echo "📊 REAL DAG STATUS (matches webserver view):"
echo "==========================================="

successful=0
failed=0
running=0
unknown=0
import_error_count=0
never_executed_count=0

for dag in $ALL_DAGS; do
    echo -n "📊 $dag: "
    
    # 1. Check for import errors first
    dag_import_error=$(echo "$import_errors" | grep "$dag" || echo "")
    if [[ -n "$dag_import_error" ]]; then
        echo "❌ IMPORT ERROR"
        ((failed++))
        ((import_error_count++))
        continue
    fi
    
    # 2. Get the most recent DAG run ID
    recent_run=$(docker compose exec airflow-scheduler bash -c "airflow dags list-runs -d $dag -o plain" | tail -n +2 | head -1 | awk '{print $2}')
    
    if [[ -z "$recent_run" || "$recent_run" == "No data found" ]]; then
        echo "⚠️  NEVER EXECUTED"
        ((unknown++))
        ((never_executed_count++))
        continue
    fi
    
    # 3. Check task-level success for the recent run
    task_states=$(docker compose exec airflow-scheduler bash -c "airflow tasks states-for-dag-run $dag $recent_run" | tail -n +3 | grep -v "^=" | grep -v "time=" | grep "| success\|| failed\|| running")
    
    if [[ -z "$task_states" ]]; then
        echo "⚠️  NO TASK DATA"
        ((unknown++))
        continue
    fi
    
    # Count task states
    total_tasks=$(echo "$task_states" | wc -l)
    success_tasks=$(echo "$task_states" | grep -c "success")
    failed_tasks=$(echo "$task_states" | grep -c "failed")
    running_tasks=$(echo "$task_states" | grep -c "running")
    
    # Determine real status based on task analysis
    if [[ $failed_tasks -gt 0 ]]; then
        echo "❌ FAILED ($failed_tasks/$total_tasks tasks failed)"
        ((failed++))
    elif [[ $running_tasks -gt 0 ]]; then
        echo "🔄 RUNNING ($running_tasks/$total_tasks tasks running)"
        ((running++))
    elif [[ $success_tasks -eq $total_tasks && $total_tasks -gt 0 ]]; then
        echo "✅ SUCCESS ($success_tasks/$total_tasks tasks completed)"
        ((successful++))
    else
        echo "⚠️  INCOMPLETE ($success_tasks/$total_tasks tasks successful)"
        ((unknown++))
    fi
done

echo ""
echo "🏁 WEBSERVER-ACCURATE FINAL REPORT"
echo "=================================="
echo "📊 Total DAGs: $((successful + failed + running + unknown))"
echo "✅ Successful: $successful"
echo "❌ Failed/Errored: $failed"
echo "🔄 Running: $running"
echo "⚠️  Incomplete/Unknown: $unknown"

if [[ $import_error_count -gt 0 ]]; then
    echo "🚨 Import Errors: $import_error_count (CRITICAL)"
fi

if [[ $never_executed_count -gt 0 ]]; then
    echo "⚠️  Never Executed: $never_executed_count"
fi

echo ""
echo "🎯 STATUS SUMMARY:"
if [[ $import_error_count -gt 0 ]]; then
    echo "❗ CRITICAL: $import_error_count DAGs have import errors - these will never work"
elif [[ $failed -gt 0 ]]; then
    echo "⚠️  $failed DAGs have execution failures"
elif [[ $never_executed_count -gt 0 ]]; then
    echo "⚠️  $never_executed_count DAGs have never executed successfully"
elif [[ $successful -gt 0 && $failed -eq 0 && $unknown -eq 0 ]]; then
    echo "🎉 ALL $successful DAGs WORKING PERFECTLY!"
    echo "   ✅ Status matches Airflow webserver exactly"
else
    echo "ℹ️  Mixed results - check individual DAG status above"
fi

echo ""
echo "🌐 This report now matches exactly what you see in Airflow webserver!"