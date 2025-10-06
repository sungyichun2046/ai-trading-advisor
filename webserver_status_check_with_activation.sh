#!/bin/bash
# Enhanced Webserver-Accurate DAG Status Report with Auto-Activation and Testing

# Function to wait for DAG completion with retries
wait_for_dag_completion() {
    local dag=$1
    local run_id=$2
    local max_wait=180  # 3 minutes max wait
    local check_interval=15  # Check every 15 seconds
    local elapsed=0
    
    echo "  ⏳ Waiting for DAG tasks to complete (max ${max_wait}s)..."
    
    while [[ $elapsed -lt $max_wait ]]; do
        sleep $check_interval
        elapsed=$((elapsed + check_interval))
        
        # Check current task states
        task_states=$(docker compose exec airflow-scheduler bash -c "airflow tasks states-for-dag-run $dag $run_id" 2>/dev/null | tail -n +3 | grep -v "^=" | grep -v "time=" | grep "| success\|| failed\|| running\|| queued" || echo "")
        
        if [[ -n "$task_states" ]]; then
            running_tasks=$(echo "$task_states" | grep -c "running" || echo "0")
            queued_tasks=$(echo "$task_states" | grep -c "queued" || echo "0")
            
            if [[ $running_tasks -eq 0 && $queued_tasks -eq 0 ]]; then
                echo "  ✅ All tasks completed after ${elapsed}s"
                return 0
            fi
            
            echo "  🔄 Still running... (${elapsed}s elapsed, $running_tasks running, $queued_tasks queued)"
        else
            echo "  ⚠️  No task data yet (${elapsed}s elapsed)"
        fi
    done
    
    echo "  ⏰ Timeout reached after ${max_wait}s - tasks may still be completing"
    return 1
}

echo "🚀 ENHANCED DAG STATUS REPORT WITH AUTO-ACTIVATION"
echo "=================================================="

# Get list of all DAGs
ALL_DAGS=$(docker compose exec airflow-scheduler airflow dags list | tail -n +3 | head -n -1 | awk '{print $1}')

echo "🔍 Checking for import errors..."
import_errors=$(docker compose exec airflow-scheduler airflow dags list-import-errors 2>/dev/null | tail -n +3 | head -n -1)
if [[ -n "$import_errors" && "$import_errors" != *"No DAG import errors"* ]]; then
    echo "🚨 IMPORT ERRORS DETECTED:"
    echo "$import_errors"
    echo ""
    echo "❌ Cannot proceed with activation due to import errors. Fix imports first."
    exit 1
else
    echo "✅ No import errors detected"
fi

echo ""
echo "📊 ANALYZING DAG STATUS AND AUTO-ACTIVATING NEVER-EXECUTED DAGs:"
echo "==============================================================="

successful=0
failed=0
running=0
unknown=0
import_error_count=0
never_executed_count=0
activated_dags=()
tested_dags=()

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
        echo "⚠️  NEVER EXECUTED - ACTIVATING NOW..."
        ((never_executed_count++))
        
        # Step 1: Unpause the DAG
        echo "  🔧 Unpausing DAG: $dag"
        docker compose exec airflow-scheduler airflow dags unpause $dag > /dev/null 2>&1
        
        # Step 2: Trigger a test run
        echo "  🎯 Triggering test run for: $dag"
        trigger_result=$(docker compose exec airflow-scheduler airflow dags trigger $dag 2>&1)
        
        if [[ $? -eq 0 ]]; then
            echo "  ✅ Successfully triggered: $dag"
            activated_dags+=($dag)
            
            # Step 3: Wait for task to start with progressive waiting
            echo "  ⏳ Waiting for DAG to initialize..."
            sleep 15
            
            # Step 4: Check if run was created
            new_run=$(docker compose exec airflow-scheduler bash -c "airflow dags list-runs -d $dag -o plain" | tail -n +2 | head -1 | awk '{print $2}')
            
            if [[ -n "$new_run" && "$new_run" != "No data found" ]]; then
                echo "  ✅ Run created successfully: $new_run"
                tested_dags+=($dag)
                
                # Wait for DAG completion with retries
                if wait_for_dag_completion "$dag" "$new_run"; then
                    # Check final task states after completion
                    task_states=$(docker compose exec airflow-scheduler bash -c "airflow tasks states-for-dag-run $dag $new_run" 2>/dev/null | tail -n +3 | grep -v "^=" | grep -v "time=" | grep "| success\|| failed\|| running\|| queued" || echo "")
                    
                    if [[ -n "$task_states" ]]; then
                        total_tasks=$(echo "$task_states" | wc -l)
                        success_tasks=$(echo "$task_states" | grep -c "success" || echo "0")
                        failed_tasks=$(echo "$task_states" | grep -c "failed" || echo "0")
                        running_tasks=$(echo "$task_states" | grep -c "running" || echo "0")
                        queued_tasks=$(echo "$task_states" | grep -c "queued" || echo "0")
                        
                        echo "  📊 Final Status: $success_tasks success, $failed_tasks failed, $running_tasks running, $queued_tasks queued (total: $total_tasks)"
                        
                        if [[ $failed_tasks -gt 0 ]]; then
                            echo "  ❌ SOME TASKS FAILED"
                            ((failed++))
                        elif [[ $success_tasks -eq $total_tasks && $total_tasks -gt 0 ]]; then
                            echo "  🎉 ALL TASKS COMPLETED SUCCESSFULLY"
                            ((successful++))
                        elif [[ $running_tasks -gt 0 || $queued_tasks -gt 0 ]]; then
                            echo "  🔄 STILL RUNNING (may need more time)"
                            ((running++))
                        else
                            echo "  ⚠️  INCOMPLETE STATUS"
                            ((unknown++))
                        fi
                    else
                        echo "  ⚠️  No task states available after wait"
                        ((unknown++))
                    fi
                else
                    # Timeout occurred, check current status
                    echo "  ⏰ DAG timed out, checking current status..."
                    task_states=$(docker compose exec airflow-scheduler bash -c "airflow tasks states-for-dag-run $dag $new_run" 2>/dev/null | tail -n +3 | grep -v "^=" | grep -v "time=" | grep "| success\|| failed\|| running\|| queued" || echo "")
                    
                    if [[ -n "$task_states" ]]; then
                        running_tasks=$(echo "$task_states" | grep -c "running" || echo "0")
                        queued_tasks=$(echo "$task_states" | grep -c "queued" || echo "0")
                        
                        if [[ $running_tasks -gt 0 || $queued_tasks -gt 0 ]]; then
                            echo "  🔄 STILL RUNNING (timeout but progressing)"
                            ((running++))
                        else
                            echo "  ⚠️  TIMEOUT - STATUS UNCLEAR"
                            ((unknown++))
                        fi
                    else
                        echo "  ❌ TIMEOUT WITH NO STATUS"
                        ((failed++))
                    fi
                fi
            else
                echo "  ❌ Failed to create run after triggering"
                ((failed++))
            fi
        else
            echo "  ❌ Failed to trigger: $trigger_result"
            ((failed++))
        fi
        continue
    fi
    
    # 3. Check task-level success for existing runs
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
        echo "🔄 RUNNING ($running_tasks/$total_tasks tasks running) - checking again in a moment..."
        # Brief wait for running tasks
        sleep 5
        
        # Re-check after brief wait
        updated_task_states=$(docker compose exec airflow-scheduler bash -c "airflow tasks states-for-dag-run $dag $recent_run" | tail -n +3 | grep -v "^=" | grep -v "time=" | grep "| success\|| failed\|| running")
        updated_running=$(echo "$updated_task_states" | grep -c "running" || echo "0")
        updated_success=$(echo "$updated_task_states" | grep -c "success" || echo "0")
        
        if [[ $updated_running -eq 0 && $updated_success -eq $total_tasks ]]; then
            echo "  ✅ Just completed! SUCCESS ($updated_success/$total_tasks tasks completed)"
            ((successful++))
        else
            echo "  🔄 Still running ($updated_running tasks)"
            ((running++))
        fi
    elif [[ $success_tasks -eq $total_tasks && $total_tasks -gt 0 ]]; then
        echo "✅ SUCCESS ($success_tasks/$total_tasks tasks completed)"
        ((successful++))
    else
        echo "⚠️  INCOMPLETE ($success_tasks/$total_tasks tasks successful)"
        ((unknown++))
    fi
done

echo ""
echo "🎯 AUTO-ACTIVATION SUMMARY:"
echo "=========================="
if [[ ${#activated_dags[@]} -gt 0 ]]; then
    echo "🔧 Activated DAGs (${#activated_dags[@]}):"
    for dag in "${activated_dags[@]}"; do
        echo "   ✅ $dag - unpaused and triggered"
    done
    echo ""
    
    if [[ ${#tested_dags[@]} -gt 0 ]]; then
        echo "🧪 Successfully Tested (${#tested_dags[@]}):"
        for dag in "${tested_dags[@]}"; do
            echo "   ✅ $dag - run created and executed"
        done
    fi
else
    echo "ℹ️  No DAGs needed activation (all had previous runs)"
fi

echo ""
echo "🏁 ENHANCED FINAL REPORT"
echo "======================="
echo "📊 Total DAGs: $((successful + failed + running + unknown))"
echo "✅ Successful: $successful"
echo "❌ Failed/Errored: $failed"
echo "🔄 Running: $running"
echo "⚠️  Incomplete/Unknown: $unknown"
echo "🔧 Auto-Activated: ${#activated_dags[@]}"

if [[ $import_error_count -gt 0 ]]; then
    echo "🚨 Import Errors: $import_error_count (CRITICAL)"
fi

if [[ $never_executed_count -gt 0 ]]; then
    echo "⚠️  Never Executed (now activated): $never_executed_count"
fi

# Final check for any remaining running DAGs
if [[ $running -gt 0 ]]; then
    echo ""
    echo "🔄 FINAL CHECK: $running DAGs still running, waiting 30 seconds for completion..."
    sleep 30
    
    # Re-check running DAGs
    final_successful=0
    final_running=0
    
    for dag in $ALL_DAGS; do
        recent_run=$(docker compose exec airflow-scheduler bash -c "airflow dags list-runs -d $dag -o plain" | tail -n +2 | head -1 | awk '{print $2}')
        
        if [[ -n "$recent_run" && "$recent_run" != "No data found" ]]; then
            task_states=$(docker compose exec airflow-scheduler bash -c "airflow tasks states-for-dag-run $dag $recent_run" | tail -n +3 | grep -v "^=" | grep -v "time=" | grep "| success\|| failed\|| running" 2>/dev/null)
            
            if [[ -n "$task_states" ]]; then
                total_tasks=$(echo "$task_states" | wc -l)
                success_tasks=$(echo "$task_states" | grep -c "success" || echo "0")
                running_tasks=$(echo "$task_states" | grep -c "running" || echo "0")
                
                if [[ $running_tasks -gt 0 ]]; then
                    ((final_running++))
                elif [[ $success_tasks -eq $total_tasks && $total_tasks -gt 0 ]]; then
                    ((final_successful++))
                fi
            fi
        fi
    done
    
    if [[ $final_running -eq 0 ]]; then
        echo "  ✅ All previously running DAGs have now completed!"
        running=0
        successful=$((successful + final_running))
    else
        echo "  🔄 $final_running DAGs still running (may need more time)"
        running=$final_running
    fi
fi

echo ""
echo "🎯 ENHANCED STATUS SUMMARY:"
if [[ $import_error_count -gt 0 ]]; then
    echo "❗ CRITICAL: $import_error_count DAGs have import errors - these will never work"
elif [[ $failed -gt 0 ]]; then
    echo "⚠️  $failed DAGs have execution failures"
    if [[ ${#activated_dags[@]} -gt 0 ]]; then
        echo "   📝 Note: ${#activated_dags[@]} DAGs were just activated and may need time to complete"
    fi
elif [[ $running -gt 0 ]]; then
    echo "🔄 $running DAGs are currently running"
    if [[ ${#activated_dags[@]} -gt 0 ]]; then
        echo "   📝 Note: ${#activated_dags[@]} DAGs were just activated and are likely among the running ones"
    fi
elif [[ $successful -gt 0 && $failed -eq 0 && $unknown -eq 0 ]]; then
    echo "🎉 ALL $successful DAGs WORKING PERFECTLY!"
    echo "   ✅ Status matches Airflow webserver exactly"
    if [[ ${#activated_dags[@]} -gt 0 ]]; then
        echo "   🚀 Successfully activated and tested ${#activated_dags[@]} new DAGs"
    fi
else
    echo "ℹ️  Mixed results - check individual DAG status above"
    if [[ ${#activated_dags[@]} -gt 0 ]]; then
        echo "   📝 Note: ${#activated_dags[@]} DAGs were just activated and may still be completing"
    fi
fi

echo ""
echo "💡 NEXT STEPS:"
if [[ ${#activated_dags[@]} -gt 0 ]]; then
    echo "   1. Wait 2-3 minutes for newly activated DAGs to complete"
    echo "   2. Re-run this script to see updated status"
    echo "   3. Check Airflow webserver: http://localhost:8080"
fi

echo ""
echo "🌐 This enhanced report activates and tests new DAGs automatically!"
echo "   Run again in a few minutes to see completion status of activated DAGs."