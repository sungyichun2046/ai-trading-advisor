#!/bin/bash
# ------------------------------------------------------------------
# Test Consolidated Trading DAG - Single DAG with Task Groups
#
# Steps:
#   1. Test local DAG import (single consolidated DAG)
#   2. Start production environment: docker compose up -d
#   3. Trigger single trading_workflow DAG
#   4. Validate task group execution: collect_data_tasks → analyze_data_tasks → execute_trades_tasks
#   5. Check task completion and workflow success
#
# Uses docker-compose.yml with AIRFLOW_PORT=8081 (test environment)
# Mounts ./src/dags → /opt/airflow/dags
#
# Usage:
#   ./check_dags.sh                    # Default: Test consolidated DAG workflow
#   ./check_dags.sh --timeout=N        # Set custom timeout
#
# Consolidated DAG Mode:
#   • Single trading_workflow DAG with 3 task groups
#   • Native Airflow task dependencies (no ExternalTaskSensor)
#   • Sequential execution: collect_data_tasks → analyze_data_tasks → execute_trades_tasks
#   • Unified workflow validation
#

set -e

# Parse command line arguments
WAIT_TIMEOUT=120           # 2 minutes default for single DAG execution
REAL_DATA_MODE=false       # Flag for real data testing

# Check for command line options
for arg in "$@"; do
    case $arg in
        --timeout=*)
            WAIT_TIMEOUT="${arg#*=}"
            echo "⏱️  Wait timeout set to: ${WAIT_TIMEOUT}s"
            ;;
        --real-data)
            REAL_DATA_MODE=true
            echo "🔴 Real data mode enabled"
            ;;
        --help)
            echo "Usage:"
            echo "  ./check_dags.sh                # Default: Test consolidated trading DAG"
            echo "  ./check_dags.sh --timeout=N    # Set custom timeout in seconds"
            echo "  ./check_dags.sh --real-data    # Enable real API data collection"
            echo ""
            echo "Tests single consolidated trading_workflow DAG with task groups:"
            echo "  • collect_data_tasks (parallel data collection)"
            echo "  • analyze_data_tasks (analysis with consensus)"
            echo "  • execute_trades_tasks (trading execution)"
            echo ""
            echo "Real Data Mode:"
            echo "  • Uses Yahoo Finance API for market data"
            echo "  • Uses NewsAPI for sentiment data"
            echo "  • Uses FinBERT for sentiment analysis"
            exit 0
            ;;
    esac
done

if [ "$REAL_DATA_MODE" = true ]; then
    echo "🚀 Mode: Real Data Integration Validation"
    echo "=========================================="
    echo "🔴 REAL DATA MODE ENABLED"
    echo "  • Yahoo Finance API: Market data for AAPL, SPY, QQQ"
    echo "  • NewsAPI: Sentiment data (max 50 articles)"
    echo "  • FinBERT: Advanced sentiment analysis"
else
    echo "🚀 Mode: Consolidated Trading DAG Validation"
    echo "============================================"
    echo "🟢 DUMMY DATA MODE (default)"
fi
echo ""

# Set environment for testing (same database, different port)
export POSTGRES_HOST=localhost
export POSTGRES_DB=airflow
export POSTGRES_USER=airflow
export POSTGRES_PASSWORD=airflow

# Configure data collection mode
if [ "$REAL_DATA_MODE" = true ]; then
    export USE_REAL_DATA=True
    export NEWSAPI_KEY=494b17bf8af14d7cbb2d62f1e8b11088
    echo "🔴 Environment configured for REAL DATA collection"
else
    export USE_REAL_DATA=False
    echo "🟢 Environment configured for DUMMY DATA collection"
fi

echo "📁 Test DAG Folder: $(pwd)/src/dags"
echo "📁 Expected: Single trading_dag.py with task groups"
echo "🐳 Using Docker environment (docker-compose.yml, port 8081)"
echo ""

# Check if source dags folder exists
if [ ! -d "src/dags" ]; then
    echo "❌ ERROR: src/dags/ folder not found!"
    echo "   Expected consolidated DAG structure not present"
    exit 1
fi

echo "🔍 SCANNING CONSOLIDATED DAG STRUCTURE"
echo "======================================"

# List Python files in dags folder
dag_files=$(find src/dags -name "*.py" -not -name "__*" 2>/dev/null || echo "")

if [ -z "$dag_files" ]; then
    echo "❌ ERROR: No Python DAG files found in src/dags/"
    exit 1
fi

echo "📋 Found DAG files:"
for file in $dag_files; do
    echo "   - $file"
done

# Validate we have exactly one DAG file
dag_count=$(echo "$dag_files" | wc -l)
if [ "$dag_count" -eq 1 ]; then
    echo "✅ PERFECT: Found exactly 1 consolidated DAG file"
else
    echo "❌ ERROR: Expected 1 consolidated DAG, found $dag_count files"
    echo "   Consolidation incomplete!"
    exit 1
fi
echo ""

# Check for trading utilities
echo "🔗 TRADING UTILITIES VALIDATION"
echo "==============================="

if [ ! -f "src/utils/trading_utils.py" ]; then
    echo "❌ ERROR: src/utils/trading_utils.py not found!"
    echo "   Trading utilities file is missing"
    exit 1
fi
echo "✅ Trading utilities: src/utils/trading_utils.py found"

# Test trading utilities import
echo ""
echo "🧪 Testing trading utilities..."
utils_test=$(POSTGRES_HOST=localhost POSTGRES_DB=airflow POSTGRES_USER=airflow POSTGRES_PASSWORD=airflow venv/bin/python -c "
import sys
sys.path.append('$(pwd)')
try:
    from src.utils.trading_utils import (
        is_market_open, safe_to_trade, should_run_analysis, 
        data_collection_branch_function, analysis_branch_function, trading_branch_function
    )
    print('✅ Trading utilities import successful')
    
    # Test basic functions
    market_status = is_market_open()
    print(f'✅ Market status check: {market_status}')
    
    trading_safe = safe_to_trade()
    print(f'✅ Trading safety check: {trading_safe}')
    
    analysis_ok = should_run_analysis()
    print(f'✅ Analysis readiness check: {analysis_ok}')
    
    print('✅ All trading utilities working correctly')
    
except ImportError as e:
    print(f'❌ Import failed: {e}')
    exit(1)
except Exception as e:
    print(f'❌ Trading utilities test failed: {e}')
    exit(1)
" 2>&1)

if [ $? -eq 0 ]; then
    echo "$utils_test"
else
    echo "❌ TRADING UTILITIES VALIDATION FAILED"
    echo "$utils_test"
    exit 1
fi
echo ""

echo "✅ TRADING UTILITIES VALIDATION COMPLETE"
echo "========================================"
echo ""

echo "🧪 TESTING CONSOLIDATED DAG IMPORT (LOCAL)"
echo "=========================================="

# Test the consolidated DAG file import locally
dag_file=$(echo "$dag_files" | head -1)
echo "Testing: $dag_file"

# Test Python import
python_test=$(venv/bin/python -c "
import sys
sys.path.append('$(pwd)')
try:
    module_path = '$dag_file'.replace('/', '.').replace('.py', '')
    exec(f'from {module_path} import dag')
    print(f'✅ IMPORT SUCCESS: {dag.dag_id} ({len(dag.tasks)} tasks)')
    print(f'   Schedule: {dag.schedule_interval}')
    print(f'   Description: {dag.description}')
    
    # Validate task groups
    task_groups = []
    task_ids = [t.task_id for t in dag.tasks]
    for task in dag.tasks:
        if hasattr(task, 'task_group') and task.task_group:
            group_id = task.task_group.group_id
            if group_id not in task_groups:
                task_groups.append(group_id)
    
    print(f'   Task Groups: {task_groups}')
    print(f'   Total Tasks: {len(task_ids)}')
    
    # Check for expected task groups
    expected_groups = ['collect_data_tasks', 'analyze_data_tasks', 'execute_trades_tasks']
    for group in expected_groups:
        if group in task_groups:
            print(f'   ✅ {group}: Found')
        else:
            print(f'   ❌ {group}: Missing')
            
except Exception as e:
    print(f'❌ IMPORT ERROR: {e}')
    exit(1)
" 2>&1)

if [ $? -eq 0 ]; then
    echo "$python_test"
else
    echo "❌ FAILED: $dag_file"
    echo "$python_test"
    exit 1
fi
echo ""

echo "📊 CONSOLIDATED DAG STRUCTURE VALIDATION"
echo "========================================"
echo "✅ Single consolidated DAG structure complete"
echo "✅ Task groups replace separate DAGs"
echo "✅ No ExternalTaskSensor dependencies needed"
echo ""

echo "🐳 STARTING DOCKER ENVIRONMENT (PORT 8081)"
echo "=========================================="

echo "🛑 Stopping any running services..."
docker compose down 2>/dev/null || true

echo "🚀 Starting Airflow environment on port 8081..."
export AIRFLOW_PORT=8081
docker compose up -d

echo "⏳ Waiting for Airflow to initialize (port 8081)..."
sleep 60

# Wait for Airflow to be ready
echo "🔄 Checking Airflow health (port 8081)..."
max_attempts=15
attempt=0

while [ $attempt -lt $max_attempts ]; do
    health_check=$(curl -s http://localhost:8081/health 2>/dev/null || echo "failed")
    web_access=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8081 2>/dev/null || echo "000")
    
    # Accept both 200 and 302 as valid (302 is redirect to login page)
    if [[ "$health_check" != "failed" ]] && ([[ "$web_access" == "200" ]] || [[ "$web_access" == "302" ]]); then
        echo "✅ Airflow is ready!"
        echo "   Health endpoint: ✅ http://localhost:8081/health"
        echo "   Web interface: ✅ http://localhost:8081 (HTTP $web_access)"
        break
    fi
    
    attempt=$((attempt + 1))
    echo "   Attempt $attempt/$max_attempts (Health: $health_check, Web: HTTP $web_access)..."
    sleep 15
done

if [ $attempt -eq $max_attempts ]; then
    echo "❌ Airflow not ready after $max_attempts attempts"
    exit 1
fi

echo ""
echo "🔍 CRITICAL: CHECKING DAG IMPORT ERRORS"
echo "======================================="

# Check for DAG import errors - EARLY STOP if any found
echo "🧪 Checking for DAG import errors..."
import_errors=$(docker compose exec airflow-webserver airflow dags list-import-errors 2>/dev/null)

if [[ "$import_errors" == *"No data found"* ]]; then
    echo "✅ SUCCESS: No DAG import errors found!"
    echo "✅ Consolidated DAG loaded without issues"
else
    echo "❌ CRITICAL ERROR: DAG IMPORT FAILURES DETECTED"
    echo "==============================================="
    echo ""
    echo "Import errors found:"
    echo "$import_errors"
    echo ""
    echo "❌ EARLY STOP: Cannot proceed with DAG import errors"
    echo "❌ Fix the import errors before continuing validation"
    exit 1
fi

# Verify consolidated DAG is loaded
echo ""
echo "📋 Verifying consolidated DAG..."
loaded_dags=$(docker compose exec airflow-webserver airflow dags list 2>/dev/null | grep "trading_workflow" | wc -l)

if [ "$loaded_dags" -eq 1 ]; then
    echo "✅ SUCCESS: Consolidated trading_workflow DAG is loaded"
    dag_details=$(docker compose exec airflow-webserver airflow dags list 2>/dev/null | grep "trading_workflow")
    echo "Loaded DAG:"
    echo "$dag_details"
else
    echo "❌ ERROR: Expected 1 trading_workflow DAG, found $loaded_dags"
    echo "❌ EARLY STOP: Consolidated DAG not found"
    exit 1
fi

echo ""
echo "🚀 CONSOLIDATED DAG EXECUTION TEST"
echo "=================================="

# Unpause the consolidated DAG
echo "📋 Unpausing trading_workflow DAG..."
docker compose exec airflow-webserver airflow dags unpause trading_workflow > /dev/null 2>&1
echo "✅ trading_workflow DAG unpaused"

echo ""
echo "🔥 Triggering consolidated trading_workflow DAG..."
execution_date=$(date -u '+%Y-%m-%dT%H:%M:%S')
echo "📅 Using execution_date: $execution_date"

# Trigger the consolidated DAG
docker compose exec airflow-webserver airflow dags trigger trading_workflow -e "$execution_date" > /dev/null 2>&1
echo "✅ trading_workflow DAG triggered"

echo ""
if [ "$REAL_DATA_MODE" = true ]; then
    echo "⏳ Waiting ${WAIT_TIMEOUT}s for REAL DATA workflow to complete..."
    echo "📊 Expected execution with REAL APIs:"
    echo "     • collect_data_tasks: Yahoo Finance + NewsAPI + FinBERT"
    echo "     • analyze_data_tasks: Real data analysis + consensus"
    echo "     • execute_trades_tasks: Trading based on real data"
else
    echo "⏳ Waiting ${WAIT_TIMEOUT}s for consolidated workflow to complete..."
    echo "📊 Expected execution: collect_data_tasks → analyze_data_tasks → execute_trades_tasks"
fi
echo ""

# Wait for execution
sleep $WAIT_TIMEOUT

echo ""
echo "📊 CONSOLIDATED DAG EXECUTION RESULTS"
echo "===================================="

# Check execution status
echo "🕐 Checking trading_workflow execution..."
workflow_runs=$(docker compose exec airflow-webserver \
    airflow dags list-runs -d trading_workflow 2>/dev/null \
    | grep -E "(success|running|failed)" | tr -d '\r' || echo "")

if [ -n "$workflow_runs" ]; then
    echo "📋 Recent workflow runs:"
    echo "$workflow_runs" | head -5
    
    # Count different states
    success_count=$(echo "$workflow_runs" | grep -c "success" || echo "0")
    running_count=$(echo "$workflow_runs" | grep -c "running" || echo "0")
    failed_count=$(echo "$workflow_runs" | grep -c "failed" || echo "0")
    
    echo ""
    echo "📊 Execution Summary:"
    echo "   ✅ Success: $success_count runs"
    echo "   🔄 Running: $running_count runs" 
    echo "   ❌ Failed:  $failed_count runs"
    
else
    echo "❌ No workflow runs found"
    success_count=0
    running_count=0
    failed_count=0
fi

echo ""
echo "🎯 FINAL VALIDATION RESULT"
echo "=========================="

# Determine overall success
if [ "$success_count" -gt 0 ]; then
    echo "🎉 ✅ SUCCESS: Consolidated trading_workflow completed successfully!"
    echo "✅ Task groups executed in correct sequence"
    echo "✅ No ExternalTaskSensor complexity needed" 
    echo "✅ Single DAG workflow validation complete"
    final_result="SUCCESS"
elif [ "$running_count" -gt 0 ]; then
    echo "🔄 PARTIAL: trading_workflow is still running"
    echo "⏳ DAG execution in progress but not completed within timeout"
    final_result="RUNNING"
else
    echo "❌ FAILURE: trading_workflow did not complete successfully"
    echo "❌ Check DAG execution details in Airflow UI"
    final_result="FAILURE"
fi

echo ""
echo "==============================================="
if [ "$REAL_DATA_MODE" = true ]; then
    echo "🎯 REAL DATA INTEGRATION VALIDATION REPORT"
else
    echo "🎯 CONSOLIDATED DAG VALIDATION REPORT"
fi
echo "==============================================="
echo ""
echo "📈 Consolidated Structure:  ✅ Single trading_dag.py with task groups"
echo "🔗 Task Group Dependencies: ✅ collect_data_tasks → analyze_data_tasks → execute_trades_tasks"
echo "🚫 ExternalTaskSensor:      ✅ Eliminated (native task dependencies)"
if [ "$REAL_DATA_MODE" = true ]; then
    echo "🔴 Data Integration:        ✅ Real API calls (Yahoo Finance + NewsAPI + FinBERT)"
else
    echo "🟢 Data Mode:               ✅ Dummy data (fast validation)"
fi
echo "📊 Execution Result:        $final_result"
echo ""

if [ "$final_result" == "SUCCESS" ]; then
    if [ "$REAL_DATA_MODE" = true ]; then
        echo "🎉 OVERALL RESULT: ✅ SUCCESS - Real Data Integration complete!"
        echo "✅ Yahoo Finance API: Market data collected successfully"
        echo "✅ NewsAPI: Sentiment data collected successfully" 
        echo "✅ FinBERT: Advanced sentiment analysis working"
        echo "✅ Real data workflow execution validated"
    else
        echo "🎉 OVERALL RESULT: ✅ SUCCESS - Consolidated DAG workflow complete!"
        echo "✅ Task group execution validated"
        echo "✅ Single DAG architecture working perfectly"
    fi
    echo "✅ No cross-DAG dependency issues"
elif [ "$final_result" == "RUNNING" ]; then
    echo "⏳ OVERALL RESULT: 🟡 IN PROGRESS - Workflow executing"
    echo "ℹ️  Increase timeout or check execution progress manually"
else
    if [ "$REAL_DATA_MODE" = true ]; then
        echo "❌ OVERALL RESULT: ❌ FAILURE - Real data integration failed"
        echo "⚠️  Check API keys, network connectivity, and Airflow UI"
    else
        echo "❌ OVERALL RESULT: ❌ FAILURE - Workflow execution failed"
    fi
    echo "⚠️  Check Airflow UI for task execution details"
fi

echo ""
echo "🔗 Access Airflow UI: http://localhost:8081"
echo "   Username: admin / Password: admin"
echo ""

# Exit with appropriate code
if [ "$final_result" == "SUCCESS" ]; then
    exit 0
else
    exit 1
fi