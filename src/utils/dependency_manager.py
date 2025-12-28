"""
Dynamic DAG Dependency Manager

Configuration-driven dependency management using shared utilities.
Provides conditional skip logic, dynamic timeouts, and cross-DAG coordination.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from airflow.models import DAG, BaseOperator
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.utils.trigger_rule import TriggerRule

# Import shared utilities
from .config_loader import load_yaml_config
from .environment_utils import get_environment, is_market_open, get_market_session
from .caching_manager import cache_cross_dag_data, get_cache_manager
from .alert_manager import send_prioritized_alert

logger = logging.getLogger(__name__)


class DependencyManager:
    """
    Manages DAG dependencies based on configuration and market conditions.
    
    Features:
    - Configuration-driven skip conditions
    - Dynamic task timeouts based on environment
    - Cross-DAG dependency validation
    - Market-aware execution logic
    - Caching of dependency states
    """
    
    def __init__(self, config_path: str = "src/config/dag_dependencies.yaml"):
        """Initialize dependency manager with configuration."""
        self.config_path = config_path
        self.config = None
        self.environment = get_environment()
        self.cache_manager = None
        
        try:
            self.config = load_yaml_config(config_path)
            self.cache_manager = get_cache_manager()
            logger.info(f"DependencyManager initialized for environment: {self.environment}")
        except Exception as e:
            logger.error(f"Failed to initialize DependencyManager: {e}")
            self.config = self._get_fallback_config()
    
    def _get_fallback_config(self) -> Dict[str, Any]:
        """Fallback configuration when main config fails to load."""
        return {
            'global': {'cache_dependencies': False, 'cache_ttl': 300, 'enable_skip_conditions': False},
            'dags': {},
            'environment_overrides': {}
        }
    
    def setup_dependencies(self, dag: DAG, dag_id: str) -> DAG:
        """
        Set up dependencies for a DAG based on configuration.
        
        Args:
            dag: Airflow DAG object
            dag_id: DAG identifier
            
        Returns:
            Enhanced DAG with dependency logic
        """
        try:
            if not self.config or dag_id not in self.config.get('dags', {}):
                logger.warning(f"No dependency config found for DAG: {dag_id}")
                return dag
            
            dag_config = self.config['dags'][dag_id]
            
            # Apply environment overrides
            dag_config = self._apply_environment_overrides(dag_config)
            
            # Add skip condition checks
            if self.config.get('global', {}).get('enable_skip_conditions', True):
                self._add_skip_conditions(dag, dag_id, dag_config)
            
            # Apply task timeouts
            self._apply_task_timeouts(dag, dag_config)
            
            # Add monitoring tasks
            self._add_monitoring_tasks(dag, dag_id, dag_config)
            
            # Cache DAG dependency state
            self._cache_dag_state(dag_id, dag_config)
            
            logger.info(f"Dependencies configured for DAG: {dag_id}")
            return dag
            
        except Exception as e:
            logger.error(f"Error setting up dependencies for {dag_id}: {e}")
            return dag
    
    def _apply_environment_overrides(self, dag_config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment-specific configuration overrides."""
        env_overrides = self.config.get('environment_overrides', {}).get(self.environment, {})
        
        if env_overrides:
            # Merge skip conditions
            if 'skip_conditions' in env_overrides:
                dag_config.setdefault('skip_conditions', {}).update(env_overrides['skip_conditions'])
            
            # Apply timeout overrides
            if 'task_timeouts' in env_overrides:
                default_timeout = env_overrides['task_timeouts'].get('default')
                if default_timeout:
                    for task_id in dag_config.get('task_dependencies', {}):
                        dag_config['task_dependencies'][task_id].setdefault('timeout', default_timeout)
        
        return dag_config
    
    def _add_skip_conditions(self, dag: DAG, dag_id: str, dag_config: Dict[str, Any]):
        """Add skip condition logic to DAG."""
        skip_conditions = dag_config.get('skip_conditions', {})
        
        if not skip_conditions:
            return
        
        # Create skip check task
        skip_check_task = BranchPythonOperator(
            task_id=f'{dag_id}_skip_check',
            python_callable=self._evaluate_skip_conditions,
            op_kwargs={
                'dag_id': dag_id,
                'skip_conditions': skip_conditions
            },
            dag=dag
        )
        
        # Create skip task
        skip_task = DummyOperator(
            task_id=f'{dag_id}_skip',
            dag=dag
        )
        
        # Create proceed task
        proceed_task = DummyOperator(
            task_id=f'{dag_id}_proceed',
            dag=dag
        )
        
        # Set up branching logic
        skip_check_task >> [skip_task, proceed_task]
        
        # Make all other tasks depend on proceed_task
        for task in dag.tasks:
            if task.task_id not in [f'{dag_id}_skip_check', f'{dag_id}_skip', f'{dag_id}_proceed']:
                proceed_task >> task
    
    def _evaluate_skip_conditions(self, dag_id: str, skip_conditions: Dict[str, Any], **context) -> str:
        """
        Evaluate skip conditions and return next task.
        
        Returns:
            Task ID to execute next
        """
        try:
            current_env = get_environment()
            
            for condition_name, condition_config in skip_conditions.items():
                if not condition_config.get('enabled', False):
                    continue
                
                # Check if condition applies to current environment
                if 'environments' in condition_config:
                    if current_env not in condition_config['environments']:
                        continue
                
                # Evaluate condition
                condition_expr = condition_config.get('condition', '')
                if self._evaluate_condition(condition_expr):
                    description = condition_config.get('description', condition_name)
                    logger.info(f"Skipping DAG {dag_id}: {description}")
                    
                    # Send alert if configured
                    if self._should_alert_skip(dag_id, condition_name):
                        send_prioritized_alert(
                            f"DAG {dag_id} skipped: {description}",
                            severity="info",
                            context={'dag_id': dag_id, 'condition': condition_name}
                        )
                    
                    return f'{dag_id}_skip'
            
            logger.info(f"All conditions passed, proceeding with DAG: {dag_id}")
            return f'{dag_id}_proceed'
            
        except Exception as e:
            logger.error(f"Error evaluating skip conditions for {dag_id}: {e}")
            return f'{dag_id}_proceed'  # Default to proceed on error
    
    def _evaluate_condition(self, condition: str) -> bool:
        """
        Safely evaluate a condition expression.
        
        Args:
            condition: String expression to evaluate
            
        Returns:
            Boolean result of condition
        """
        try:
            # Create safe execution context with utility functions
            safe_context = {
                'is_market_open': is_market_open,
                'get_market_session': get_market_session,
                'get_environment': get_environment,
                'check_data_freshness': self._check_data_freshness,
                'get_market_data_count': self._get_market_data_count,
                'get_market_volatility': self._get_market_volatility,
                'get_consensus_confidence': self._get_consensus_confidence,
                'check_risk_limits_exceeded': self._check_risk_limits_exceeded,
                # Add safe built-ins
                'True': True,
                'False': False,
                'None': None
            }
            
            # Safely evaluate expression
            result = eval(condition, {"__builtins__": {}}, safe_context)
            return bool(result)
            
        except Exception as e:
            logger.error(f"Error evaluating condition '{condition}': {e}")
            return False  # Default to not skip on error
    
    def _check_data_freshness(self) -> float:
        """Check freshness of collected data (0.0 = stale, 1.0 = fresh)."""
        try:
            if self.cache_manager:
                data_status = self.cache_manager.get('data_collection_status')
                if data_status:
                    last_update = datetime.fromisoformat(data_status.get('timestamp', ''))
                    age_minutes = (datetime.now() - last_update).total_seconds() / 60
                    return max(0.0, 1.0 - (age_minutes / 60))  # Fresh for 1 hour
            return 0.5  # Default moderate freshness
        except:
            return 0.5
    
    def _get_market_data_count(self) -> int:
        """Get count of available market data points."""
        try:
            if self.cache_manager:
                data_count = self.cache_manager.get('market_data_count')
                return data_count or 0
            return 100  # Default safe count
        except:
            return 100
    
    def _get_market_volatility(self) -> float:
        """Get current market volatility (0.0 = low, 1.0 = high)."""
        try:
            if self.cache_manager:
                volatility = self.cache_manager.get('market_volatility')
                return volatility or 0.2
            return 0.2  # Default moderate volatility
        except:
            return 0.2
    
    def _get_consensus_confidence(self) -> float:
        """Get confidence level of analysis consensus (0.0 = low, 1.0 = high)."""
        try:
            if self.cache_manager:
                confidence = self.cache_manager.get('consensus_confidence')
                return confidence or 0.7
            return 0.7  # Default moderate confidence
        except:
            return 0.7
    
    def _check_risk_limits_exceeded(self) -> bool:
        """Check if trading risk limits are exceeded."""
        try:
            if self.cache_manager:
                risk_status = self.cache_manager.get('risk_status')
                return risk_status and risk_status.get('limits_exceeded', False)
            return False  # Default to safe
        except:
            return False
    
    def _apply_task_timeouts(self, dag: DAG, dag_config: Dict[str, Any]):
        """Apply timeout configurations to tasks."""
        task_deps = dag_config.get('task_dependencies', {})
        
        for task in dag.tasks:
            if task.task_id in task_deps:
                timeout = task_deps[task.task_id].get('timeout')
                if timeout and hasattr(task, 'timeout'):
                    task.timeout = timeout
    
    def _add_monitoring_tasks(self, dag: DAG, dag_id: str, dag_config: Dict[str, Any]):
        """Add monitoring and alerting tasks."""
        monitoring_config = self.config.get('monitoring', {})
        
        if monitoring_config.get('dependency_failures', {}).get('enabled'):
            monitor_task = PythonOperator(
                task_id=f'{dag_id}_dependency_monitor',
                python_callable=self._monitor_dependencies,
                op_kwargs={'dag_id': dag_id, 'config': monitoring_config},
                dag=dag,
                trigger_rule=TriggerRule.ALL_DONE  # Run regardless of upstream success/failure
            )
            
            # Make monitor task depend on all other tasks
            for task in dag.tasks:
                if task.task_id != monitor_task.task_id:
                    task >> monitor_task
    
    def _monitor_dependencies(self, dag_id: str, config: Dict[str, Any], **context):
        """Monitor dependency execution and send alerts."""
        try:
            dag_run = context.get('dag_run')
            if not dag_run:
                return
            
            # Check for task failures
            failed_tasks = []
            for task_instance in dag_run.get_task_instances():
                if task_instance.state == 'failed':
                    failed_tasks.append(task_instance.task_id)
            
            if failed_tasks:
                failure_config = config.get('dependency_failures', {})
                if failure_config.get('enabled'):
                    send_prioritized_alert(
                        f"Dependency failures in DAG {dag_id}: {', '.join(failed_tasks)}",
                        severity="high",
                        channels=failure_config.get('channels', ['log']),
                        context={'dag_id': dag_id, 'failed_tasks': failed_tasks}
                    )
            
            # Update dependency state in cache
            if self.cache_manager:
                self.cache_manager.set(
                    f'{dag_id}_dependency_state',
                    {
                        'timestamp': datetime.now().isoformat(),
                        'failed_tasks': failed_tasks,
                        'total_tasks': len(dag_run.get_task_instances()),
                        'success_rate': (len(dag_run.get_task_instances()) - len(failed_tasks)) / len(dag_run.get_task_instances())
                    },
                    ttl=3600
                )
                
        except Exception as e:
            logger.error(f"Error monitoring dependencies for {dag_id}: {e}")
    
    def _cache_dag_state(self, dag_id: str, dag_config: Dict[str, Any]):
        """Cache DAG dependency state for cross-DAG coordination."""
        if not self.cache_manager or not self.config.get('global', {}).get('cache_dependencies'):
            return
        
        try:
            cache_key = f'dag_dependency_state_{dag_id}'
            state = {
                'dag_id': dag_id,
                'config_timestamp': datetime.now().isoformat(),
                'environment': self.environment,
                'skip_conditions_enabled': len(dag_config.get('skip_conditions', {})),
                'dependencies': dag_config.get('dependencies', []),
                'downstream': dag_config.get('downstream', [])
            }
            
            ttl = self.config.get('global', {}).get('cache_ttl', 3600)
            cache_cross_dag_data(cache_key, state, ttl)
            
        except Exception as e:
            logger.error(f"Error caching DAG state for {dag_id}: {e}")
    
    def _should_alert_skip(self, dag_id: str, condition_name: str) -> bool:
        """Determine if skip condition should trigger an alert."""
        try:
            monitoring_config = self.config.get('monitoring', {})
            skip_monitoring = monitoring_config.get('skip_rate_monitoring', {})
            
            if not skip_monitoring.get('enabled'):
                return False
            
            # Check skip rate over time window
            if self.cache_manager:
                skip_key = f'{dag_id}_skip_history'
                skip_history = self.cache_manager.get(skip_key, [])
                
                # Add current skip
                skip_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'condition': condition_name
                })
                
                # Keep only recent skips (24h window)
                cutoff = datetime.now() - timedelta(hours=24)
                skip_history = [s for s in skip_history 
                              if datetime.fromisoformat(s['timestamp']) > cutoff]
                
                # Update cache
                self.cache_manager.set(skip_key, skip_history, ttl=86400)  # 24 hours
                
                # Calculate skip rate
                threshold = skip_monitoring.get('threshold', 0.5)
                if len(skip_history) > 10:  # Only alert if we have enough data
                    skip_rate = len(skip_history) / 10  # Approximate rate
                    return skip_rate > threshold
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking skip alert for {dag_id}: {e}")
            return False
    
    def get_cross_dag_dependencies(self, dag_id: str) -> Dict[str, Any]:
        """Get cross-DAG dependencies for a specific DAG."""
        try:
            cross_deps = self.config.get('cross_dag_dependencies', {})
            result = {}
            
            for dep_name, dep_config in cross_deps.items():
                if (dep_config.get('target_dag') == dag_id or 
                    dag_id in dep_config.get('applies_to', [])):
                    result[dep_name] = dep_config
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting cross-DAG dependencies for {dag_id}: {e}")
            return {}
    
    def validate_dependencies(self, dag_id: str) -> Dict[str, Any]:
        """
        Validate DAG dependencies and return status.
        
        Returns:
            Validation result with status and details
        """
        try:
            validation_result = {
                'dag_id': dag_id,
                'timestamp': datetime.now().isoformat(),
                'valid': True,
                'errors': [],
                'warnings': [],
                'dependencies_checked': 0
            }
            
            if dag_id not in self.config.get('dags', {}):
                validation_result['errors'].append(f"No configuration found for DAG: {dag_id}")
                validation_result['valid'] = False
                return validation_result
            
            dag_config = self.config['dags'][dag_id]
            
            # Check upstream dependencies
            for upstream_dag in dag_config.get('dependencies', []):
                validation_result['dependencies_checked'] += 1
                
                if self.cache_manager:
                    upstream_state = self.cache_manager.get(f'dag_dependency_state_{upstream_dag}')
                    if not upstream_state:
                        validation_result['warnings'].append(f"Upstream DAG {upstream_dag} state not cached")
            
            # Validate skip conditions
            skip_conditions = dag_config.get('skip_conditions', {})
            for condition_name, condition_config in skip_conditions.items():
                condition_expr = condition_config.get('condition', '')
                if condition_expr:
                    try:
                        # Test condition syntax
                        self._evaluate_condition(condition_expr)
                    except Exception as e:
                        validation_result['errors'].append(f"Invalid condition '{condition_name}': {e}")
                        validation_result['valid'] = False
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating dependencies for {dag_id}: {e}")
            return {
                'dag_id': dag_id,
                'valid': False,
                'errors': [str(e)],
                'warnings': [],
                'dependencies_checked': 0
            }


# Global instance for convenience
_dependency_manager = None

def get_dependency_manager() -> DependencyManager:
    """Get global dependency manager instance."""
    global _dependency_manager
    if _dependency_manager is None:
        _dependency_manager = DependencyManager()
    return _dependency_manager


# Convenience functions for DAG integration
def setup_dag_dependencies(dag: DAG, dag_id: str) -> DAG:
    """Setup dependencies for a DAG using the global dependency manager."""
    return get_dependency_manager().setup_dependencies(dag, dag_id)


def validate_dag_dependencies(dag_id: str) -> Dict[str, Any]:
    """Validate dependencies for a DAG using the global dependency manager."""
    return get_dependency_manager().validate_dependencies(dag_id)