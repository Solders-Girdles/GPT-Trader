"""
Parallel Execution System for Bot V2 Orchestration

Provides concurrent execution of independent slices with dependency management,
thread-safe result aggregation, and graceful error handling.
"""

import concurrent.futures
import logging
import time
from typing import Dict, Any, List, Callable, Optional, Set, Union
from dataclasses import dataclass, field
from threading import Lock
import traceback


@dataclass
class Task:
    """Represents a parallelizable task with dependencies"""
    name: str
    func: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    dependencies: Set[str] = field(default_factory=set)
    timeout: float = 30.0
    retry_count: int = 0
    max_retries: int = 2
    
    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}
        if self.dependencies is None:
            self.dependencies = set()


class ParallelExecutor:
    """
    Thread-safe parallel execution engine with dependency management
    
    Features:
    - Respects task dependencies 
    - Thread-safe result aggregation
    - Graceful error handling with retries
    - Batch processing capabilities
    - Resource management
    """
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.logger = logging.getLogger("parallel_executor")
        self._results_lock = Lock()
        self._completed_lock = Lock()
        
    def execute_parallel(self, tasks: List[Task]) -> Dict[str, Any]:
        """
        Execute tasks in parallel while respecting dependencies
        
        Args:
            tasks: List of Task objects to execute
            
        Returns:
            Dictionary mapping task names to results
        """
        results = {}
        completed = set()
        failed = set()
        task_map = {t.name: t for t in tasks}
        
        self.logger.info(f"Starting parallel execution of {len(tasks)} tasks")
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            round_num = 0
            
            while len(completed) + len(failed) < len(tasks):
                round_num += 1
                
                # Find tasks ready to run (dependencies satisfied)
                ready_tasks = [
                    t for t in tasks 
                    if (t.name not in completed and 
                        t.name not in failed and
                        t.dependencies.issubset(completed))
                ]
                
                if not ready_tasks:
                    # Check for circular dependencies or unresolvable states
                    remaining = [t.name for t in tasks if t.name not in completed and t.name not in failed]
                    self.logger.warning(f"No ready tasks found. Remaining: {remaining}")
                    break
                
                self.logger.info(f"Round {round_num}: Executing {len(ready_tasks)} tasks")
                
                # Submit ready tasks for execution
                future_to_task = {}
                for task in ready_tasks:
                    future = executor.submit(self._execute_single_task, task)
                    future_to_task[future] = task
                
                # Wait for completion and collect results
                for future in concurrent.futures.as_completed(future_to_task):
                    task = future_to_task[future]
                    
                    try:
                        result = future.result(timeout=1.0)  # Quick timeout since work is done
                        
                        with self._results_lock:
                            results[task.name] = result
                        
                        with self._completed_lock:
                            completed.add(task.name)
                            
                        self.logger.info(f"Task '{task.name}' completed successfully")
                        
                    except Exception as e:
                        self.logger.error(f"Task '{task.name}' failed: {str(e)}")
                        
                        with self._results_lock:
                            results[task.name] = {
                                'error': str(e),
                                'traceback': traceback.format_exc(),
                                'status': 'failed'
                            }
                        
                        with self._completed_lock:
                            failed.add(task.name)
        
        execution_time = time.time() - start_time
        success_count = len(completed)
        failure_count = len(failed)
        
        self.logger.info(
            f"Parallel execution completed in {execution_time:.2f}s. "
            f"Success: {success_count}, Failed: {failure_count}"
        )
        
        return results
    
    def _execute_single_task(self, task: Task) -> Any:
        """Execute a single task with retry logic"""
        last_exception = None
        
        for attempt in range(task.max_retries + 1):
            try:
                self.logger.debug(f"Executing task '{task.name}' (attempt {attempt + 1})")
                
                # Execute with timeout
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as single_executor:
                    future = single_executor.submit(task.func, *task.args, **task.kwargs)
                    result = future.result(timeout=task.timeout)
                
                return {
                    'result': result,
                    'status': 'success',
                    'attempts': attempt + 1
                }
                
            except concurrent.futures.TimeoutError:
                last_exception = f"Task timed out after {task.timeout}s"
                self.logger.warning(f"Task '{task.name}' timed out (attempt {attempt + 1})")
                
            except Exception as e:
                last_exception = str(e)
                self.logger.warning(f"Task '{task.name}' failed (attempt {attempt + 1}): {e}")
                
                if attempt < task.max_retries:
                    time.sleep(0.1 * (attempt + 1))  # Exponential backoff
        
        # All attempts failed
        raise Exception(f"Task failed after {task.max_retries + 1} attempts. Last error: {last_exception}")
    
    def execute_batch(self, items: List[Any], process_func: Callable, 
                     batch_size: int = 10, timeout: float = 30.0) -> List[Any]:
        """
        Execute function on items in parallel batches
        
        Args:
            items: List of items to process
            process_func: Function to apply to each item
            batch_size: Number of items per batch
            timeout: Timeout per item
            
        Returns:
            List of results (None for failed items)
        """
        results = [None] * len(items)  # Pre-allocate to maintain order
        
        self.logger.info(f"Starting batch execution of {len(items)} items in batches of {batch_size}")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all items with their indices
            future_to_index = {}
            
            for i, item in enumerate(items):
                future = executor.submit(process_func, item)
                future_to_index[future] = i
            
            # Collect results maintaining order
            completed_count = 0
            failed_count = 0
            
            for future in concurrent.futures.as_completed(future_to_index):
                index = future_to_index[future]
                
                try:
                    result = future.result(timeout=1.0)  # Quick timeout since work is done
                    results[index] = result
                    completed_count += 1
                    
                except Exception as e:
                    self.logger.error(f"Batch item {index} failed: {e}")
                    results[index] = None
                    failed_count += 1
        
        self.logger.info(f"Batch execution completed. Success: {completed_count}, Failed: {failed_count}")
        return results
    
    def execute_map_reduce(self, items: List[Any], map_func: Callable, 
                          reduce_func: Callable, batch_size: int = 10) -> Any:
        """
        Execute map-reduce pattern in parallel
        
        Args:
            items: Items to process
            map_func: Function to apply to each item
            reduce_func: Function to combine results
            batch_size: Batch size for mapping
            
        Returns:
            Final reduced result
        """
        self.logger.info(f"Starting map-reduce on {len(items)} items")
        
        # Map phase
        mapped_results = self.execute_batch(items, map_func, batch_size)
        
        # Filter out None results
        valid_results = [r for r in mapped_results if r is not None]
        
        # Reduce phase
        if not valid_results:
            return None
            
        try:
            final_result = valid_results[0]
            for result in valid_results[1:]:
                final_result = reduce_func(final_result, result)
            
            self.logger.info("Map-reduce completed successfully")
            return final_result
            
        except Exception as e:
            self.logger.error(f"Reduce phase failed: {e}")
            raise
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get current executor statistics"""
        return {
            'max_workers': self.max_workers,
            'logger_name': self.logger.name,
            'logger_level': self.logger.level
        }