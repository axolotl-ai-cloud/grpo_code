import atexit
import signal
import sys
from concurrent.futures import (
    BrokenExecutor,
    FIRST_EXCEPTION,
    ProcessPoolExecutor,
    wait,
)
from pathlib import Path
from typing import Callable

import grpo_code

from grpo_code.wasm import does_code_run, PythonWasmEnvironment

_executor = None


def worker_init(wasm_path: str, fuel: int):
    import grpo_code.wasm as wasm
    wasm.worker_env = PythonWasmEnvironment(wasm_path, fuel)

def cleanup_executor():
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=False)
        _executor = None


def cleanup_and_exit():
    cleanup_executor()
    sys.exit(0)


def get_multiprocessing_executor(max_processes: int, wasm_path: str, fuel: int):
    global _executor
    if _executor is None:
        _executor = ProcessPoolExecutor(
            max_workers=max_processes,
            initializer=worker_init,
            initargs=(wasm_path, fuel),
        )
        atexit.register(cleanup_executor)
        signal.signal(signal.SIGINT, cleanup_and_exit)
        signal.signal(signal.SIGTERM, cleanup_and_exit)
    return _executor


def run_tasks_with_multiprocessing_executor(
    executor: callable, tasks: list[str], timeout: int
):
    futures_to_index = {
        executor.submit(does_code_run, task): i for i, task in enumerate(tasks)
    }
    futures = list(futures_to_index)  # list of futures for wait()
    results = [0.0] * len(tasks)

    while futures:
        # Wait for either first exception or timeout
        done, futures = wait(futures, timeout=timeout, return_when=FIRST_EXCEPTION)

        if futures and len(done) == 0:
            print(
                f"WARNING: Tasks timed out after {timeout} seconds, recreating process pool..."
            )
            cleanup_executor()
            break

        # Process completed futures
        for future in done:
            task_index = futures_to_index[future]
            results[task_index] = future.result()

    return results


def test_multiprocessing_executor():
    """Test function to verify multiprocessing executor works correctly"""
    from grpo_code.executor import execute_tasks

    # Define test tasks - some normal, some that should timeout
    test_tasks = [
        "print('This is a quick task')",  # Should succeed
        "for i in range(5): print(i)",  # Should succeed
        "import time; time.sleep(10)",  # Should timeout with default settings
        "print('Another quick task')",  # Should succeed
        "1/0",  # Should fail with ZeroDivisionError
    ]

    # Set up parameters
    wasm_path = Path(grpo_code.__file__).parent.parent / "wasm" / "python-3.12.0.wasm"
    fuel = 500_000_000
    max_processes = 2
    timeout = 2

    # Run the tasks using the executor API
    results = execute_tasks(test_tasks, max_processes, wasm_path, fuel, timeout)

    # Print results
    print("Test Results:")
    for i, (task, result) in enumerate(zip(test_tasks, results)):
        task_preview = task[:30] + "..." if len(task) > 30 else task
        print(
            f"Task {i}: {task_preview} -> {'Success' if result > 0 else 'Failed'} ({result})"
        )

    return results


if __name__ == "__main__":
    # Run the test when the file is executed directly
    test_results = test_multiprocessing_executor()
    print(
        f"Test completed with {sum(1 for r in test_results if r > 0)}/{len(test_results)} successful tasks"
    )
