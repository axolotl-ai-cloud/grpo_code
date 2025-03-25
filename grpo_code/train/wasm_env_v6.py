import atexit
import math
import multiprocessing
import os
import re
import signal
import sys
import time
from concurrent.futures import BrokenExecutor, ProcessPoolExecutor, TimeoutError
from pathlib import Path

from wasmtime import Config, Engine, Linker, Module, Store, WasiConfig

import grpo_code

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Compile regex patterns once for better performance
ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
FORMAT_PATTERN = re.compile(
    r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>.*", re.DOTALL
)


def extract_xml_answer(text):
    """Extract the answer from the XML tags using compiled regex."""
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return match.group(1).strip() if match else ""


WASM_PATH = os.environ.get(
    "WASM_PATH", Path(grpo_code.__file__).parent.parent / "wasm" / "python-3.12.0.wasm"
)

if not os.path.exists(WASM_PATH):
    raise FileNotFoundError(f"WASM file not found at {WASM_PATH}")

# Maximum number of processes to use
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
MAX_PROCESSES = int(os.environ.get("MAX_PROCESSES", 32)) // WORLD_SIZE

# Task timeout - maximum time (in seconds) to wait for a worker to complete a task
TASK_TIMEOUT = int(os.environ.get("TASK_TIMEOUT", 1))

_executor = None

from grpo_code.wasm import PythonWasmEnvironment


# Worker function with initialization
def worker_init():
    """Initialize worker-specific environment."""
    global worker_env
    worker_env = PythonWasmEnvironment(wasm_path=WASM_PATH, fuel=1_000_000_000)


def runs_without_error(code: str) -> float:
    """Execute code in the worker's WASM environment and check if it runs without errors."""
    global worker_env
    try:
        worker_env.run_code(code)
        return 1.0
    except Exception as e:
        print(f"Error running code: {e}")
        return 0.0


def cleanup_executor():
    """Cleanup the global executor."""
    global _executor
    if _executor is not None:
        print("Shutting down global executor...")
        _executor.shutdown(wait=False)
        _executor = None


def cleanup_and_exit(signum, frame):
    """Clean up and exit on signals."""
    print(f"Received signal {signum}, cleaning up...")
    cleanup_executor()
    sys.exit(0)


def check_format(text):
    """Check if text follows the expected format pattern."""
    return (
        0.25
        if re.match(r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>.*", text, re.S)
        else 0.0
    )


def get_executor():
    """Get or create the global ProcessPoolExecutor."""
    global _executor
    if _executor is None:
        _executor = ProcessPoolExecutor(
            max_workers=MAX_PROCESSES, initializer=worker_init
        )
        # Register cleanup on exit
        atexit.register(cleanup_executor)
        # Register signal handlers
        signal.signal(signal.SIGINT, cleanup_and_exit)
        signal.signal(signal.SIGTERM, cleanup_and_exit)
    return _executor


def run_with_executor(func, tasks, timeout=TASK_TIMEOUT):
    """Run tasks using the global ProcessPoolExecutor with timeout and error handling."""
    global _executor

    executor = get_executor()
    futures = [executor.submit(func, task) for task in tasks]

    results = []
    for future in futures:
        try:
            result = future.result(timeout=timeout)
            results.append(result)
        except TimeoutError:
            print(
                f"WARNING: Task timed out after {timeout} seconds, recreating process pool..."
            )
            cleanup_executor()  # Clean up the old executor
            results.append(0.0)
        except BrokenExecutor:
            print(f"WARNING: Process pool was broken, recreating process pool...")
            cleanup_executor()
            results.append(0.0)
        except Exception as e:
            print(f"WARNING: Task failed with error: {e}")
            results.append(0.0)

    return results


def multiprocessing_code_execution_reward_func(
    completions: list[list[dict]], **kwargs
) -> list[float]:
    """Reward function for executing the code."""
    model_answers = [
        extract_xml_answer(completion[0]["content"]) for completion in completions
    ]
    results = run_with_executor(runs_without_error, model_answers)
    return [0.5 if result == 1.0 else -0.25 for result in results]


def multiprocessing_answer_execution_reward_func(
    completions: list[list[dict]], answers: list[list[str]], **kwargs
) -> list[float]:
    """Reward function for executing the code with test cases."""
    model_answers = [
        extract_xml_answer(completion[0]["content"]) for completion in completions
    ]

    # Prepare all test cases
    tasks = []
    test_indices = []
    for i, (code, tests) in enumerate(zip(model_answers, answers)):
        for test in tests:
            tasks.append(code + "\n" + test)
            test_indices.append(i)

    # Run all tests
    results = run_with_executor(runs_without_error, tasks)

    # Group results by completion
    completion_results = {}
    for idx, result in zip(test_indices, results):
        if idx not in completion_results:
            completion_results[idx] = []
        completion_results[idx].append(result)

    # Calculate rewards
    rewards = []
    for i in range(len(completions)):
        if i in completion_results:
            test_results = completion_results[i]
            accuracy = sum(test_results) / len(test_results)
            reward = math.pow(accuracy, 3) * 2
        else:
            reward = 0.0
        rewards.append(reward)

    return rewards


def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Non-multiprocessing version of format checking."""
    responses = [completion[0]["content"] for completion in completions]
    return [check_format(r) for r in responses]


if __name__ == "__main__":
    try:
        # Print CPU information
        print(f"Number of CPU cores available: {os.cpu_count()}")
        print(f"Using maximum of {MAX_PROCESSES} worker processes")
        print(f"Task timeout set to {TASK_TIMEOUT} seconds")

        example_completion = [
            [
                {
                    "role": "user",
                    "content": """
        <reasoning>
        We need to implement a function that increments its input by 1.
        </reasoning>
        <answer>
        def foo(a):
            return a + 1
        </answer>""",
                }
            ]
        ]
        num_copies = 32
        large_completions = example_completion * num_copies
        large_answers = [["assert foo(1) == 2", "assert foo(2) == 3"]] * num_copies

        # Number of runs for averaging
        num_runs = 100

        print("-" * 100)
        print(f"Testing with {num_copies} copies of the example completion")
        print(f"Running each test {num_runs} times and reporting averages")
        print(
            f"Using timeout-aware ProcessPoolExecutor with {MAX_PROCESSES} max workers"
        )
        print("-" * 100)

        # Run each test multiple times and track total execution times
        mp_execution_total_time = 0
        mp_answer_execution_total_time = 0
        mp_soft_format_total_time = 0
        soft_format_total_time = 0
        total_run_time = 0

        for i in range(num_runs):
            run_start_time = time.time()

            # print(f"Run {i+1}/{num_runs}...")

            # Test code execution reward function
            start_time = time.time()
            mp_execution_results = multiprocessing_code_execution_reward_func(
                large_completions
            )
            mp_execution_time = time.time() - start_time
            mp_execution_total_time += mp_execution_time
            # print(f"  Code execution: {mp_execution_time:.4f}s")

            # Test multiprocessing_answer_execution_reward_func
            start_time = time.time()
            mp_answer_execution_results = multiprocessing_answer_execution_reward_func(
                large_completions, large_answers
            )
            mp_answer_time = time.time() - start_time
            mp_answer_execution_total_time += mp_answer_time
            # print(f"  Answer execution: {mp_answer_time:.4f}s")

            # Test multiprocessing_soft_format_reward_func
            start_time = time.time()
            mp_soft_format_results = multiprocessing_soft_format_reward_func(
                large_completions
            )
            mp_soft_format_time = time.time() - start_time
            mp_soft_format_total_time += mp_soft_format_time
            # print(f"  Format check: {mp_soft_format_time:.4f}s")

            # Test soft_format_reward_func
            start_time = time.time()
            soft_format_results = soft_format_reward_func(large_completions)
            soft_format_time = time.time() - start_time
            soft_format_total_time += soft_format_time
            # print(f"  Non-MP format check: {soft_format_time:.4f}s")

            # Calculate total time for this run
            run_time = time.time() - run_start_time
            total_run_time += run_time

            # print(f"  Run {i+1} completed in {run_time:.4f} seconds")

        # Calculate and print average times
        print("-" * 100)
        print("AVERAGE EXECUTION TIMES OVER", num_runs, "RUNS:")
        print("-" * 100)
        print(
            f"multiprocessing_code_execution_reward_func avg time: {mp_execution_total_time/num_runs:.4f} seconds"
        )
        print(
            f"multiprocessing_answer_execution_reward_func avg time: {mp_answer_execution_total_time/num_runs:.4f} seconds"
        )
        print(
            f"multiprocessing_soft_format_reward_func avg time: {mp_soft_format_total_time/num_runs:.4f} seconds"
        )
        print(
            f"soft_format_reward_func avg time: {soft_format_total_time/num_runs:.4f} seconds"
        )
        print(f"Average total time per run: {total_run_time/num_runs:.4f} seconds")
        print(f"Total benchmark time: {total_run_time:.4f} seconds")
        print("-" * 100)

        # Manual test with the 60% accuracy case
        print("\n" + "=" * 80)
        print("MANUAL TEST: multiprocessing_answer_execution_reward_func")
        print("=" * 80)

        # Create a simple test case - function that only works with positive numbers
        test_completion = [
            [
                {
                    "role": "user",
                    "content": """
        <reasoning>
        Let's create a function that adds two numbers, but it only handles positive numbers correctly.
        </reasoning>
        <answer>
        def add(a, b):
            if a < 0 or b < 0:
                return 0  # Incorrect for negative numbers
            return a + b  # Correct for positive numbers
        </answer>""",
                }
            ]
        ]

        # Test cases - 3 should pass, 2 should fail (60% accuracy)
        test_answers = [
            [
                "assert add(1, 2) == 3",  # Pass
                "assert add(5, 7) == 12",  # Pass
                "assert add(10, 20) == 30",  # Pass
                "assert add(-1, 5) == 4",  # Fail - will return 0
                "assert add(-5, -10) == -15",  # Fail - will return 0
            ]
        ]

        print("Input code:")
        print(extract_xml_answer(test_completion[0][0]["content"]))
        print("\nTest cases:")
        for tc in test_answers[0]:
            print(f"- {tc}")

        # Run the function and measure time
        print("\nExecuting test...")
        start_time = time.time()
        results = multiprocessing_answer_execution_reward_func(
            test_completion, test_answers
        )
        elapsed = time.time() - start_time

        print(f"\nResults: {results}")
        print(f"Execution time: {elapsed:.4f} seconds")

        # Calculate the expected reward (accuracy^3 * 2)
        expected_reward = math.pow(3 / 5, 3) * 2
        print(f"Expected reward with 3/5 accuracy: {expected_reward:.4f}")

        # Add assertion for partial success implementation
        # Allow some floating point tolerance
        assert (
            abs(results[0] - expected_reward) < 0.1
        ), f"Expected ~{expected_reward:.4f} for 60% accuracy, got {results[0]}"
        print(
            "✅ Assertion passed: 60% correct implementation received expected reward"
        )

        print("\nSimulation test - handling timeouts")
        print("=" * 80)

    finally:
        # Ensure executor is cleaned up
        cleanup_executor()
        print("Test completed - all pools should be properly shut down")
