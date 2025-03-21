from multiprocessing import Pool
import os
import re
from wasmtime import Config, Engine, Linker, Module, Store, WasiConfig
import math
import atexit
from contextlib import contextmanager

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Define extract_xml_answer here to avoid circular import
def extract_xml_answer(text):
    """Extract the answer from the XML tags."""
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


class PythonWasmEnvironment:
    """A reusable WASM environment for running Python code."""

    def __init__(self, wasm_path="../wasm/python-3.12.0.wasm", fuel=1_000_000_000):
        """Initialize the WASM environment."""
        self.wasm_path = wasm_path
        self.fuel = fuel

        # Set up the engine and linker
        engine_cfg = Config()
        engine_cfg.consume_fuel = True
        engine_cfg.cache = True

        self.engine = Engine(engine_cfg)
        self.linker = Linker(self.engine)
        self.linker.define_wasi()

        # Load the Python module
        self.python_module = Module.from_file(self.engine, self.wasm_path)

    def run_code(self, code):
        """Run Python code in the WASM environment."""
        config = WasiConfig()
        config.argv = ("python", "-c", code)

        store = Store(self.engine)
        store.set_fuel(self.fuel)
        store.set_wasi(config)

        instance = self.linker.instantiate(store, self.python_module)
        start = instance.exports(store)["_start"]

        start(store)


# Maximum number of processes to use
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
MAX_PROCESSES = min(64, os.cpu_count()) // WORLD_SIZE

# Global process pool
_process_pool = None


def _cleanup_global_pool():
    """Clean up the global process pool on program exit."""
    global _process_pool
    if _process_pool is not None:
        print("Cleaning up global process pool...")
        _process_pool.close()
        _process_pool.join()
        _process_pool = None


# Register cleanup function
atexit.register(_cleanup_global_pool)


def worker_init():
    """Initialize a worker-specific WASM environment."""
    global worker_env
    worker_env = PythonWasmEnvironment()


@contextmanager
def get_process_pool():
    """Returns a reusable process pool with properly initialized workers."""
    global _process_pool

    if _process_pool is None:
        _process_pool = Pool(processes=MAX_PROCESSES, initializer=worker_init)

    try:
        yield _process_pool
    except Exception as e:
        # If there's an exception, recreate the pool on next use
        if _process_pool:
            _process_pool.terminate()
            _process_pool = None
        raise e


# Worker functions that use the worker-local environment
def runs_without_error(code: str) -> bool:
    """Execute code in the worker's WASM environment and check if it runs without errors."""
    global worker_env
    try:
        worker_env.run_code(code)
        return 1.0
    except Exception:
        return 0.0


def check_format(text):
    """Check if text follows the expected format pattern."""
    return 0.25 if re.match(r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>.*", text, re.S) else 0.0


def does_compile_syntax_check(predicted_answer: str) -> float:
    """Check if the code has valid syntax."""
    try:
        compile(predicted_answer, "<string>", "exec")
        return 0.25
    except Exception:
        return -0.1


# Optimized reward functions with persistent pool
def multiprocessing_code_execution_reward_func(completions: list[list[dict]], **kwargs) -> list[float]:
    """Reward function for executing the code."""
    model_answers = [extract_xml_answer(completion[0]["content"]) for completion in completions]

    # Calculate optimal chunk size based on workload
    optimal_workers = min(len(model_answers), MAX_PROCESSES)
    chunk_size = max(1, len(model_answers) // optimal_workers)

    with get_process_pool() as pool:
        results = pool.map(runs_without_error, model_answers, chunksize=chunk_size)

    results = [0.5 if result == 1.0 else -0.25 for result in results]
    return results


def multiprocessing_answer_execution_reward_func(
    completions: list[list[dict]], answers: list[list[str]], **kwargs
) -> list[float]:
    """Reward function for executing the code with test cases."""
    model_answers = [extract_xml_answer(completion[0]["content"]) for completion in completions]

    tasks = []
    for i in range(len(model_answers)):
        for test_case in answers[i]:
            tasks.append(model_answers[i] + "\n" + test_case)

    # Calculate optimal chunk size for flattened tasks
    optimal_workers = min(len(tasks), MAX_PROCESSES)
    chunk_size = max(1, len(tasks) // optimal_workers)

    with get_process_pool() as pool:
        results = pool.map(runs_without_error, tasks, chunksize=chunk_size)

    # Reconstruct results by completion
    rewards = [0.0] * len(completions)
    result_idx = 0
    for i in range(len(completions)):
        test_case_count = len(answers[i])
        completion_results = results[result_idx : result_idx + test_case_count]
        result_idx += test_case_count
        accuracy = sum(completion_results) / test_case_count
        rewards[i] = math.pow(accuracy, 2) * 2  # Square and multiply by 2 as in other functions

    return rewards


def multiprocessing_soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that loosely checks if the completion has a specific format."""
    responses = [completion[0]["content"] for completion in completions]

    # Calculate optimal chunk size
    optimal_workers = min(len(responses), MAX_PROCESSES)
    chunk_size = max(1, len(responses) // optimal_workers)

    with get_process_pool() as pool:
        results = pool.map(check_format, responses, chunksize=chunk_size)

    return results


def multiprocessing_syntax_check_reward_func(completions: list[list[dict]], **kwargs) -> list[float]:
    """Reward function for checking if the code has valid syntax."""
    model_answers = [extract_xml_answer(completion[0]["content"]) for completion in completions]

    # Calculate optimal chunk size
    optimal_workers = min(len(model_answers), MAX_PROCESSES)
    chunk_size = max(1, len(model_answers) // optimal_workers)

    with get_process_pool() as pool:
        compile_rewards = pool.map(does_compile_syntax_check, model_answers, chunksize=chunk_size)

    return compile_rewards


if __name__ == "__main__":
    try:
        import time

        # Print CPU information
        print(f"Number of CPU cores available: {os.cpu_count()}")

        example_completion = [
            [
                {
                    "role": "user",
                    "content": """
        <reasoning>
        ...
        </reasoning>
        <answer>
        def foo(a):
            return a + 1
        </answer>""",
                }
            ]
        ]
        num_copies = 16
        large_completions = example_completion * num_copies
        large_answers = [["assert foo(1) == 2", "assert foo(2) == 3"]] * 16

        # Number of runs for averaging
        num_runs = 100
        
        print("-" * 100)
        print(f"Testing with {num_copies} copies of the example completion")
        print(f"Running each test {num_runs} times and reporting averages")
        print("-" * 100)

        # Run each test multiple times and track total execution times
        mp_execution_total_time = 0
        mp_answer_execution_total_time = 0
        mp_soft_format_total_time = 0
        mp_syntax_check_total_time = 0
        total_run_time = 0

        for i in range(num_runs):
            run_start_time = time.time()
            
            if i % 10 == 0:
                print(f"Run {i+1}/{num_runs}...")
                
            # Test code execution reward function
            start_time = time.time()
            mp_execution_results = multiprocessing_code_execution_reward_func(large_completions)
            mp_execution_total_time += time.time() - start_time

            # Test multiprocessing_answer_execution_reward_func
            start_time = time.time()
            mp_answer_execution_results = multiprocessing_answer_execution_reward_func(large_completions, large_answers)
            mp_answer_execution_total_time += time.time() - start_time

            # Test multiprocessing_soft_format_reward_func
            start_time = time.time()
            mp_soft_format_results = multiprocessing_soft_format_reward_func(large_completions)
            mp_soft_format_total_time += time.time() - start_time

            # Test multiprocessing_syntax_check_reward_func
            start_time = time.time()
            mp_syntax_check_results = multiprocessing_syntax_check_reward_func(large_completions)
            mp_syntax_check_total_time += time.time() - start_time
            
            # Calculate total time for this run
            run_time = time.time() - run_start_time
            total_run_time += run_time
            
            if i % 10 == 0:
                print(f"  Run {i+1} completed in {run_time:.4f} seconds")

        # Calculate and print average times
        print("-" * 100)
        print("AVERAGE EXECUTION TIMES OVER", num_runs, "RUNS:")
        print("-" * 100)
        print(f"multiprocessing_code_execution_reward_func avg time: {mp_execution_total_time/num_runs:.4f} seconds")
        print(f"multiprocessing_answer_execution_reward_func avg time: {mp_answer_execution_total_time/num_runs:.4f} seconds")
        print(f"multiprocessing_soft_format_reward_func avg time: {mp_soft_format_total_time/num_runs:.4f} seconds")
        print(f"multiprocessing_syntax_check_reward_func avg time: {mp_syntax_check_total_time/num_runs:.4f} seconds")
        print(f"Average total time per run: {total_run_time/num_runs:.4f} seconds")
        print(f"Total benchmark time: {total_run_time:.4f} seconds")
        print("-" * 100)

    finally:
        # Explicit cleanup, though atexit would handle this too
        if "_process_pool" in globals() and _process_pool is not None:
            print("Explicit cleanup of process pool")
            _process_pool.close()
            _process_pool.join()
