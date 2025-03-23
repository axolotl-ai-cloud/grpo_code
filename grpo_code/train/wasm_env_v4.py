from concurrent.futures import ProcessPoolExecutor
import os
import re
from wasmtime import Config, Engine, Linker, Module, Store, WasiConfig
import math
import atexit
import signal
import time
from loky import get_reusable_executor

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Define extract_xml_answer here to avoid circular import
def extract_xml_answer(text):
    """Extract the answer from the XML tags."""
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

WASM_PATH = "/workspace/data/grpo_code/grpo_code/wasm/python-3.12.0.wasm"

if not os.path.exists(WASM_PATH):
    raise FileNotFoundError(f"WASM file not found at {WASM_PATH}")


class PythonWasmEnvironment:
    """A reusable WASM environment for running Python code."""

    def __init__(self, wasm_path=WASM_PATH, fuel=1_000_000_000):
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
        """Run Python code in the WASM environment with timeout."""
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
MAX_PROCESSES = int(os.environ.get("MAX_PROCESSES", 32)) // WORLD_SIZE

try:
    PythonWasmEnvironment()
except Exception as e:
    print("Error initializing PythonWasmEnvironment")
    print(e)
    raise e

# Global process executor - persist between calls for efficiency
_process_executor = None

# def _cleanup_global_executor():
#     """Clean up the global process executor on program exit."""
#     global _process_executor
#     if _process_executor is not None:
#         print("Cleaning up global process executor...")
#         _process_executor.shutdown()
#         _process_executor = None

# def sigHandler(signum, frame):
#     """Handle SIGTERM signal."""
#     print("SIGTERM signal received")
#     global _process_executor
#     if _process_executor is not None:
#         _process_executor.shutdown(wait=False)  # Non-blocking shutdown
#         _process_executor = None
#     os._exit(0)

# signal.signal(signal.SIGTERM, sigHandler)
# atexit.register(_cleanup_global_executor)

# # Worker-specific environment
def worker_init():
    """Initialize worker-specific environment."""
    global worker_env
    worker_env = PythonWasmEnvironment()


def runs_without_error(code: str) -> float:
    """Execute code in the worker's WASM environment and check if it runs without errors."""
    global worker_env
    try:
        print(code)
        worker_env.run_code(code)
        return 1.0
    except Exception as e:
        print(e)
        return 0.0

def check_format(text):
    """Check if text follows the expected format pattern."""
    return 0.25 if re.match(r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>.*", text, re.S) else 0.0

def get_process_executor(max_workers=None):
    """Returns a ProcessPoolExecutor with the specified number of workers.
    
    Creates a new executor if necessary, or returns the existing one.
    """

    _process_executor = get_reusable_executor(max_workers=max_workers, timeout=2, initializer=worker_init)
    
    return _process_executor

# Reward functions using ProcessPoolExecutor
def multiprocessing_code_execution_reward_func(completions: list[list[dict]], **kwargs) -> list[float]:
    """Reward function for executing the code."""
    model_answers = [extract_xml_answer(completion[0]["content"]) for completion in completions]
    
    optimal_workers = min(len(model_answers), MAX_PROCESSES)
    chunk_size = max(1, len(model_answers) // optimal_workers)

    # Use shared executor for better performance
    executor = get_process_executor(optimal_workers)
    results = list(executor.map(runs_without_error, model_answers, chunksize=chunk_size))
    
    results = [0.5 if result == 1.0 else -0.25 for result in results]
    return results

def multiprocessing_answer_execution_reward_func(
    completions: list[list[dict]], answers: list[list[str]], **kwargs
) -> list[float]:
    """Reward function for executing the code with test cases."""
    model_answers = [extract_xml_answer(completion[0]["content"]) for completion in completions]

    tasks = []
    test_indices = []  # Track which completion each task belongs to
    
    for i in range(len(model_answers)):
        for test_case in answers[i]:
            tasks.append(model_answers[i] + "\n" + test_case)
            test_indices.append(i)

    optimal_workers = min(len(tasks), MAX_PROCESSES)
    chunk_size = max(1, len(tasks) // optimal_workers)

    # Use shared executor for better performance
    executor = get_process_executor(optimal_workers)
    results = list(executor.map(runs_without_error, tasks, chunksize=chunk_size))

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
            reward = math.pow(accuracy, 3) * 2  # Cube accuracy and multiply by 2
        else:
            reward = 0.0
        rewards.append(reward)
    
    return rewards

def multiprocessing_soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that loosely checks if the completion has a specific format."""
    responses = [completion[0]["content"] for completion in completions]

    optimal_workers = min(len(responses), MAX_PROCESSES)
    chunk_size = max(1, len(responses) // optimal_workers)

    # Use fresh executor
    executor = get_process_executor(optimal_workers)
    results = list(executor.map(check_format, responses, chunksize=chunk_size))

    return results

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """
    Reward function that loosely checks if the completion has a specific format,
    without using multiprocessing.
    """
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>.*"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, re.S) for r in responses]
    return [0.25 if match else 0.0 for match in matches]

if __name__ == "__main__":
    try:
        # Print CPU information
        print(f"Number of CPU cores available: {os.cpu_count()}")
        print(f"Using maximum of {MAX_PROCESSES} worker processes")

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
        print(f"Using ProcessPoolExecutor with worker-specific WASM environments")
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
            mp_execution_results = multiprocessing_code_execution_reward_func(large_completions)
            mp_execution_time = time.time() - start_time
            mp_execution_total_time += mp_execution_time
            # print(f"  Code execution: {mp_execution_time:.4f}s")

            # Test multiprocessing_answer_execution_reward_func
            start_time = time.time()
            mp_answer_execution_results = multiprocessing_answer_execution_reward_func(large_completions, large_answers)
            mp_answer_time = time.time() - start_time
            mp_answer_execution_total_time += mp_answer_time
            # print(f"  Answer execution: {mp_answer_time:.4f}s")

            # Test multiprocessing_soft_format_reward_func
            start_time = time.time()
            mp_soft_format_results = multiprocessing_soft_format_reward_func(large_completions)
            mp_soft_format_time = time.time() - start_time
            mp_soft_format_total_time += mp_soft_format_time
            # print(f"  Format check: {mp_soft_format_time:.4f}s")

            # Test soft_format_reward_func
            start_time = time.time()
            soft_format_results = soft_format_reward_func(large_completions)
            soft_format_time = time.time() - start_time
            soft_format_total_time += soft_format_time
            # print(f"  Format check: {soft_format_time:.4f}s")

            
            # Calculate total time for this run
            run_time = time.time() - run_start_time
            total_run_time += run_time
            
            # print(f"  Run {i+1} completed in {run_time:.4f} seconds")

        # Calculate and print average times
        print("-" * 100)
        print("AVERAGE EXECUTION TIMES OVER", num_runs, "RUNS:")
        print("-" * 100)
        print(f"multiprocessing_code_execution_reward_func avg time: {mp_execution_total_time/num_runs:.4f} seconds")
        print(f"multiprocessing_answer_execution_reward_func avg time: {mp_answer_execution_total_time/num_runs:.4f} seconds")
        print(f"multiprocessing_soft_format_reward_func avg time: {mp_soft_format_total_time/num_runs:.4f} seconds")
        print(f"soft_format_reward_func avg time: {soft_format_total_time/num_runs:.4f} seconds")
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
</answer>"""
                }
            ]
        ]

        # Test cases - 3 should pass, 2 should fail (60% accuracy)
        test_answers = [[
            "assert add(1, 2) == 3",  # Pass
            "assert add(5, 7) == 12",  # Pass
            "assert add(10, 20) == 30",  # Pass
            "assert add(-1, 5) == 4",  # Fail - will return 0
            "assert add(-5, -10) == -15"  # Fail - will return 0
        ]]

        print("Input code:")
        print(extract_xml_answer(test_completion[0][0]["content"]))
        print("\nTest cases:")
        for tc in test_answers[0]:
            print(f"- {tc}")

        # Run the function and measure time
        print("\nExecuting test...")
        start_time = time.time()
        results = multiprocessing_answer_execution_reward_func(test_completion, test_answers)
        elapsed = time.time() - start_time

        print(f"\nResults: {results}")
        print(f"Execution time: {elapsed:.4f} seconds")

        # Calculate the expected reward (accuracy^3 * 2)
        expected_reward = math.pow(3/5, 3) * 2
        print(f"Expected reward with 3/5 accuracy: {expected_reward:.4f}")

        # Add assertion for partial success implementation 
        # Allow some floating point tolerance
        assert abs(results[0] - expected_reward) < 0.1, f"Expected ~{expected_reward:.4f} for 60% accuracy, got {results[0]}"
        print("✅ Assertion passed: 60% correct implementation received expected reward")

    finally:
        # Ensure executor is properly shutdown
        if "_process_executor" in globals() and _process_executor is not None:
            print("Explicit cleanup of process executor")
            _process_executor.shutdown()
            _process_executor = None 