from concurrent.futures import ProcessPoolExecutor, TimeoutError, BrokenExecutor
import os
import multiprocessing
import re
import time
from wasmtime import Config, Engine, Linker, Module, Store, WasiConfig
import math
import atexit
import signal
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Compile regex patterns once for better performance
ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
FORMAT_PATTERN = re.compile(r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>.*", re.DOTALL)

def extract_xml_answer(text):
    """Extract the answer from the XML tags using compiled regex."""
    match = ANSWER_PATTERN.search(text)
    return match.group(1).strip() if match else ""

WASM_PATH = "/workspace/data/grpo_code/grpo_code/wasm/python-3.12.0.wasm"

if not os.path.exists(WASM_PATH):
    raise FileNotFoundError(f"WASM file not found at {WASM_PATH}")

# Maximum number of processes to use
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
MAX_PROCESSES = int(os.environ.get("MAX_PROCESSES", 32)) // WORLD_SIZE

# Task timeout - maximum time (in seconds) to wait for a worker to complete a task
TASK_TIMEOUT = int(os.environ.get("TASK_TIMEOUT", 1))

# Global executor
_executor = None

def get_executor():
    """Get or create the global ProcessPoolExecutor."""
    global _executor
    if _executor is None:
        _executor = ProcessPoolExecutor(
            max_workers=MAX_PROCESSES,
            initializer=worker_init
        )
        # Register cleanup on exit
        atexit.register(cleanup_executor)
        # Register signal handlers
        signal.signal(signal.SIGINT, cleanup_and_exit)
        signal.signal(signal.SIGTERM, cleanup_and_exit)
    return _executor

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
        config.inherit_env = False

        store = Store(self.engine)
        store.set_fuel(self.fuel)
        store.set_wasi(config)

        instance = self.linker.instantiate(store, self.python_module)
        start = instance.exports(store)["_start"]
        start(store)

# Worker function with initialization
def worker_init():
    """Initialize worker-specific environment."""
    global worker_env
    worker_env = PythonWasmEnvironment()

def runs_without_error(code: str) -> float:
    """Execute code in the worker's WASM environment and check if it runs without errors."""
    global worker_env
    try:
        worker_env.run_code(code)
        return 1.0
    except Exception as e:
        return 0.0

def check_format(text):
    """Check if text follows the expected format pattern."""
    return 0.25 if FORMAT_PATTERN.match(text) else 0.0

def get_executor():
    """Get or create the global ProcessPoolExecutor."""
    global _executor
    if _executor is None:
        _executor = ProcessPoolExecutor(
            max_workers=MAX_PROCESSES,
            initializer=worker_init
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
            print(f"WARNING: Task timed out after {timeout} seconds, recreating process pool...")
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

# Reward functions using ProcessPoolExecutor
def multiprocessing_code_execution_reward_func(completions: list[list[dict]], **kwargs) -> list[float]:
    """Reward function for executing the code."""
    model_answers = [extract_xml_answer(completion[0]["content"]) for completion in completions]
    results = run_with_executor(runs_without_error, model_answers)
    return [0.5 if result == 1.0 else -0.25 for result in results]

def multiprocessing_answer_execution_reward_func(
    completions: list[list[dict]], answers: list[list[str]], **kwargs
) -> list[float]:
    """Reward function for executing the code with test cases."""
    model_answers = [extract_xml_answer(completion[0]["content"]) for completion in completions]
    
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

def multiprocessing_soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that loosely checks if the completion has a specific format."""
    responses = [completion[0]["content"] for completion in completions]
    return run_with_executor(check_format, responses)

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

        # Create test cases with both valid and invalid code
        test_completions = [
            [  # Valid code
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
            ],
            [  # Syntax error
                {
                    "role": "user",
                    "content": """
                    <reasoning>
                    This code has a syntax error.
                    </reasoning>
                    <answer>
                    def foo(a)
                        return a + 1  # Missing colon
                    </answer>""",
                }
            ],
            [  # Runtime error
                {
                    "role": "user",
                    "content": """
                    <reasoning>
                    This code will cause a runtime error.
                    </reasoning>
                    <answer>
                    def foo(a):
                        return a + undefined_variable
                    </answer>""",
                }
            ],
            [  # Memory intensive
                {
                    "role": "user",
                    "content": """
                    <reasoning>
                    This code uses a lot of memory.
                    </reasoning>
                    <answer>
                    def foo(a):
                        x = [1] * 1000000  # Large list
                        return a + len(x)
                    </answer>""",
                }
            ]
        ]

        # Create corresponding test cases
        test_answers = [
            ["assert foo(1) == 2", "assert foo(2) == 3"],  # Valid tests
            ["assert foo(1) == 2"],  # Won't run due to syntax error
            ["assert foo(1) == 2"],  # Won't run due to runtime error
            ["assert foo(1) == 1000001"]  # Memory intensive test
        ]

        # Number of copies to simulate load
        num_copies = 8
        large_completions = test_completions * num_copies
        large_answers = test_answers * num_copies

        print("-" * 100)
        print(f"Testing with {len(large_completions)} total test cases ({num_copies} copies of each type)")
        print("Test cases include: valid code, syntax errors, runtime errors, and memory-intensive code")
        print("-" * 100)

        # Run tests multiple times to observe behavior
        num_runs = 10
        for i in range(num_runs):
            print(f"\nRun {i+1}/{num_runs}")
            
            start_time = time.time()
            results = multiprocessing_code_execution_reward_func(large_completions)
            execution_time = time.time() - start_time
            
            # Count different result types
            successes = sum(1 for r in results if r == 0.5)
            failures = sum(1 for r in results if r == -0.25)
            
            print(f"Execution time: {execution_time:.2f}s")
            print(f"Results: {len(results)} total, {successes} successes, {failures} failures")
            print(f"Average reward: {sum(results)/len(results):.3f}")

    finally:
        cleanup_executor()
        print("\nTest completed - executor cleaned up") 