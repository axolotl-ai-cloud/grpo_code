import os
import re
import math
import time
from concurrent.futures import ProcessPoolExecutor
from wasmtime import Config, Engine, Linker, Module, Store, WasiConfig

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set spawn method for multiprocessing
# import multiprocessing
# multiprocessing.set_start_method("spawn", force=True)

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
print(f"Setting MAX_PROCESSES to {MAX_PROCESSES}")

# Optimize the WASM environment for faster initialization
class PythonWasmEnvironment:
    """A reusable WASM environment for running Python code."""

    def __init__(self, wasm_path=WASM_PATH, fuel=1_000_000_000):
        # Streamlined configuration with JIT optimization
        engine_cfg = Config()
        engine_cfg.consume_fuel = True
        engine_cfg.cache = True

        self.engine = Engine(engine_cfg)
        self.linker = Linker(self.engine)
        self.linker.define_wasi()
        self.python_module = Module.from_file(self.engine, wasm_path)
        self.fuel = fuel

    def run_code(self, code):
        # Optimize WasiConfig creation
        config = WasiConfig()
        config.argv = ("python", "-c", code)
        config.inherit_env = False  # Don't inherit environment variables

        store = Store(self.engine)
        store.set_fuel(self.fuel)
        store.set_wasi(config)

        instance = self.linker.instantiate(store, self.python_module)
        start = instance.exports(store)["_start"]
        start(store)

# Optimized worker functions using batching
def batch_run_code(batch):
    """Run multiple code snippets in a single worker process."""
    env = PythonWasmEnvironment()  # Create environment once per batch
    results = []
    
    for code in batch:
        try:
            env.run_code(code)
            results.append(1.0)
        except Exception:
            results.append(0.0)
    
    return results

def batch_format_check(batch):
    """Check format of multiple completions in a single worker process."""
    results = []
    for text in batch:
        results.append(0.25 if FORMAT_PATTERN.match(text) else 0.0)
    return results

def batch_syntax_check(batch):
    """Check syntax of multiple code snippets in a single worker process."""
    results = []
    for code in batch:
        try:
            compile(code, "<string>", "exec")
            results.append(0.25)
        except Exception:
            results.append(-0.1)
    return results

# Helper function to split data into batches
def create_batches(items, batch_size):
    """Split items into batches of specified size."""
    return [items[i:i+batch_size] for i in range(0, len(items), batch_size)]

# Optimized reward functions using batching
def multiprocessing_code_execution_reward_func(completions: list[list[dict]], **kwargs) -> list[float]:
    """Reward function for executing the code."""
    # Extract answers efficiently
    model_answers = [extract_xml_answer(completion[0]["content"]) for completion in completions]
    
    # Calculate optimal batch size and worker count
    batch_size = max(5, len(model_answers) // (MAX_PROCESSES * 2) + 1)
    batches = create_batches(model_answers, batch_size)
    worker_count = min(MAX_PROCESSES, len(batches))
    
    # Process batches in parallel
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        batch_results = list(executor.map(batch_run_code, batches))
    
    # Flatten results
    flat_results = []
    for batch in batch_results:
        flat_results.extend(batch)
    
    # Calculate rewards
    return [0.5 if result == 1.0 else -0.25 for result in flat_results[:len(completions)]]

def multiprocessing_answer_execution_reward_func(
    completions: list[list[dict]], answers: list[list[str]], **kwargs
) -> list[float]:
    """Reward function for executing the code with test cases."""
    model_answers = [extract_xml_answer(completion[0]["content"]) for completion in completions]

    all_tasks = []
    test_indices = []  # Track which completion each task belongs to
    
    for i in range(len(model_answers)):
        model_code = model_answers[i]
        for test_case in answers[i]:
            all_tasks.append(model_code + "\n" + test_case)
            test_indices.append(i)
    
    # Create larger batches for fewer process creations
    batch_size = max(5, len(all_tasks) // (MAX_PROCESSES * 2) + 1)
    batches = create_batches(all_tasks, batch_size)
    worker_count = min(MAX_PROCESSES, len(batches))
    
    # Process batches in parallel
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        batch_results = list(executor.map(batch_run_code, batches))
    
    # Flatten results
    flat_results = []
    for batch in batch_results:
        flat_results.extend(batch)
    
    # Map results back to completions
    completion_results = {}
    for idx, result in zip(test_indices, flat_results):
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

    # Calculate optimal batch size and worker count
    batch_size = max(10, len(responses) // (MAX_PROCESSES * 2) + 1)
    batches = create_batches(responses, batch_size)
    worker_count = min(MAX_PROCESSES, len(batches))
    
    # Process batches in parallel
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        batch_results = list(executor.map(batch_format_check, batches))
    
    # Flatten results
    flat_results = []
    for batch in batch_results:
        flat_results.extend(batch)
    
    return flat_results[:len(completions)]

def multiprocessing_syntax_check_reward_func(completions: list[list[dict]], **kwargs) -> list[float]:
    """Reward function for checking if the code has valid syntax."""
    model_answers = [extract_xml_answer(completion[0]["content"]) for completion in completions]

    # Calculate optimal batch size and worker count
    batch_size = max(10, len(model_answers) // (MAX_PROCESSES * 2) + 1)
    batches = create_batches(model_answers, batch_size)
    worker_count = min(MAX_PROCESSES, len(batches))
    
    # Process batches in parallel
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        batch_results = list(executor.map(batch_syntax_check, batches))
    
    # Flatten results
    flat_results = []
    for batch in batch_results:
        flat_results.extend(batch)
    
    return flat_results[:len(completions)]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """
    Reward function that loosely checks if the completion has a specific format,
    without using multiprocessing.
    """
    responses = [completion[0]["content"] for completion in completions]
    matches = [FORMAT_PATTERN.match(r) for r in responses]
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
        num_runs = 5
        
        print("-" * 100)
        print(f"Testing with {num_copies} copies of the example completion")
        print(f"Running each test {num_runs} times and reporting averages")
        print(f"Using optimized batching with {MAX_PROCESSES} max workers")
        print("-" * 100)

        # Run each test multiple times and track total execution times
        mp_execution_total_time = 0
        mp_answer_execution_total_time = 0
        mp_soft_format_total_time = 0
        mp_syntax_check_total_time = 0
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

            # Test multiprocessing_syntax_check_reward_func
            start_time = time.time()
            mp_syntax_check_results = multiprocessing_syntax_check_reward_func(large_completions)
            mp_syntax_time = time.time() - start_time
            mp_syntax_check_total_time += mp_syntax_time
            # print(f"  Syntax check: {mp_syntax_time:.4f}s")
            
            # Test non-mp soft_format_reward_func
            start_time = time.time()
            soft_format_results = soft_format_reward_func(large_completions)
            soft_format_time = time.time() - start_time
            soft_format_total_time += soft_format_time
            # print(f"  Non-MP format check: {soft_format_time:.4f}s")
            
            # Calculate total time for this run
            run_time = time.time() - run_start_time
            total_run_time += run_time
            
            # print(f"  Run {i+1} completed in {run_time:.4f} seconds")
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

    except Exception as e:
        import traceback
        print(f"ERROR: {e}")
        traceback.print_exc() 