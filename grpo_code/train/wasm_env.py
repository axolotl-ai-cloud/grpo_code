from multiprocessing import Pool
import os
import re
from wasmtime import Config, Engine, Linker, Module, Store, WasiConfig
import math

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
        """Run Python code in the WASM environment."""
        config = WasiConfig()
        config.argv = ("python", "-c", code)

        store = Store(self.engine)
        store.set_fuel(self.fuel)
        store.set_wasi(config)

        instance = self.linker.instantiate(store, self.python_module)
        start = instance.exports(store)["_start"]

        start(store)


WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
MAX_PROCESSES = int(os.environ.get("MAX_PROCESSES", 128)) // WORLD_SIZE

try:
    global env
    env = PythonWasmEnvironment()
except Exception as e:
    print("Error initializing PythonWasmEnvironment")
    print(e)
    raise e


def does_execute(code: str) -> bool:
    try:
        env.run_code(code)
        return 1.0
    except Exception as e:
        return 0.0


def does_compile(predicted_answer: str, **kwargs) -> float:
    try:
        run_python_code(predicted_answer)
        return 1.0
    except Exception as e:
        return 0.0


def run_python_code(code: str, fuel: int = 1_000_000_000) -> str:
    """
    Run Python code in a WASM environment with a configurable fuel limit.

    Args:
        code: The Python code to execute
        fuel: The maximum number of instructions to execute
              - 1M: Simple syntax checks
              - 10M: Basic test cases
              - 100M: Standard workloads (default)
              - 500M: Complex algorithms
              - 1B+: Very intensive operations

    Returns:
        The result of the execution
    """
    engine_cfg = Config()
    engine_cfg.consume_fuel = True
    engine_cfg.cache = True

    linker = Linker(Engine(engine_cfg))
    linker.define_wasi()

    python_module = Module.from_file(linker.engine, "wasm/python-3.12.0.wasm")

    config = WasiConfig()

    config.argv = ("python", "-c", code)

    store = Store(linker.engine)

    store.set_fuel(fuel)
    store.set_wasi(config)
    instance = linker.instantiate(store, python_module)

    start = instance.exports(store)["_start"]
    start(store)


def compile_reward_func(completions: list[list[dict]], **kwargs) -> list[float]:
    """
    Reward function for compiling the code.
    """
    model_answers = [extract_xml_answer(completion[0]["content"]) for completion in completions]
    results = [does_execute(model_answer) for model_answer in model_answers]
    return [0.5 if result == 1.0 else -0.25 for result in results]


def multiprocessing_code_execution_reward_func(completions: list[list[dict]], **kwargs) -> list[float]:
    model_answers = [extract_xml_answer(completion[0]["content"]) for completion in completions]

    optimal_workers = min(len(model_answers), MAX_PROCESSES)
    chunk_size = max(1, len(model_answers) // optimal_workers)

    with Pool(processes=min(len(model_answers), MAX_PROCESSES)) as pool:
        results = pool.map(does_execute, model_answers, chunksize=chunk_size)
    results = [0.5 if result == 1.0 else -0.25 for result in results]
    return results


def multiprocessing_answer_execution_reward_func(
    completions: list[list[dict]], answers: list[list[str]], **kwargs
) -> list[float]:
    model_answers = [extract_xml_answer(completion[0]["content"]) for completion in completions]

    tasks = []
    for i in range(len(model_answers)):
        for test_case in answers[i]:
            tasks.append(model_answers[i] + "\n" + test_case)

    # flatten tests to maximize throughput
    optimal_workers = min(len(tasks), MAX_PROCESSES)
    chunk_size = max(1, len(tasks) // optimal_workers)
    with Pool(processes=optimal_workers) as pool:
        results = pool.map(does_execute, tasks, chunksize=chunk_size)

    # Reconstruct results by completion
    rewards = [0.0] * len(completions)
    result_idx = 0
    for i in range(len(completions)):
        test_case_count = len(answers[i])
        completion_results = results[result_idx : result_idx + test_case_count]
        result_idx += test_case_count
        accuracy = sum(completion_results) / test_case_count
        rewards[i] = math.pow(accuracy, 3) * 2  # Square and multiply by 2 as in other functions

    return rewards


def calculate_accuracy(completion: str, answers: list[str], **kwargs) -> float:
    """
    Reward function for having the correct answer.
    """
    score = 0
    for test_case in answers:
        score += does_execute(completion + "\n" + test_case)
    accuracy = score / len(answers)
    return math.pow(accuracy, 2) * 2


def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """
    Reward function that loosely checks if the completion has a specific format,
    without penalizing adherence to newlines.
    """
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>.*"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, re.S) for r in responses]
    return [0.25 if match else 0.0 for match in matches]


def check_format(text):
    return 0.25 if re.match(r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>.*", text, re.S) else 0.0


def multiprocessing_soft_format_reward_func(completions, **kwargs) -> list[float]:
    """
    Reward function that loosely checks if the completion has a specific format,
    without penalizing adherence to newlines.
    """
    responses = [completion[0]["content"] for completion in completions]
    max_processes = min(len(responses), MAX_PROCESSES)
    with Pool(processes=max_processes) as pool:
        results = pool.map(check_format, responses)
    return results


def does_compile_syntax_check(predicted_answer: str, **kwargs) -> float:

    try:
        compile(predicted_answer, "<string>", "exec")
        return 0.25
    except Exception as e:
        return -0.1


def does_compile_syntax_check_reward_func(completions: list[list[dict]], **kwargs) -> list[float]:
    model_answers = [extract_xml_answer(completion[0]["content"]) for completion in completions]
    return [does_compile_syntax_check(model_answer) for model_answer in model_answers]


def multiprocessing_does_compile_syntax_check_reward_func(completions: list[list[dict]], **kwargs) -> list[float]:
    model_answers = [extract_xml_answer(completion[0]["content"]) for completion in completions]
    max_processes = min(len(model_answers), MAX_PROCESSES)
    with Pool(processes=max_processes) as pool:
        compile_rewards = pool.map(does_compile_syntax_check, model_answers)
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
        sleep(1000000)
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
        print("-" * 100)

        # Run each test multiple times and track total execution times
        compile_env_total_time = 0
        mp_compile_env_total_time = 0
        mp_answer_env_total_time = 0
        mp_soft_format_total_time = 0
        mp_compile_syntax_check_total_time = 0
        total_run_time = 0

        for i in range(num_runs):
            run_start_time = time.time()
            
            if i % 10 == 0:
                print(f"Run {i+1}/{num_runs}...")

            # Test compile_env_reward_func
            start_time = time.time()
            compile_env_results = compile_reward_func(large_completions)
            compile_env_total_time += time.time() - start_time
                
            # Test multiprocessing_compile_env_reward_func
            start_time = time.time()
            mp_compile_env_results = multiprocessing_code_execution_reward_func(large_completions)
            mp_compile_env_total_time += time.time() - start_time

            # Test multiprocessing_answer_env_reward_func
            start_time = time.time()
            mp_answer_env_results = multiprocessing_answer_execution_reward_func(large_completions, large_answers)
            mp_answer_env_total_time += time.time() - start_time

            # Test multiprocessing_soft_format_reward_func
            start_time = time.time()
            mp_soft_format_results = multiprocessing_soft_format_reward_func(large_completions)
            mp_soft_format_total_time += time.time() - start_time

            # Test multiprocessing_does_compile_syntax_check_reward_func
            start_time = time.time()
            mp_compile_syntax_check_results = soft_format_reward_func(large_completions)
            mp_compile_syntax_check_total_time += time.time() - start_time
            
            # Calculate total time for this run
            run_time = time.time() - run_start_time
            total_run_time += run_time
            
            if i % 10 == 0:
                print(f"  Run {i+1} completed in {run_time:.4f} seconds")

        # Calculate and print average times
        print("-" * 100)
        print("AVERAGE EXECUTION TIMES OVER", num_runs, "RUNS:")
        print("-" * 100)
        print(f"compile_env_reward_func avg time: {compile_env_total_time/num_runs:.4f} seconds")
        print(f"multiprocessing_compile_env_reward_func avg time: {mp_compile_env_total_time/num_runs:.4f} seconds")
        print(f"multiprocessing_answer_env_reward_func avg time: {mp_answer_env_total_time/num_runs:.4f} seconds")
        print(f"multiprocessing_soft_format_reward_func avg time: {mp_soft_format_total_time/num_runs:.4f} seconds")
        print(f"soft format avg time: {mp_compile_syntax_check_total_time/num_runs:.4f} seconds")
        print(f"Average total time per run: {total_run_time/num_runs:.4f} seconds")
        print(f"Total benchmark time: {total_run_time:.4f} seconds")
        print("-" * 100)

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
        # No explicit cleanup needed in this version as pools are created and destroyed per function call
        pass
    # manually run the answer env reward fn to test
    
