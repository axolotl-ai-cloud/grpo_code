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


class PythonWasmEnvironment:
    """A reusable WASM environment for running Python code."""

    def __init__(self, wasm_path="wasm/python-3.12.0.wasm", fuel=1_000_000_000):
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


global env
env = PythonWasmEnvironment()
MAX_PROCESSES = 64


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
    return [does_compile(model_answer) for model_answer in model_answers]


def multiprocessing_compile_env_reward_func(completions: list[list[dict]], **kwargs) -> list[float]:
    model_answers = [extract_xml_answer(completion[0]["content"]) for completion in completions]
    max_processes = min(len(model_answers), MAX_PROCESSES)

    with Pool(processes=max_processes) as pool:
        results = pool.map(does_execute, model_answers)
    results = [0.5 if result == 1.0 else -0.25 for result in results]
    return results


def multiprocessing_compiles_reward_func(completions: list[list[dict]], **kwargs) -> list[float]:
    model_answers = [extract_xml_answer(completion[0]["content"]) for completion in completions]
    max_processes = min(len(model_answers), MAX_PROCESSES)

    with Pool(processes=max_processes) as pool:
        # Map does_compile to each model answer, then transform results to 0.5 for success, -0.25 for failure
        raw_results = pool.map(does_compile, model_answers)

    compile_rewards = [0.5 if result == 1.0 else -0.25 for result in raw_results]
    return compile_rewards


def multiprocessing_answer_env_reward_func(
    completions: list[list[dict]], answers: list[list[str]], **kwargs
) -> list[float]:
    model_answers = [extract_xml_answer(completion[0]["content"]) for completion in completions]

    tasks = []
    for i in range(len(model_answers)):
        for test_case in answers[i]:
            tasks.append(model_answers[i] + "\n" + test_case)

    # flatten tests to maximize throughput
    max_processes = min(len(tasks), MAX_PROCESSES)
    with Pool(processes=max_processes) as pool:
        results = pool.map(does_execute, tasks)

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


def multiprocessing_answer_reward_func(
    completions: list[list[dict]], answers: list[list[str]], **kwargs
) -> list[float]:
    # First extract answers, then check compilation
    model_answers = [extract_xml_answer(completion[0]["content"]) for completion in completions]
    tasks = []
    for i in range(len(model_answers)):
        for test_case in answers[i]:
            tasks.append(model_answers[i] + "\n" + test_case)

    # flatten tests to maximize throughput
    max_processes = min(len(tasks), MAX_PROCESSES)

    with Pool(processes=max_processes) as pool:
        results = pool.map(does_compile, tasks)

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
    for _ in range(100):
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
        large_answers = [["assert foo(1) == 2", "assert foo(2) == 3"]] * 32

        print("-" * 100)
        print(f"Testing with {num_copies} copies of the example completion")
        print("-" * 100)

        # test compile_reward_fun

        # # Test compile_reward_func
        # start_time = time.time()
        # compile_results = compile_reward_func(large_completions)
        # compile_time = time.time() - start_time
        # print(compile_results)
        # print(f"compile_reward_func time: {compile_time:.4f} seconds")
        # print("-" * 100)

        # Test multiprocessing_compiles_reward_func
        start_time = time.time()
        mp_compile_env_results = multiprocessing_compile_env_reward_func(large_completions)
        mp_compile_time = time.time() - start_time
        print(mp_compile_env_results)
        print(f"multiprocessing_compiles_env_reward_func time: {mp_compile_time:.4f} seconds")
        # print(f"Results match: {compile_results == mp_compile_env_results}")
        print("-" * 100)

        # Test multiprocessing_compiles_reward_func
        start_time = time.time()
        mp_compile_results = multiprocessing_compiles_reward_func(large_completions)
        mp_compile_time = time.time() - start_time
        print(mp_compile_results)
        print(f"multiprocessing_compiles_reward_func time: {mp_compile_time:.4f} seconds")
        # print(f"Results match: {compile_results == mp_compile_results}")
        print("-" * 100)

        # Test calculate_accuracy (via a wrapper to match signature)
        def accuracy_wrapper(completions, answers):
            model_answers = [extract_xml_answer(completion[0]["content"]) for completion in completions]
            return [
                calculate_accuracy(model_answer, answer_list)
                for model_answer, answer_list in zip(model_answers, answers)
            ]

        # start_time = time.time()
        # accuracy_results = accuracy_wrapper(large_completions, large_answers)
        # accuracy_time = time.time() - start_time
        # print(f"accuracy_wrapper time: {accuracy_time:.4f} seconds")

        # Test multiprocessing_answer_reward_func
        start_time = time.time()
        mp_accuracy_results = multiprocessing_answer_reward_func(large_completions, large_answers)
        mp_accuracy_time = time.time() - start_time
        print(f"multiprocessing_answer_reward_func time: {mp_accuracy_time:.4f} seconds")
        # print(f"Speedup: {accuracy_time / mp_accuracy_time:.2f}x")
        # print(f"Results match: {accuracy_results == mp_accuracy_results}")
        print("-" * 100)

        # test multiprocess_answer_env_reward_func
        start_time = time.time()
        mp_accuracy_env_results = multiprocessing_answer_env_reward_func(large_completions, large_answers)
        mp_accuracy_env_time = time.time() - start_time
        print(f"multiprocessing_answer_env_reward_func time: {mp_accuracy_env_time:.4f} seconds")
        # print(f"Results match: {accuracy_results == mp_accuracy_env_results}")
        print("-" * 100)

        # test soft_format_reward_func
        start_time = time.time()
        soft_format_results = soft_format_reward_func(large_completions)
        soft_format_time = time.time() - start_time
        print(soft_format_results)
        print(f"soft_format_reward_func time: {soft_format_time:.4f} seconds")
        print("-" * 100)

        # test multiprocessing_soft_format_reward_func
        start_time = time.time()
        mp_soft_format_results = multiprocessing_soft_format_reward_func(large_completions)
        mp_soft_format_time = time.time() - start_time
        print(mp_soft_format_results)
        print(f"multiprocessing_soft_format_reward_func time: {mp_soft_format_time:.4f} seconds")
        print(f"Results match: {soft_format_results == mp_soft_format_results}")
        print("-" * 100)

        def syntax_check_wrapper(completions):
            model_answers = [extract_xml_answer(completion[0]["content"]) for completion in completions]
            return [does_compile_syntax_check(model_answer) for model_answer in model_answers]

        # test syntax_check_wrapper
        start_time = time.time()
        syntax_check_results = syntax_check_wrapper(large_completions)
        syntax_check_time = time.time() - start_time
        print(syntax_check_results)
        print(f"syntax_check_wrapper time: {syntax_check_time:.4f} seconds")
        print("-" * 100)

        # test multiprocessing_does_compile_syntax_check_reward_func
        start_time = time.time()
        mp_compile_syntax_check_results = multiprocessing_does_compile_syntax_check_reward_func(large_completions)
        mp_compile_syntax_check_time = time.time() - start_time
        print(mp_compile_syntax_check_results)
        print(
            f"multiprocessing_does_compile_syntax_check_reward_func time: {mp_compile_syntax_check_time:.4f} seconds"
        )
        print(f"Results match: {syntax_check_results == mp_compile_syntax_check_results}")
        print("-" * 100)
