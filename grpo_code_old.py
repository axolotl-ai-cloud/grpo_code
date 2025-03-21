import os
# Set this at the very top of your file, before any other imports
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from multiprocessing import Pool
import re
from wasmtime import Config, Engine, Linker, Module, Store, WasiConfig
import math

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>

Additionally, you may optionally use the following imports:

import time
import itertools
from itertools import accumulate, product, permutations, combinations
import collections
from collections import Counter, OrderedDict, deque, defaultdict, ChainMap
from functools import lru_cache
import math
from typing import List, Dict, Tuple, Optional, Any

If you choose to use any of these imports, ensure they are included
inside the <answer> </answer> tags, e.g:

<answer>
import time
import math
...
</answer>

You may not utilise any other imports or filesystem operations. 
"""

class Result:
    def __init__(self, result, mem_size, data_len, consumed):
        self.result = result
        self.mem_size = mem_size
        self.data_len = data_len
        self.consumed = consumed

    def __str__(self):
        return f"""\
result:

{self.result}

mem size pages of 64kb: {self.mem_size}
data length: {self.data_len}
fuel consumed: {self.consumed}
"""
def axolotl_acecode_transform(cfg, *args, **kwargs):
    def transform_fn(example, tokenizer=None):
        return {
            "prompt": [{"role": "user", "content": example["question"] + "\n\n" + SYSTEM_PROMPT}],
            "answers": example["test_cases"],
        }
    return transform_fn, {"remove_columns": ["question", "test_cases"]}

def run_python_code(code: str, fuel: int = 100_000_000) -> str:
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

def does_compile_syntax_check(predicted_answer: str, **kwargs) -> float:
    try:
        compile(predicted_answer, '<string>', 'exec')
        return 1.0
    except Exception as e:
        return 0.0
    
def compile_syntax_check_reward_func(completions: list[list[dict]], **kwargs) -> list[float]:
    model_answers = [extract_xml_answer(completion[0]["content"]) for completion in completions]
    return [does_compile_syntax_check(model_answer) for model_answer in model_answers]

def multiprocessing_does_compile_syntax_check_reward_func(completions: list[list[dict]], **kwargs) -> list[float]:
    model_answers = [extract_xml_answer(completion[0]["content"]) for completion in completions]
    max_processes = min(len(model_answers), 128)
    with Pool(processes=max_processes) as pool:
        compile_rewards = pool.map(does_compile_syntax_check, model_answers)
    return compile_rewards

def does_compile(predicted_answer: str, **kwargs) -> float:
    try:
        run_python_code(predicted_answer)
        return 1.0
    except Exception as e:
        return 0.0

def multiprocessing_compiles_reward_func(completions: list[list[dict]], **kwargs) -> list[float]:
    model_answers = [extract_xml_answer(completion[0]["content"]) for completion in completions]
    max_processes = min(len(model_answers), 128)
    with Pool(processes=max_processes) as pool:
        compile_rewards = pool.map(does_compile, model_answers)
    return compile_rewards

def multiprocessing_filtered_compiles_reward_func(completions: list[list[dict]], answers: list[list[str]], **kwargs) -> list[float]:
    # First extract answers, then check compilation
    model_answers = [extract_xml_answer(completion[0]["content"]) for completion in completions]
    
    fn_name_rewards = function_name_reward_func(completions, answers)

    # Only process answers that have the correct function name
    valid_indices = [i for i in range(len(model_answers)) if fn_name_rewards[i] > 0]
    
    # If there are no valid answers, return early
    if not valid_indices:
        return [0.0] * len(completions)
    
    # Use multiprocessing to check compilation
    max_processes = min(len(valid_indices), 128)
    with Pool(processes=max_processes) as pool:
        valid_compile_rewards = pool.map(does_compile, [model_answers[i] for i in valid_indices])
    
    # Reconstruct the full list of rewards
    compile_rewards = [0.0] * len(completions)
    for idx, valid_idx in enumerate(valid_indices):
        compile_rewards[valid_idx] = valid_compile_rewards[idx]
    
    return compile_rewards
 
def soft_pass_k_answer_reward_func(completions: list[list[dict]], answers: list[list[str]], **kwargs) -> list[float]:
    """
    Calculate a softer version of pass@k that rewards partial success.
    
    Instead of requiring all test cases to pass, this function:
    1. Calculates the success rate for each completion
    2. Weights completions by their success rate
    3. Computes a weighted pass@k score
    
    Args:
        completions: List of model completions
        answers: List of test cases for each completion
        
    Returns:
        List of rewards based on the soft pass@k score
    """
    # Extract model answers
    model_answers = [extract_xml_answer(completion[0]["content"]) for completion in completions]
    k = len(completions)
    
    # Calculate success rate for each completion
    success_rates = []
    for i, model_answer in enumerate(model_answers):
        test_cases = answers[i]
        if not test_cases:
            success_rates.append(0.0)
            continue
            
        passed_tests = 0
        for test_case in test_cases:
            if is_answer_correct(model_answer, test_case):
                passed_tests += 1
                
        success_rate = passed_tests / len(test_cases)
        success_rates.append(success_rate)
    
    # Calculate weighted pass@k
    # Instead of counting only perfect solutions, we weight each solution by its success rate
    weighted_c = sum(success_rates)
    
    # Modified formula: 1 - (1 - weighted_c/k)^k
    if k > 0:
        soft_pass_at_k = 1.0 - (1.0 - weighted_c/k)**k
    else:
        soft_pass_at_k = 0.0
    
    # Return the same reward for all completions
    return [soft_pass_at_k * 4.0] * len(completions)  # Scale to match other rewards

def multiprocessing_soft_pass_k_answer_reward_func(completions: list[list[dict]], answers: list[list[str]], **kwargs) -> list[float]:
    """
    Reward function for having the correct answer, which uses multiprocessing to speed up the process.
    Similar to soft_pass_k_answer_reward_func but with multiprocessing.
    """
    # Extract model answers
    model_answers = [extract_xml_answer(completion[0]["content"]) for completion in completions]
    k = len(completions)
    
    # Create tasks for all test cases
    tasks = []
    test_case_counts = []
    for i, model_answer in enumerate(model_answers):
        test_cases = answers[i]
        test_case_counts.append(len(test_cases))
        for test_case in test_cases:
            tasks.append((model_answer, test_case))
    
    # Use multiprocessing to check if the answers are correct
    max_processes = min(len(tasks), 128)
    with Pool(processes=max_processes) as pool:
        results = pool.starmap(is_answer_correct, tasks)
    
    # Calculate success rate for each completion
    success_rates = []
    result_idx = 0
    for i in range(len(completions)):
        test_case_count = test_case_counts[i]
        if test_case_count == 0:
            success_rates.append(0.0)
            continue
            
        completion_results = results[result_idx:result_idx + test_case_count]
        result_idx += test_case_count
        success_rate = sum(completion_results) / test_case_count
        success_rates.append(success_rate)
    
    # Calculate weighted pass@k
    weighted_c = sum(success_rates)
    
    # Modified formula: 1 - (1 - weighted_c/k)^k
    if k > 0:
        soft_pass_at_k = 1.0 - (1.0 - weighted_c/k)**k
    else:
        soft_pass_at_k = 0.0
    
    # Return the same reward for all completions
    return [soft_pass_at_k * 2.0] * len(completions)  # Scale to match other rewards

def multiprocessing_filtered_answer_reward_func(completions: list[list[dict]], answers: list[list[str]], **kwargs) -> list[float]:
    """
    Reward function for having the correct answer, which uses multiprocessing to speed up the process.
    Similar to accuracy_reward_func but with multiprocessing.
    """
    # Check function names FIRST, just like in compile_reward_func
    fn_name_rewards = function_name_reward_func(completions, answers)
    model_answers = [extract_xml_answer(completion[0]["content"]) for completion in completions]
    
    # Only process answers that have the correct function name
    valid_indices = [i for i in range(len(model_answers)) if fn_name_rewards[i] > 0]
    
    # If there are no valid answers, return early
    if not valid_indices:
        return [0.0] * len(completions)
    
    # Create tasks only for valid answers
    tasks = []
    for i in valid_indices:
        for test_case in answers[i]:
            tasks.append((model_answers[i], test_case))
    
    # Use multiprocessing to check if the answers are correct
    max_processes = min(len(tasks), 128)
    with Pool(processes=max_processes) as pool:
        results = pool.starmap(is_answer_correct, tasks)
    
    # Calculate accuracy for each valid completion
    rewards = [0.0] * len(completions)
    result_idx = 0
    for i in valid_indices:
        test_case_count = len(answers[i])
        completion_results = results[result_idx:result_idx + test_case_count]
        result_idx += test_case_count
        accuracy = sum(completion_results) / test_case_count
        rewards[i] = math.pow(accuracy, 2) * 2
    
    return rewards

def multiprocessing_answer_reward_func(completions: list[list[dict]], answers: list[list[str]], **kwargs) -> list[float]:
    # First extract answers, then check compilation
    model_answers = [extract_xml_answer(completion[0]["content"]) for completion in completions]
    tasks = []      
    for i in range(len(model_answers)):
        for test_case in answers[i]:
            tasks.append((model_answers[i], test_case))

    # flatten tests to maximize throughput
    max_processes = min(len(tasks), 128)

    with Pool(processes=max_processes) as pool:
        results = pool.starmap(is_answer_correct, tasks)
    
    # Reconstruct results by completion
    rewards = [0.0] * len(completions)
    result_idx = 0
    for i in range(len(completions)):
        test_case_count = len(answers[i])
        completion_results = results[result_idx:result_idx + test_case_count]
        result_idx += test_case_count
        accuracy = sum(completion_results) / test_case_count
        rewards[i] = math.pow(accuracy, 2) * 2  # Square and multiply by 2 as in other functions
    
    return rewards

def is_answer_correct(completion: str, answer:str, **kwargs) -> float:
    try:
        run_python_code(completion + "\n\n" + answer)
        return 1.0
    except Exception as e:
        return 0
    
def calculate_accuracy(completion: str, answers: list[str], **kwargs) -> float:
    """
    Reward function for having the correct answer.
    """
    score = 0
    for test_case in answers:
        score += is_answer_correct(completion, test_case)
    accuracy = score / len(answers)
    return math.pow(accuracy, 2) * 2

def function_name_reward_func(completions: list[list[dict]], answers: list[list[str]], **kwargs) -> list[float]:
    """
    Reward function for having the correct function name.
    """
    correct_fn_names = [answers[0].split("(")[0].split("assert ")[1] for answers in answers]
    model_answers = [extract_xml_answer(completion[0]["content"]) for completion in completions]
    return [0.5 if "def " + correct_fn_name + "(" in model_answer else -0.5 for correct_fn_name, model_answer in zip(correct_fn_names, model_answers)]

def compile_reward_func(completions: list[list[dict]], **kwargs) -> list[float]:
    """
    Reward function for compiling the code.
    """
    model_answers = [extract_xml_answer(completion[0]["content"]) for completion in completions]
    return [does_compile(model_answer) for model_answer in model_answers]
def accuracy_reward_func(completions: list[list[dict]], answers: list[list[str]], **kwargs) -> list[float]:
    """
    Reward function for having the correct function name.
    """
    model_answers = [extract_xml_answer(completion[0]["content"]) for completion in completions]
    return [calculate_accuracy(model_answer, answers) for model_answer in model_answers]

def correctness_reward_func(completions: list[list[dict]], answers: list[list[str]], **kwargs) -> list[float]:
    """
    Args:
        completions (list[list[dict]]): A list of completions from the model to be scored.
        answers (list[list[str]]): A list of answers to be scored against. These will be in the format
            ["assert fn() == test_case_zero", "assert fn() == test_case_one", ...]

    """
    fn_name_rewards = function_name_reward_func(completions, answers)

    # # check if the completion compiles
    # compile_rewards = []
    # for fn_name_reward, model_answer in zip(fn_name_rewards, model_answers):
    #     if fn_name_reward == 0.25:
    #         compile_rewards.append(compiles_reward_func(model_answer))
    #     else:
    #         compile_rewards.append(0.0)
    
    # check if the completion is correct
    correct_rewards = []
    for i in range(len(completions)):
        if fn_name_rewards[i] > 0:
            extracted_answer = extract_xml_answer(completions[i][0]["content"])
            correct_rewards.append(calculate_accuracy(extracted_answer, answers[i]))
        else:
            correct_rewards.append(-0.1)
    # import pdb; pdb.set_trace()
    print(f"total_fn_name_reward: {sum(fn_name_rewards)}")
    print(f"total_correct_reward: {sum(correct_rewards)}")
    # combine all the rewards
    return correct_rewards

def extract_xml_answer(text: str) -> str:
    # collect the answer between the last <answer></answer> tag
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    """
    Reward function for having exactly one of each <reasoning>, </reasoning>, <answer>, and </answer> tag.
    """
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\s?$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """
    Reward function that loosely checks if the completion has a specific format,
    without penalizing adherence to newlines.
    """
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>.*"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, re.S) for r in responses]
    return [0.25 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>") == 1:
        count += 0.125
    if text.count("</reasoning>") == 1:
        count += 0.125
    if text.count("<answer>") == 1:
        count += 0.125
    if text.count("</answer>") == 1:
        count += 0.125
        # penalize extra tokens after the answer tag
    count -= (len(text.split("</answer>")[-1]) - 1) * 0.001
    return count

if __name__ == "__main__":
    
    # exit()
    # Add this before your existing tests
#     # Test soft_format_reward_func with a specific input
#     test_input = [
#         [
#             {
#                 "content": """<reasoning>
# To solve the problem of finding the longest common substring between two strings, one approach can be to generate all possible substrings from both strings, then compare them, and finally find the longest one that exists in both strings.

# However, to optimize this approach, we could skip generating all possible substrings. Instead, we can use a dictionary to keep track of the lengths of common substrings and the first occurrence of such substrings in both strings. This reduces the complexity but may still be inefficient for very large input strings.

# A more efficient approach is to use dynamic programming to build a 2D array where each cell `dp[i][j]` represents the length of the longest common suffix of the substrings `s1[0...i]` and `s2[0...j]`. By iterating over the characters in both strings and comparing them, we increment the corresponding cell in the `dp` array. This approach ensures that we only consider potential common substrings and their lengths efficiently.

# Finally, we trace back the `dp` array to determine the actual longest common substring, starting from the cell with the maximum value in `dp`.

# The function will return the longest common substring or an empty string if there is no common substring.
# </reasoning>

# <answer>
# def longest_common_substring(str1: str, str2: str) -> str:
#     m, n = len(str1), len(str2)
#     dp = [[0] * (n + 1) for _ in range(m + 1)]
#     longest_len = 0
#     end_position = 0

#     for i in range(m):
#         for j in range(n):
#             if str1[i] == str2[j]:
#                 dp[i + 1][j + 1] = dp[i][j] + 1
#                 if dp[i + 1][j + 1] > longest_len:
#                     longest_len = dp[i + 1][j + 1]
#                     end_position = i + 1

#     return str1[end_position - longest_len: end_position]

# # Test the function
# str1 = "ABABC"
# str2 = "BABC"
# print(f"Longest common substring: {longest_common_substring(str1, str2)}")  # Expected: "ABAB"
# </answer>

# This function implements the dynamic programming approach to finding the longest common substring between two input strings. It avoids listing all potential substrings and instead uses efficient array operations and"""
#             }
#         ]
#     ]
#     # Test function_name_reward_func
#     test_answers = [["assert longest_common_substring('ABABC', 'BABC') == 'ABC'"]]  # Expected function name
    
#     # Also test compile rewards
#     compile_result = compile_reward_func(test_input, test_answers)
#     print("\nTesting compile_reward_func:")
#     print(f"Result: {compile_result}")
#     # Run the function_name_reward_func on the test input
#     function_name_result = function_name_reward_func(test_input, test_answers)
    
#     # Print the result
#     print("\nTesting function_name_reward_func:")
#     print(f"Expected function name: {test_answers[0]}")
#     print(f"Result: {function_name_result}")
#     # Run the soft_format_reward_func on the test input
#     result = soft_format_reward_func(test_input)
    
#     # Print the result
#     print("Testing soft_format_reward_func:")
#     print(f"Input: {test_input[0][0]['content'][:100]}...")
#     print(f"Result: {result}")
#     exit()
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
    
    # Create multiple copies to better measure performance difference
    num_copies = 16
    large_completions = example_completion * num_copies
    large_answers = [["assert foo(1) == 2", "assert foo(2) == 3"]] * 32

    print("-" * 100)
    print(f"Testing with {num_copies} copies of the example completion")
    print("-" * 100)
    
    # Test compile_reward_func
    start_time = time.time()
    compile_results = compile_reward_func(large_completions)
    compile_time = time.time() - start_time
    print(f"compile_reward_func time: {compile_time:.4f} seconds")
    
    # Test multiprocessing_compiles_reward_func
    start_time = time.time()
    mp_compile_results = multiprocessing_compiles_reward_func(large_completions, large_answers)
    mp_compile_time = time.time() - start_time
    print(f"multiprocessing_compiles_reward_func time: {mp_compile_time:.4f} seconds")
    print(f"Speedup: {compile_time / mp_compile_time:.2f}x")
    print(f"Results match: {compile_results == mp_compile_results}")
    print("-" * 100)
    
    # Test calculate_accuracy (via a wrapper to match signature)
    def accuracy_wrapper(completions, answers):
        model_answers = [extract_xml_answer(completion[0]["content"]) for completion in completions]
        return [calculate_accuracy(model_answer, answer_list) for model_answer, answer_list in zip(model_answers, answers)]
    
    start_time = time.time()
    accuracy_results = accuracy_wrapper(large_completions, large_answers)
    accuracy_time = time.time() - start_time
    print(f"accuracy_wrapper time: {accuracy_time:.4f} seconds")
    
    # Test multiprocessing_answer_reward_func
    start_time = time.time()
    mp_accuracy_results = multiprocessing_answer_reward_func(large_completions, large_answers)
    mp_accuracy_time = time.time() - start_time
    print(f"multiprocessing_answer_reward_func time: {mp_accuracy_time:.4f} seconds")
    print(f"Speedup: {accuracy_time / mp_accuracy_time:.2f}x")
    print(f"Results match: {accuracy_results == mp_accuracy_results}")
    print("-" * 100)

