import math
import os
import re
from pathlib import Path

import grpo_code
from grpo_code.executor import execute_tasks

WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
MAX_PROCESSES = max(1, int(os.environ.get("MAX_PROCESSES", 1)) // WORLD_SIZE)
TASK_TIMEOUT = int(os.environ.get("TASK_TIMEOUT", 1))
WASM_PATH = os.environ.get(
    "WASM_PATH", Path(grpo_code.__file__).parent.parent / "wasm" / "python-3.12.0.wasm"
)
FUEL = int(os.environ.get("FUEL", 1_000_000_000))

if not os.path.exists(WASM_PATH):
    raise FileNotFoundError(f"WASM file not found at {WASM_PATH}")


def extract_xml_answer(text: str) -> str:
    """
    Extract text between <answer> and </answer> tags.

    Args:
        text (str): The text to extract the answer from.
    Returns:
        str: The answer extracted from the text. "" if no answer is found.

    """
    match = re.search(r"<answer>(.*?)</answer>", text, re.S)
    return match.group(1).strip() if match else ""


def code_execution_reward_func(completions: list[list[dict]], **kwargs) -> list[float]:
    """
    Reward function for code execution.

    Args:
        completions (list[list[dict]]): The predicted code completions to execute. This takes the format
            [
                [
                    {"role": "user", "content": "<reasoning>...</reasoning><answer>...</answer>"}
                ]
            ]
    Returns:
        list[float]: The rewards for the completions. Each completion is rewarded 0.5 if the code executes, -0.25 otherwise.
    """
    model_answers = [
        extract_xml_answer(completion[0]["content"]) for completion in completions
    ]
    task_results = execute_tasks(
        model_answers, MAX_PROCESSES, WASM_PATH, FUEL, TASK_TIMEOUT
    )
    return [0.5 if result == 1.0 else -0.25 for result in task_results]


def answer_execution_reward_func(
    completions: list[list[dict]], answers: list[list[str]], **kwargs
) -> list[float]:
    """
    Reward function for answer execution.

    Args:
        completions (list[list[dict]]): The predicted code completions to execute. This takes the format
            [
                [
                    {"role": "user", "content": "<reasoning>...</reasoning><answer>...</answer>"}
                ]
            ]
        answers (list[list[str]]): The expected answers to the code completions. These take the form of executable
            assert statements, e.g.
            [
                [
                    "assert foo(1) == 2",
                    "assert foo(2) == 3",
                ]
            ]
    Returns:
        list[float]: The accuracy rewards for the completions. Each completion is rewarded
            (accuracy)^3 * 2, where accuracy is the proportion of test cases that pass.
    """
    model_answers = [
        extract_xml_answer(completion[0]["content"]) for completion in completions
    ]
    tasks = []
    test_indices = []
    for i, (code, tests) in enumerate(zip(model_answers, answers)):
        for test in tests:
            tasks.append(code + "\n" + test)
            test_indices.append(i)

    task_results = execute_tasks(tasks, MAX_PROCESSES, WASM_PATH, FUEL, TASK_TIMEOUT)

    completion_results = {}
    for idx, result in zip(test_indices, task_results):
        if idx not in completion_results:
            completion_results[idx] = []
        completion_results[idx].append(result)

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
    """
    Reward function for soft format checking.

    Args:
        completions (list[list[dict]]): The predicted code completions to execute. This takes the format
            [
                [
                    {"role": "user", "content": content}
                ]
            ]
    Returns:
        list[float]: The rewards for the completions. Each completion is rewarded 0.25 if the format is correct, 0.0 otherwise.
    """

    responses = [completion[0]["content"] for completion in completions]
    rewards = []
    for response in responses:
        if re.match(
            r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>.*", response, re.S
        ):
            rewards.append(0.25)
        else:
            rewards.append(0.0)
    return rewards


def test_rewards():
    """
    Test the reward functions and measure their performance.
    This function tests:
    1. code_execution_reward_func - for checking if code executes
    2. answer_execution_reward_func - for checking if code passes test cases
    3. soft_format_reward_func - for checking if responses follow the expected format

    The test includes both performance benchmarking and functional verification.
    """
    import time

    print("=" * 80)
    print("TESTING REWARD FUNCTIONS")
    print("=" * 80)

    # Create example code completion for testing
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

    # Create multiple copies to test with larger batch
    num_copies = 32
    large_completions = example_completion * num_copies
    large_answers = [["assert foo(1) == 2", "assert foo(2) == 3"]] * num_copies

    # Number of runs for averaging performance
    num_runs = 5

    print(f"Testing with {num_copies} copies of the example completion")
    print(f"Running each test {num_runs} times and reporting averages")
    print("-" * 80)

    # Track execution times
    code_execution_total_time = 0
    answer_execution_total_time = 0
    soft_format_total_time = 0
    total_run_time = 0

    try:
        for i in range(num_runs):
            run_start_time = time.time()

            if i % 2 == 0:
                print(f"Run {i+1}/{num_runs}...")

            # Test code_execution_reward_func
            start_time = time.time()
            code_execution_results = code_execution_reward_func(large_completions)
            code_execution_time = time.time() - start_time
            code_execution_total_time += code_execution_time

            # Test answer_execution_reward_func
            start_time = time.time()
            answer_execution_results = answer_execution_reward_func(
                large_completions, large_answers
            )
            answer_execution_time = time.time() - start_time
            answer_execution_total_time += answer_execution_time

            # Test soft_format_reward_func
            start_time = time.time()
            soft_format_results = soft_format_reward_func(large_completions)
            soft_format_time = time.time() - start_time
            soft_format_total_time += soft_format_time

            # Calculate total time for this run
            run_time = time.time() - run_start_time
            total_run_time += run_time

            if i % 2 == 0:
                print(f"  Run {i+1} completed in {run_time:.4f} seconds")

        # Calculate and print average times
        print("-" * 80)
        print("AVERAGE EXECUTION TIMES OVER", num_runs, "RUNS:")
        print("-" * 80)
        print(
            f"code_execution_reward_func avg time: {code_execution_total_time/num_runs:.4f} seconds"
        )
        print(
            f"answer_execution_reward_func avg time: {answer_execution_total_time/num_runs:.4f} seconds"
        )
        print(
            f"soft_format_reward_func avg time: {soft_format_total_time/num_runs:.4f} seconds"
        )
        print(f"Average total time per run: {total_run_time/num_runs:.4f} seconds")
        print(f"Total benchmark time: {total_run_time:.4f} seconds")
        print("-" * 80)

        # Manual test with the 60% accuracy case
        print("\n" + "=" * 80)
        print("MANUAL TEST: answer_execution_reward_func")
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
        results = answer_execution_reward_func(test_completion, test_answers)
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

        # Test format checking
        print("\n" + "=" * 80)
        print("MANUAL TEST: soft_format_reward_func")
        print("=" * 80)

        format_tests = [
            [
                {
                    "role": "user",
                    "content": "<reasoning>Test reasoning</reasoning>\n<answer>Test answer</answer>",
                }
            ],
            [{"role": "user", "content": "No tags here, just plain text"}],
            [
                {
                    "role": "user",
                    "content": "<reasoning>Multi\nline\nreasoning</reasoning>\n<answer>Multi\nline\nanswer</answer>",
                }
            ],
        ]

        format_results = soft_format_reward_func(format_tests)
        print("Format check results:")
        for i, (test, result) in enumerate(zip(format_tests, format_results)):
            print(f"Test {i+1}: {result:.2f} - {'PASSED' if result > 0 else 'FAILED'}")
            print(
                f"  Content: {test[0]['content'][:50]}..."
                if len(test[0]["content"]) > 50
                else test[0]["content"]
            )
            print()

        expected_format_results = [0.25, 0.0, 0.25]
        assert (
            format_results == expected_format_results
        ), f"Expected {expected_format_results}, got {format_results}"
        print("✅ Assertion passed: Format check produces expected rewards")

    except Exception as e:
        import traceback

        print(f"ERROR: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    test_rewards()
