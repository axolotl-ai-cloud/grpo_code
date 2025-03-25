from concurrent.futures import ProcessPoolExecutor

from grpo_code.wasm import does_code_run, PythonWasmEnvironment

_executor, worker_env = None, None


def get_executor(
    max_processes: int, wasm_path: str, fuel: int
) -> None | ProcessPoolExecutor:
    """
    Get the executor for the given number of processes and WASM environment.
    For parallel execution, we use a multiprocessing executor, where a pool
    of workers, each with their own WASM environment, execute the tasks in parallel.

    For single process execution, we instead initialize a global WASM environment.

    Args:
        max_processes (int): The maximum number of processes to use.
        wasm_path (str): The path to the .wasm file.
        fuel (int): The amount of fuel to use for the WASM environment.

    Returns:
        executor (None | ProcessPoolExecutor): If parallel execution is requested,
            we return a ProcessPoolExecutor, otherwise we return None.
    """
    global _executor, worker_env
    if max_processes > 1:
        from grpo_code.parallel_executor import get_multiprocessing_executor

        _executor = get_multiprocessing_executor(max_processes, wasm_path, fuel)
        return _executor
    if worker_env is None:
        import grpo_code.wasm as wasm

        wasm.worker_env = PythonWasmEnvironment(wasm_path, fuel)


def execute_tasks(
    tasks: list[str], max_processes: int, wasm_path: str, fuel: int, task_timeout: int
):
    """
    Run a list of code snippets in a WASM environment.

    Args:
        tasks (list[str]): The list of code snippets to run.
        max_processes (int): The maximum number of processes to use.
        wasm_path (str): The path to the .wasm file.
        fuel (int): The amount of fuel to use for the WASM environment.
        task_timeout (int): If using multiprocessing, the timeout for each task.

    Returns:
        list[bool]: The list of results from running the code snippets.
    """
    executor = get_executor(max_processes, wasm_path, fuel)
    if max_processes > 1:
        from grpo_code.parallel_executor import run_tasks_with_multiprocessing_executor
        return run_tasks_with_multiprocessing_executor(executor, tasks, task_timeout)
    else:
        return list(map(does_code_run, tasks))
