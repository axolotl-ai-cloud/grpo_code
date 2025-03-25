from wasmtime import Config, Engine, Linker, Module, Store, WasiConfig

worker_env = None


class PythonWasmEnvironment:
    """A reusable WASM environment for running Python code.

    Args:
        wasm_path (str): The path to the .wasm file.
        fuel (int): The amount of fuel to use for the WASM environment.
    """

    def __init__(self, wasm_path: str, fuel: int):
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

    def run_code(self, code: str):
        """Run Python code in the WASM environment subject to fuel limits.

        Args:
            code (str): The Python code to run.

        """
        config = WasiConfig()
        config.argv = ("python", "-c", code)
        config.inherit_env = False

        store = Store(self.engine)
        store.set_fuel(self.fuel)
        store.set_wasi(config)

        instance = self.linker.instantiate(store, self.python_module)
        start = instance.exports(store)["_start"]
        start(store)


def does_code_run(code: str) -> bool:
    """Execute code in the worker's WASM environment and check if it runs without errors.

    Args:
        code (str): The Python code to run.

    Returns:
        bool: True if the code runs without errors, False otherwise.
    """
    global worker_env
    try:
        worker_env.run_code(code)
        return True
    except Exception:
        return False
