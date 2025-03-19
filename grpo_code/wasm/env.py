from wasmtime import Config, Engine, Linker, Module, Store, WasiConfig


class PythonWasmEnvironment:
    """A reusable WASM environment for running Python code."""

    _instance = None

    def __init__(self, wasm_path="./python-3.12.0.wasm", fuel=1_000_000_000):
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


    @classmethod
    def get_instance(cls, wasm_path="./python-3.12.0.wasm", fuel=1_000_000_000):
        """Get or create the singleton instance of the environment."""
        if cls._instance is None:
            cls._instance = cls(wasm_path, fuel)
        return cls._instance

    @classmethod
    def set_instance(cls, instance):
        """Set the singleton instance."""
        if not isinstance(instance, cls):
            raise TypeError(f"Expected instance of {cls.__name__}, got {type(instance).__name__}")
        cls._instance = instance

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

env = PythonWasmEnvironment(wasm_path="./python-3.12.0.wasm", fuel=1_000_000_000)
PythonWasmEnvironment.set_instance(env)

def does_execute(code: str) -> bool:
    try:
        env.run_code(code)
        return True
    except Exception as e:
        return False

