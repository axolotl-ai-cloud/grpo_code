## Installation

```bash
pip install uv
uv venv
source .venv/bin/activate
uv pip install -e .
uv pip install -U packaging setuptools wheel ninja
uv pip install --no-build-isolation git+https://github.com/axolotl-ai-cloud/axolotl.git
uv pip install vllm==0.7.2 flash-attn 
uv pip install --no-deps git+https://github.com/huggingface/trl.git@main
```


## Evals

https://github.com/huggingface/ioi

https://livecodebench.github.io/

https://github.com/QwenLM/Qwen2.5-Coder/tree/main/qwencoder-eval/instruct

https://github.com/bigcode-project/bigcode-evaluation-harness/tree/main

## Python WASM Runtime

This project uses Python 3.12.0 compiled to WebAssembly from VMware Labs.

### Setup Options

#### A) Quick Setup (Default)
The Python WASM runtime is automatically downloaded during installation. No additional steps required.

#### B) Verify an Existing Download
If you already have the WASM file and want to verify its integrity:

1. Ensure you have both `python-3.12.0.wasm` and `python-3.12.0.wasm.sha256sum` in your directory
2. Run the verification command:

   **Linux/macOS:**
   ```bash
   sha256sum -c ./wasm/python-3.12.0.wasm.sha256sum
   ```

#### C) Manual Download
To download the runtime files yourself:

1. Download the Python WASM runtime:
   ```bash
   curl -LO https://github.com/vmware-labs/webassembly-language-runtimes/releases/download/python%2F3.12.0%2B20231211-040d5a6/python-3.12.0.wasm -o ./wasm/python-3.12.0.wasm
   ```

2. Download the SHA256 checksum file:
   ```bash
   curl -LO https://github.com/vmware-labs/webassembly-language-runtimes/releases/download/python%2F3.12.0%2B20231211-040d5a6/python-3.12.0.wasm.sha256sum -o ./wasm/python-3.12.0.wasm.sha256sum
   ```

3. Verify the download:
   ```bash
   sha256sum -c ./wasm/python-3.12.0.wasm.sha256sum
   ```

4. Place both files in your project directory or specify the path in your configuration.