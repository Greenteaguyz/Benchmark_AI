import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime

REQUIRED_DIRS = [
    "data/responses",
    "evaluation",
    "tool",
    "docs",
]

BASELINE_MODEL = "phi4-mini-reasoning"
TEST_PROMPT = "Explain in two steps how binary search works."


def run_command(command: list[str]) -> tuple[int, str]:
    """Execute a system shell command and capture stdout/stderr."""
    try:
        res = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        return res.returncode, res.stdout.strip()
    except FileNotFoundError:
        return -1, f"Command not found: {command[0]}"


def step_create_directories():
    print("[1/5] Initializing project folder layout...")
    for directory in REQUIRED_DIRS:
        os.makedirs(directory, exist_ok=True)
    print(f"Created: {', '.join(REQUIRED_DIRS)}")


def step_check_gpu():
    print("\n[2/5] Inspecting GPU and available VRAM...")
    code, out = run_command(["nvidia-smi"])
    if code != 0:
        print("WARNING: nvidia-smi failed. Ensure NVIDIA drivers are installed and on your PATH.")
        return "NVIDIA GPU inspection failed or not present."
    print("NVIDIA driver detected.")
    return out


def step_check_ollama():
    print("\n[3/5] Checking Ollama installation...")
    code, out = run_command(["ollama", "--version"])
    if code != 0:
        print("ERROR: Ollama is not installed or not in PATH.")
        print("Install Ollama from https://ollama.com before continuing.")
        sys.exit(1)
    print(f"Ollama detected: {out}")
    return out


def step_pull_and_test_model():
    print(f"\n[4/5] Pulling and running baseline model ({BASELINE_MODEL})...")
    print("Pulling model weights (this may take a few minutes)...")
    pull_code, pull_out = run_command(["ollama", "pull", BASELINE_MODEL])
    if pull_code != 0:
        print(f"ERROR pulling {BASELINE_MODEL}: {pull_out}")
        sys.exit(1)

    print("Running initial smoke inference...")
    run_code, run_out = run_command(["ollama", "run", BASELINE_MODEL, TEST_PROMPT])
    if run_code != 0:
        print(f"ERROR executing baseline inference: {run_out}")
        sys.exit(1)

    print("\nModel Output:")
    print("-" * 50)
    print(run_out)
    print("-" * 50)
    return run_out


def step_generate_hardware_specs(ollama_version: str, gpu_details: str, baseline_output: str):
    print("\n[5/5] Generating docs/hardware_specs.md...")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    specs_content = f"""# Hardware & Environment Specifications Log
*Generated on: {timestamp}*

## System Information
- **Operating System:** {platform.system()} {platform.release()} ({platform.version()})
- **Architecture:** {platform.machine()}
- **Processor:** {platform.processor()}
- **Python Version:** {platform.python_version()}

## Ollama Runtime
- **Version:** {ollama_version}

## GPU & Driver State (nvidia-smi)
```text
{gpu_details}
```

## Baseline Model Smoke Test
- **Model:** {BASELINE_MODEL}
- **Prompt:** {TEST_PROMPT}

### Output
```
{baseline_output}
```
"""

    output_path = os.path.join("docs", "hardware_specs.md")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(specs_content)
    print(f"Saved: {output_path}")


def main():
    print("=" * 60)
    print("  AI Project Environment Setup")
    print("=" * 60)

    step_create_directories()
    gpu_details = step_check_gpu()
    ollama_version = step_check_ollama()
    baseline_output = step_pull_and_test_model()
    step_generate_hardware_specs(ollama_version, gpu_details, baseline_output)

    print("\nSetup complete! Check docs/hardware_specs.md for your environment report.")


if __name__ == "__main__":
    main()
