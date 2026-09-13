"""
Inspect and log computer and environment specifications
strictly conforming to Page 2 & Page 5 of the Internship Project Plan:
- Operating system and version
- GPU model, driver version, and available VRAM
- CPU model and system RAM
- Ollama version
- Exact model names, tags, and download sizes
- Model storage drive and available disk space
"""

import os
import platform
import shutil
import subprocess
from datetime import datetime

OUTPUT_PATH = os.path.join("docs", "hardware_specs.md")


def run_cmd(cmd: list[str]) -> str:
    try:
        res = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        return res.stdout.strip()
    except Exception as e:
        return f"Error: {e}"


def get_gpu_info():
    # Try nvidia-smi with specific query format
    query_cmd = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total,memory.free",
        "--format=csv,noheader,nounits"
    ]
    raw_query = run_cmd(query_cmd)
    if not raw_query.startswith("Error"):
        lines = [line.strip() for line in raw_query.splitlines() if line.strip()]
        if lines:
            parts = [p.strip() for p in lines[0].split(",")]
            if len(parts) >= 3:
                name = parts[0]
                driver = parts[1]
                vram_mb = parts[2]
                vram_gb = round(float(vram_mb) / 1024, 2)
                return {
                    "gpu_model": name,
                    "driver_version": driver,
                    "vram": f"{vram_gb} GB ({vram_mb} MB)",
                    "raw": run_cmd(["nvidia-smi"])
                }
    return {
        "gpu_model": "N/A or Not Detected",
        "driver_version": "N/A",
        "vram": "N/A",
        "raw": run_cmd(["nvidia-smi"])
    }


def get_cpu_and_ram():
    cpu_name = platform.processor()
    # Try PowerShell for friendly CPU name
    ps_cpu = run_cmd(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"])
    if ps_cpu and not ps_cpu.startswith("Error"):
        cpu_name = ps_cpu.strip()

    # Total physical memory
    ps_ram = run_cmd(["powershell", "-NoProfile", "-Command", "[Math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 2)"])
    ram_gb = f"{ps_ram.strip()} GB" if ps_ram and not ps_ram.startswith("Error") else "Unknown"

    return cpu_name, ram_gb


def get_ollama_info():
    version_out = run_cmd(["ollama", "--version"])
    version = version_out.replace("ollama version is", "").strip() if version_out else "Not detected"

    # Ollama storage path
    # Default on Windows is %USERPROFILE%\.ollama\models
    user_home = os.path.expanduser("~")
    custom_models_path = os.environ.get("OLLAMA_MODELS")
    storage_path = custom_models_path if custom_models_path else os.path.join(user_home, ".ollama", "models")
    
    drive_letter = os.path.splitdrive(os.path.abspath(storage_path))[0] or "C:"
    
    # Disk space
    try:
        total, used, free = shutil.disk_usage(drive_letter + "\\")
        free_gb = round(free / (1024 ** 3), 2)
        total_gb = round(total / (1024 ** 3), 2)
        disk_info = f"{drive_letter} ({free_gb} GB free of {total_gb} GB)"
    except Exception as e:
        disk_info = f"Error querying disk: {e}"

    return version, storage_path, disk_info


def generate_specs():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    gpu_info = get_gpu_info()
    cpu_name, ram_gb = get_cpu_and_ram()
    ollama_version, storage_path, disk_info = get_ollama_info()

    doc_content = f"""# Hardware & Software Specifications Sheet
*Generated on: {timestamp}*  
*Conforms to: Student Internship Project Plan (Page 2 & 5: Computer information to record)*

---

## 1. Operating System
- **OS Name & Version:** {platform.system()} {platform.release()} (Build {platform.version()})
- **Architecture:** {platform.machine()} (64-bit)

## 2. GPU & VRAM State
- **GPU Model:** {gpu_info["gpu_model"]}
- **Driver Version:** {gpu_info["driver_version"]}
- **Available VRAM:** {gpu_info["vram"]}

```text
{gpu_info["raw"]}
```

## 3. CPU & System Memory
- **Processor:** {cpu_name}
- **Total System RAM:** {ram_gb}

## 4. Ollama Runtime
- **Ollama Version:** {ollama_version}

## 5. Model Storage & Disk Space
- **Storage Drive & Path:** `{storage_path}`
- **Available Disk Space:** {disk_info}

## 6. Selected Models & Download Sizes (Page 2 Target)
| Package (Tag) | Alias | Parameters | Size (Disk) | Purpose in Benchmark | Official Link |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `phi4-mini-reasoning` | `PHI` | 3.84B | ~3.2 GB | Lightweight reasoning baseline for constrained hardware | [Ollama Library](https://ollama.com/library/phi4-mini-reasoning:latest) |
| `deepseek-r1:7b` | `DSR1` | 7.62B | ~4.7 GB | Compact distilled reasoning model | [Ollama Library](https://ollama.com/library/deepseek-r1:7b) |
| `qwen3:8b` | `QWEN` | 8.19B | ~5.2 GB | Larger general reasoning comparison model | [Ollama Library](https://ollama.com/library/qwen3:8b) |
"""

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(doc_content)

    print(f"Hardware specs successfully generated at {OUTPUT_PATH}")


if __name__ == "__main__":
    generate_specs()
