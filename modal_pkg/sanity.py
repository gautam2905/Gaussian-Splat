"""GPU sanity check — confirms Modal auth + H100 reachable.

Run: `modal run modal_pkg/sanity.py`
Expected: `{'gpu': 'NVIDIA H100 80GB HBM3, 81559 MiB'}` (or similar).
Cost: ~$0.01 (5 seconds H100).

Note: package is named modal_pkg/ (not modal/) to avoid shadowing the Modal SDK.
      base_image includes .add_local_python_source("modal_pkg") so the package
      is available in containers for cross-module imports.
"""
import modal

from modal_pkg.shared import app, base_image, volume, DATA_PATH


@app.function(
    image=base_image,
    gpu="H100",
    volumes={DATA_PATH: volume},
    timeout=120,
)
def gpu_sanity() -> dict:
    import subprocess

    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"]
    )
    return {"gpu": out.decode().strip()}


@app.local_entrypoint()
def main() -> None:
    print(gpu_sanity.remote())
