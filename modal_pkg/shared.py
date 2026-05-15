"""Shared Modal app + image defs for all ML functions in this project."""
import modal

app = modal.App("museums-ml")

base_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg", "libgl1")
    .pip_install("torch==2.4.0", "torchvision==0.19.0", "numpy", "pillow", "trimesh")
    # Include the local modal_pkg package in every container so cross-module
    # imports (from modal_pkg.shared import ...) work remotely.
    .add_local_python_source("modal_pkg")
)

volume = modal.Volume.from_name("museums-exhibits", create_if_missing=True)
DATA_PATH = "/data"
