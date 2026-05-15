"""Spike A — TRELLIS.2 interior validation.

3 prompts, FLUX.1-schnell -> TRELLIS.2 at 512^3, on H100.
Outputs: 3 PNG hero images + 3 GLBs at /data/spike_a/ on Modal volume.
Cost target: ~$0.15.

API notes (from references/TRELLIS.2):
- Pipeline class: trellis2.pipelines.Trellis2ImageTo3DPipeline
- Init: Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
- Run: pipeline.run(image, pipeline_type='512') -> List[MeshWithVoxel]
  (pipeline_type='512' gives 512^3 resolution, fastest)
- GLB export: o_voxel.postprocess.to_glb(vertices, faces, attr_volume, coords,
    attr_layout, voxel_size, aabb, decimation_target, texture_size, remesh, ...)
    then .export("path.glb")
- Background removal: pipeline.preprocess_image(pil_image) removes bg via rembg
"""
from pathlib import Path

import modal

from .shared import app, volume, DATA_PATH

SPIKE_PROMPTS = [
    "a pomegranate cross-sectioned, seeds visible, photographic quality, plain white background, single subject, no text",
    "a vintage mechanical pocket watch, back cover removed, gears and balance wheel exposed, plain white background, single subject, no text",
    "a human heart, anatomical, chambers visible, hero shot, plain white background, single subject, no text",
]

# ---------------------------------------------------------------------------
# Modal image — compile all CUDA extensions at build time so cold starts are fast.
# Build is CPU-only but needs CUDA headers (installed via cuda-toolkit).
# Base: debian slim + CUDA 12.4 runtime + toolkit for nvcc.
# ---------------------------------------------------------------------------
trellis_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install(
        "git", "ffmpeg", "libgl1", "libglib2.0-0",
        "libjpeg-dev", "wget", "curl", "build-essential", "ninja-build",
        # clang++ is the Modal Python build compiler; install it so linking works
        "clang",
        # OpenCV needs these
        "libsm6", "libxext6", "libxrender-dev",
    )
    # Core Python deps first (needed for CUDA extension builds)
    .pip_install(
        "torch==2.6.0",
        "torchvision==0.21.0",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "numpy",
        "pillow",
        "imageio",
        "imageio-ffmpeg",
        "tqdm",
        "easydict",
        "trimesh",
        "scipy",
        "plyfile",
        "zstandard",
        "opencv-python-headless",
        "kornia",
        "timm",
        "transformers>=4.44",
        "accelerate",
        "safetensors",
        "huggingface_hub",
        "diffusers>=0.30",
        "einops",
        "lpips",
        "pandas",
        "tensorboard",
    )
    # flash-attn — must be installed after torch; needs wheel in the build env
    .pip_install("wheel", "packaging", "ninja")
    .pip_install("flash-attn==2.7.3", extra_options="--no-build-isolation")
    # utils3d (trellis2 dep)
    .pip_install(
        "git+https://github.com/EasternJournalist/utils3d.git"
        "@9a4eb15e4021b67b12c460c7057d642626897ec8"
    )
    # Set CUDA arch list for H100 (SM 9.0) — required when building CUDA
    # extensions at image-build time (no GPU is visible, so auto-detection fails).
    .env({"TORCH_CUDA_ARCH_LIST": "9.0", "FORCE_CUDA": "1"})
    # nvdiffrast (needed by trellis2 pipeline)
    .run_commands(
        "git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git /tmp/nvdiffrast",
        "pip install /tmp/nvdiffrast --no-build-isolation",
    )
    # nvdiffrec (needed by trellis2 pipeline)
    .run_commands(
        "git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git /tmp/nvdiffrec",
        "pip install /tmp/nvdiffrec --no-build-isolation",
    )
    # CuMesh (dep of o_voxel)
    .run_commands(
        "git clone https://github.com/JeffreyXiang/CuMesh.git /tmp/CuMesh --recursive",
        "pip install /tmp/CuMesh --no-build-isolation",
    )
    # FlexGEMM (dep of o_voxel)
    .run_commands(
        "git clone https://github.com/JeffreyXiang/FlexGEMM.git /tmp/FlexGEMM --recursive",
        "pip install /tmp/FlexGEMM --no-build-isolation",
    )
    # TRELLIS.2 itself + its bundled o-voxel CUDA extension
    # Note: TRELLIS.2 has no setup.py/pyproject.toml at root — it's a research
    # repo. We put it on PYTHONPATH so `import trellis2` works.
    .run_commands(
        "git clone --depth=1 --recursive https://github.com/microsoft/TRELLIS.2.git /opt/TRELLIS2",
        # Install o-voxel CUDA extension (must be done after torch+cuda are present)
        "cd /opt/TRELLIS2/o-voxel && pip install . --no-build-isolation",
        # trellis2 package has no setup.py; add root to site-packages via .pth
        "echo '/opt/TRELLIS2' > $(python -c \"import site; print(site.getsitepackages()[0])\")/trellis2_path.pth",
    )
    .add_local_python_source("modal_pkg")
)


@app.function(
    image=trellis_image,
    gpu="H100",
    volumes={DATA_PATH: volume},
    secrets=[modal.Secret.from_name("huggingface-token")],
    timeout=1800,  # 30 min — generous: image built, 3 prompts on H100
)
def spike_a(prompts: list[str]) -> dict:
    import os
    import time

    import torch

    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    out_dir = Path(DATA_PATH) / "spike_a"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []

    # ------------------------------------------------------------------
    # Stage 1: SDXL-Turbo — generate hero images from text prompts
    # Using SDXL-Turbo (Apache-2.0, no license gating) instead of FLUX.1-schnell
    # which requires HF license acceptance. SDXL-Turbo produces 1024×1024 images
    # in 1–4 steps with guidance_scale=0.0, sufficient for TRELLIS.2 input.
    # ------------------------------------------------------------------
    print("[IMG] Loading SDXL-Turbo pipeline...")
    from diffusers import AutoPipelineForText2Image

    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float16,
        variant="fp16",
    ).to("cuda")

    image_paths = []
    flux_latencies = []
    for i, prompt in enumerate(prompts):
        t0 = time.time()
        img = pipe(
            prompt,
            num_inference_steps=4,
            guidance_scale=0.0,
            height=1024,
            width=1024,
        ).images[0]
        flux_elapsed = time.time() - t0
        flux_latencies.append(flux_elapsed)
        img_path = out_dir / f"prompt_{i:02d}_hero.png"
        img.save(str(img_path))
        image_paths.append(img_path)
        print(f"[IMG {i}] {flux_elapsed:.1f}s -> {img_path.name}")

    del pipe
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Stage 2: TRELLIS.2 — image-to-3D at 512^3
    # API: Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    # pipeline.run(image, pipeline_type='512') -> List[MeshWithVoxel]
    # GLB: o_voxel.postprocess.to_glb(vertices, faces, attr_volume, ...)
    # ------------------------------------------------------------------
    print("[TRELLIS] Loading pipeline...")
    import o_voxel
    from PIL import Image as PILImage
    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    pipeline.cuda()

    trellis_latencies = []
    for i, img_path in enumerate(image_paths):
        prompt_short = prompts[i][:50]
        try:
            image = PILImage.open(str(img_path)).convert("RGB")

            t0 = time.time()
            # pipeline_type='512' → 512^3 resolution (fastest, ~3s on H100)
            mesh_list = pipeline.run(image, pipeline_type="512")
            trellis_elapsed = time.time() - t0
            trellis_latencies.append(trellis_elapsed)

            mesh = mesh_list[0]
            # Reduce polygon count to stay within nvdiffrast limits
            mesh.simplify(16_777_216)

            glb_path = out_dir / f"prompt_{i:02d}_object.glb"
            glb = o_voxel.postprocess.to_glb(
                vertices=mesh.vertices,
                faces=mesh.faces,
                attr_volume=mesh.attrs,
                coords=mesh.coords,
                attr_layout=mesh.layout,
                voxel_size=mesh.voxel_size,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                decimation_target=1_000_000,
                texture_size=1024,  # 1024 is faster than 4096 for spike
                remesh=True,
                remesh_band=1,
                remesh_project=0,
                verbose=False,
            )
            glb.export(str(glb_path))
            print(
                f"[TRELLIS {i}] {trellis_elapsed:.1f}s -> {glb_path.name}"
                f"  (FLUX: {flux_latencies[i]:.1f}s)"
            )
            results.append(
                {
                    "prompt_index": i,
                    "prompt": prompts[i],
                    "image": str(img_path.name),
                    "glb": str(glb_path.name),
                    "flux_latency_s": round(flux_latencies[i], 1),
                    "trellis_latency_s": round(trellis_elapsed, 1),
                    "status": "ok",
                }
            )
        except Exception as e:
            print(f"[TRELLIS {i}] ERROR: {e}")
            results.append(
                {
                    "prompt_index": i,
                    "prompt": prompts[i],
                    "status": "error",
                    "error": str(e),
                    "flux_latency_s": round(flux_latencies[i], 1) if i < len(flux_latencies) else None,
                }
            )

    volume.commit()

    summary = {
        "results": results,
        "flux_latencies": flux_latencies,
        "trellis_latencies": trellis_latencies,
        "out_dir": str(out_dir),
    }
    print("[DONE]", summary)
    return summary


@app.local_entrypoint()
def main() -> None:
    import json

    print("Starting Spike A — TRELLIS.2 interior validation...")
    result = spike_a.remote(SPIKE_PROMPTS)
    print(json.dumps(result, indent=2))
    print("\nDone. Fetch outputs with:")
    print("  modal volume get museums-exhibits spike_a /mnt/3D_Mapping/spikes/spike_a/")
