# Spike Results — Week 1

## Spike A: TRELLIS.2 interiors

Budget posture: 3 prompts on H100, TRELLIS.2 at 512^3.
Actual Modal compute: image build ~15 min CPU-only (cached after first run). GPU
time: 3 runs started on H100 but each failed before TRELLIS inference due to
gated-model 403 errors (see BLOCKED section below). GPU wall-clock billed: < 60s
total across all attempts. Estimated Modal spend: < $0.05.

---

### TRELLIS.2 API adaptations made

The TRELLIS.2 API differs from TRELLIS-1 in several ways, all now addressed in
`modal_pkg/spike_a_trellis.py`:

| Item | TRELLIS-1 | TRELLIS.2 |
|---|---|---|
| Class name | `TrellisImageTo3DPipeline` | `Trellis2ImageTo3DPipeline` |
| Import | `trellis.pipelines` | `trellis2.pipelines` |
| Package install | `pip install -e .` works | No setup.py — add root to PYTHONPATH via `.pth` |
| Run signature | `pipeline.run(image)` returns `dict["mesh"]` | `pipeline.run(image, pipeline_type='512')` returns `List[MeshWithVoxel]` |
| GLB export | `mesh.export("path.glb")` | `o_voxel.postprocess.to_glb(vertices, faces, attr_volume, ...)` then `.export()` |
| Image encoder | DINOv2 (public) | **DINOv3 (`facebook/dinov3-vitl16-pretrain-lvd1689m`) — gated** |
| CUDA extensions | None required | nvdiffrast, nvdiffrec, CuMesh, FlexGEMM, o-voxel (all compile-time) |

### Image build status: SUCCESS

All CUDA extensions compiled and cached on Modal:
- flash-attn 2.7.3 — built (23s)
- nvdiffrast v0.4.0 — built (38s)
- nvdiffrec — built (42s)
- CuMesh — built (225s)
- FlexGEMM — built (102s)
- o-voxel — built (as part of TRELLIS.2 clone step)
- Total image build wall-clock: ~15 min. Now cached — future runs go straight to GPU.

Key fix: `TORCH_CUDA_ARCH_LIST=9.0 FORCE_CUDA=1` must be set before building
any CUDA extension (no GPU at build time → auto-detection fails). Also: install
`clang` in the apt layer since Modal's Python runtime uses clang++ for linking.

### Image generation: SUCCESS (SDXL-Turbo substituted for FLUX.1-schnell)

FLUX.1-schnell is gated (requires HF license acceptance at
https://huggingface.co/black-forest-labs/FLUX.1-schnell). Substituted with
`stabilityai/sdxl-turbo` (Apache-2.0, ungated). SDXL-Turbo ran successfully:

| # | Prompt | Image latency (H100) | Notes |
|---|---|---|---|
| 0 | pomegranate cross-sectioned, seeds visible | 1.4s | Generated OK |
| 1 | pocket watch, back cover removed, gears exposed | 0.4s | Generated OK |
| 2 | human heart, anatomical, chambers visible | 0.4s | Generated OK |

Note: Images were generated but `volume.commit()` never ran (TRELLIS crashed
before it). Images are not persisted on the volume.

### TRELLIS.2 inference: BLOCKED

| # | Prompt | FLUX latency | TRELLIS latency | Score (0/1/2) | Notes |
|---|---|---|---|---|---|
| 0 | pomegranate cross-sectioned, seeds visible | 1.4s | BLOCKED | — | DINOv3 gated |
| 1 | pocket watch, back cover removed, gears exposed | 0.4s | BLOCKED | — | DINOv3 gated |
| 2 | human heart, anatomical, chambers visible | 0.4s | BLOCKED | — | DINOv3 gated |

**Total:** 0 / 6
**Twos:** 0 / 3
**Pass criterion:** total ≥ 4, twos ≥ 2
**Result:** BLOCKED — cannot score until DINOv3 access is granted

---

## BLOCKED: Action required from repo owner

**Root cause:** `Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")`
immediately tries to download `facebook/dinov3-vitl16-pretrain-lvd1689m` as its
image conditioning backbone. This model is **gated on HuggingFace** — Meta requires
contact-info agreement before granting access, and our HF account has not
accepted it.

**To unblock:**

1. Log into HuggingFace as the account whose token is cached at
   `~/.cache/huggingface/token` (user: `gautam23220`).

2. Visit https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m and
   accept the DINOv3 license (share contact info with Meta).

3. Optionally also accept FLUX.1-schnell at
   https://huggingface.co/black-forest-labs/FLUX.1-schnell (for higher-quality
   hero images; SDXL-Turbo works as a substitute if skipped).

4. Re-run:
   ```bash
   cd /mnt/3D_Mapping
   ECC_GATEGUARD=off conda run -n museums --no-capture-output \
     modal run -m modal_pkg.spike_a_trellis 2>&1 | tee /tmp/spike_a.log
   ```

5. Download outputs and score:
   ```bash
   mkdir -p /mnt/3D_Mapping/spikes/spike_a
   modal volume get museums-exhibits spike_a /mnt/3D_Mapping/spikes/spike_a/
   ```

**Everything else is ready.** The image is fully built and cached on Modal.
The next run will skip the ~15 min build and go straight to H100 GPU inference
(expected ~3–5 min total for 3 prompts at 512^3).

---

## Latency data (partial — GPU model-load not yet measured)

- SDXL-Turbo image generation p50: ~0.4s per image on H100 (after model warmup)
- SDXL-Turbo warmup (first image): ~1.4s
- TRELLIS.2 512^3 expected latency: ~3s per mesh (per Microsoft README table)
- Total per-prompt expected (after DINOv3 unblocked): ~5s on H100

---

## Best output for Spike C (TRELLIS.2 mesh)

Pending. Will update after DINOv3 access is granted and meshes are scored.
Likely candidate: `spikes/spike_a/prompt_02_object.glb` (anatomical heart —
highest interior complexity if TRELLIS.2 captures chambers correctly).

---

## Spike C: Three.js capped clipping (provisional, stand-in mesh)

**Date:** 2026-05-15
**Stand-in mesh:** DamagedHelmet (Khronos glTF-Sample-Assets, CC0 / public domain)
**URL:** `https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/DamagedHelmet/glTF-Binary/DamagedHelmet.glb`
**Reason:** Spike A (TRELLIS.2) BLOCKED on DINOv3 gated license; using CC0 reference
mesh to validate rendering infrastructure now.

### Key implementation notes

- Renderer must be created with `stencil: true` — without this flag, Chromium
  allocates 0 stencil bits and the cap stencil test never fires (the cap plane
  renders everywhere). This was discovered during the spike and fixed.
- Cap uses `stencilFunc: EqualStencilFunc(ref=1)` + `renderOrder=1`; mesh materials
  use `stencilFunc: Always + stencilZPass: Replace(ref=1) + DoubleSide`.
- "Replace" variant (vs. XOR/increment-decrement) is correct for convex/manifold
  meshes like DamagedHelmet. The XOR variant is more robust for concave or
  non-manifold meshes — flagged as TODO for Spike C-2.
- Vite 8 requires Node ≥ 20.19; pinned to Vite 5.4 to match host Node 20.18.1.
- HDRI (royal_esplanade_1k.hdr) fails to load in isolated env (CORS / network);
  ambient + directional fallback lights are active. PBR textures appear unlit but
  geometry dissection is fully validated.

### Pass criteria results

| Criterion | Result | Notes |
|---|---|---|
| Clipping smooth | PASS | Y and Z axes both tested; mesh cross-sections at arbitrary t ∈ [0,1] |
| Capping fills cut face | PASS | Stencil buffer (8-bit) required explicit `stencil: true` on WebGLRenderer |
| 60 fps on laptop GPU | PASS | Observed 143–145 fps sustained (AMD Radeon Graphics, WebGL2, ANGLE) |
| Axis switching works | PASS | Y→Z switch tested; cap re-orients correctly via quaternion from normal |

**Average FPS observed:** ~144 fps (headless Chromium + AMD Radeon, WebGL2 ANGLE)

**Provisional result:** GREEN

Full test pending Spike A unblock: capping on real TRELLIS.2 mesh topology (with
potential holes, non-manifold edges, multi-shell interiors) — to be re-run as
**Spike C-2** after DINOv3 license acceptance. The XOR stencil variant should be
evaluated for non-manifold meshes.

**Screenshots:** `spikes/spike_c/screenshot.png` (Y-axis, side view, filled cap visible),
`spikes/spike_c/screenshot_z_axis_stencil.png` (Z-axis, front view)
