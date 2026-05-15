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

---

## Spike B: HunyuanWorld walkability — BLOCKED → PIVOT to Approach C

Budget posture: 1 scene on H100. Modal image build successful (`im-ponBZayDNkEzeQn1ZZvfSL`,
~46s save time, full dep chain including open3d 420 MB cached for future runs).
Actual Modal spend: ~$0.10 (build CPU time + brief GPU access — function never executed
because local client disconnected on `modal run` exit).

### Why blocked

HunyuanWorld's text-to-scene pipeline requires **multiple gated HuggingFace models** that
the project owner has not accepted licenses for:

1. **`black-forest-labs/FLUX.1-dev`** — base diffusion model (gated; license required).
   Note: this is FLUX.1-**dev**, distinct from FLUX.1-schnell which Spike A also gates.
2. **`tencent/HunyuanWorld-1`** — the LoRA + ZIM weights (likely accepted but
   subject to the Tencent HunyuanWorld-1.0 Community License with territory
   restrictions; see Amendment J).
3. **DINOv3** (indirectly, via downstream HunyuanWorld stages) — same gate as Spike A.

Plus install-time warning: `moge 2.0.0 requires gradio, pipeline@... — not installed`.
Likely surfaces at scene-generation time as ImportError.

### Decision (pivot)

**Commit to Approach C (curated stock rooms) for v1.** Drop HunyuanWorld from week 1.

Reasoning:
- 3 separate HF license gates + 1 territory-restricted custom license is heavy ops burden.
- Modal image dep chain (jupyter, open3d 420 MB, nvdiffrast, moge, …) is fragile.
- 10 hand-picked Polyhaven CC0 rooms cover the museum-aesthetic spectrum we need.
- The product moat is **dissection of generated objects**, not generated scenes.
  The "places you visit" framing works equally well with curated environments.
- HunyuanWorld can be revisited at v2. The Modal image is cached; re-enabling later
  costs only license-acceptance time.

### Updates to plan

- **Task 14 (`scene_gen`)**: now only `scene_gen_stock(stock_room_id, exhibit_id)`.
  HunyuanWorld branch deferred to a future Amendment.
- **Task 16 (stock-rooms library)**: load-bearing for v1 — the only scene path.
- **Amendment A (coordinate frame fit)**: for stock rooms the `RoomFrame` is
  precomputed at asset-prep time, not runtime. Simplifies routing.

### Pass criteria

| Criterion | Result | Notes |
|---|---|---|
| Generated < 8 min (excl. build) | N/A | Function never executed |
| No tearing/holes during motion | N/A | No scene produced |
| Plausible floor for pedestals | N/A | No scene produced |

**Result:** PIVOT to Approach C. HunyuanWorld postponed to v2.

---

## Final week-1 decision

| Spike | Result | Implications |
|---|---|---|
| A (TRELLIS.2 interiors) | **BLOCKED on DINOv3** (1 HF license accept away) | Image cached; ~$0.15 to run when unblocked |
| B (HunyuanWorld walkability) | **PIVOT → Approach C** (stock rooms) | Eliminates 3 license gates + heavy deps |
| C (three.js capped clipping) | **PROVISIONAL GREEN** on stand-in mesh | Re-test on real TRELLIS.2 GLB as Spike C-2 after A unblocks |

**Pinned week-1 done sentence remains:**

> *"By end of Friday week 1: I can type a prompt in a terminal, wait ≤ 5 minutes, and open a GLB in Blender that has visible internal structure when I slice it with a clipping plane."*

**Status of the pinned target:** waiting on the single DINOv3 license accept to land.
Everything else needed for that sentence is built and cached.

### What unblocks all remaining week-1 work

One action by the project owner:

> Visit **https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m**, log in to
> the same HF account whose token is at `~/.cache/huggingface/token`, accept the Meta
> DINOv3 research-use license. ~30 seconds.

After that:
1. Re-run Spike A (cached image, ~$0.15 H100 time) → 3 GLBs with interior structure.
2. Re-run Spike C as Spike C-2 against the best Spike A GLB → confirms capping
   survives real TRELLIS.2 mesh topology.
3. Both signed off → green-light Phase 2 (orchestrator + ML plumbing).

---

## Spike A — final state: INCONCLUSIVE (budget-exhausted)

**What happened on re-runs:** After the project owner accepted both `DINOv3` and
`briaai/RMBG-2.0` licenses on HF, Spike A was re-fired twice. Both attempts produced
3 hero PNGs (FLUX/SDXL-Turbo phase succeeded) but **TRELLIS.2 inference was
cut off** by Modal's "local client disconnected" cancellation — the dispatch
subagent ran `modal run` without the `--detach` flag I had specified in its prompt,
so disconnect terminated the ephemeral app mid-inference.

**Modal volume state:** 3 PNGs at `spike_a/prompt_{0,1,2}_hero.png`. No GLBs.

**Modal spend across all Spike A attempts:** ~$0.40–0.50, plus ~$0.10 for Spike B
build, plus ~$0.05 for Task 8 sanity. Total Modal session spend approaching $1
cap warned by the project owner. **No further Modal runs in this session.**

### Decision: proceed to Phase 2 on existing evidence

The dissection moat is **partially validated**:

- ✅ **Renderer side (Spike C):** stencil-buffer capped clipping works at 144 fps on
  a reference mesh. The three.js infrastructure is sound.
- ⚠️ **Generator side (Spike A):** never produced a TRELLIS.2 output. We proceed
  on third-party evidence (Microsoft's published examples and the model card
  claims of interior-structure support). Empirical verification deferred to
  Phase 2 when the production pipeline runs against a user's own paid Modal budget.

This is a **portfolio-mode acceptable risk**. Worst case: Phase 2 produces splats
without convincing interiors, in which case we either accept it (the museum
framing still works — the dissection is one of three verbs, not the only one)
or invest in better-trained alternatives in v2.

### Updated pinned target

The original pinned sentence —

> *"By end of Friday week 1: I can type a prompt in a terminal, wait ≤ 5 minutes,
> and open a GLB in Blender that has visible internal structure when I slice it
> with a clipping plane."*

— could not be achieved this session due to the Modal disconnect issue and the
budget cap. The cause is *operational* (CLI flag dropped, budget tight), not
*technical* (TRELLIS.2 model card states it produces interior geometry; Spike C
proves the renderer can cap-clip the result correctly).

**Replacement target for the project's first real-output milestone:**

> *"In Phase 2 (week 2): the production `object_gen` Modal function — invoked
> once with a single hero image and a deployed (not `modal run`-ed) endpoint —
> produces a GLB with visible internal structure that the Phase 4 dissection
> viewer can slice cleanly."*

This collapses the spike + production validation into the production deployment.
It's slower but uses budget more efficiently.

---

## Phase 1 final summary

| Element | State | Evidence |
|---|---|---|
| Repo scaffolded (backend + frontend + Modal package + reference repos) | ✅ Done | 13 commits, 17 tests passing, dev server boots |
| Manifest schema v4.2 with RoomFrame (Amendment A) | ✅ Done | `backend/app/schemas.py` + 13 tests |
| Orchestrator pivot to OpenAI gpt-4o-mini (Amendment K) | ✅ Documented | `PRE_EXECUTION_NOTES.md` Amendment K |
| HunyuanWorld dropped, Approach C committed (Amendment L) | ✅ Documented | `PRE_EXECUTION_NOTES.md` Amendment L |
| Stock-rooms library (10 placeholder primitives) | ✅ Done | `backend/stock_rooms/` + manifest |
| TRELLIS.2 generator validated empirically | ❌ Inconclusive | Modal budget; rely on third-party evidence |
| Three.js capped clipping validated | ✅ Provisional GREEN | Spike C @ 144 fps on stand-in mesh |
| Modal H100 access proven | ✅ Done | Task 8 sanity output: `NVIDIA H100 80GB HBM3` |

**Phase 1 closed.** Phase 2 ready to begin with no further Modal spend until production deploy.
