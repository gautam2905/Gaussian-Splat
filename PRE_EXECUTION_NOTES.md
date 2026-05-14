# Pre-Execution Notes — addenda to the spec and plan

> **Status:** binding amendments to `docs/superpowers/specs/2026-05-14-interactive-museums-design.md` and `docs/superpowers/plans/2026-05-14-interactive-museums-plan.md`. Subagents executing the plan must read this file before starting their task.

> **Pinned week-1 done:** *"By end of Friday week 1: I can type a prompt in a terminal, wait ≤ 5 minutes, and open a GLB in Blender that has visible internal structure when I slice it with a clipping plane."* If polishing anything else by Friday week 1, drift has occurred — return to spike.

---

## Amendment A — Coordinate frame fitting between scene_gen and manifest

**Problem.** HunyuanWorld outputs a scene in its own arbitrary coordinate frame: floor at some y, walls at unknown positions, forward direction undefined. Claude cannot know any of this. If Claude emits world-coordinate pedestal positions, they will float in midair, sink into the floor, or clip into walls.

**Fix.**

1. **Manifest schema change.** Pedestal `position` and spline `control_points` are emitted by Claude in **normalized room-local coordinates `[0, 1]`** (a unit box), not in metres. The schema field comment must say so. The orchestrator system prompt must instruct: "emit positions and spline control points in normalized 0–1 unit-box coordinates; the backend will transform these into the scene's actual coordinate frame."

2. **New backend module:** `backend/app/scene_geometry/frame_fit.py` with:
   - `fit_room_frame(glb_path: Path) -> RoomFrame` — loads the GLB, estimates floor plane (PCA + RANSAC on lowest 10% of vertices), finds principal horizontal axes (PCA on x/z projection), returns `RoomFrame(origin, basis_x, basis_y, basis_z, extent)`.
   - `transform_manifest_to_world(manifest: Manifest, frame: RoomFrame) -> Manifest` — applies the frame to every pedestal `position` and spline `control_points` entry, producing a new manifest with world-coordinate values.

3. **Routing change.** `routing.run_pipeline` between scene_gen and objects_gen:
   ```
   manifest = orchestrate(prompt)
   manifest.scene.scene_url = call_scene_*(...)
   frame = fit_room_frame(scene_glb_path)
   manifest = transform_manifest_to_world(manifest, frame)
   manifest.scene.room_frame = frame.dict()  # store for debug
   write_manifest(manifest)
   # then proceed to objects_gen
   ```

4. **Stock-room frames** are pre-computed and stored alongside each GLB: `backend/stock_rooms/rooms/<id>.glb` + `backend/stock_rooms/rooms/<id>.frame.json`. The fallback path reads the precomputed frame instead of re-fitting.

5. **Manifest schema additions** (additive, backward-compatible — bump `schema_version` to `"v4.2"`):
   ```python
   class RoomFrame(BaseModel):
       origin: list[float] = Field(min_length=3, max_length=3)
       basis_x: list[float] = Field(min_length=3, max_length=3)
       basis_y: list[float] = Field(min_length=3, max_length=3)
       basis_z: list[float] = Field(min_length=3, max_length=3)
       extent: list[float] = Field(min_length=3, max_length=3)  # room size in metres
   ```
   Added to `Scene` as `room_frame: RoomFrame | None = None` (filled by backend after frame fit).

6. **Validation.** `fit_room_frame` raises `RoomFrameFitFailed` if no consistent floor plane found (variance > threshold). This triggers scene-fallback to a stock room (which has a known frame).

**Affected plan tasks:** Task 3 (schema), Task 13 (orchestrator prompt), Task 14 (scene_gen), Task 16 (stock-room frames precomputed), Task 17 (routing pipeline), new Task 14.5 inserted for `frame_fit`.

---

## Amendment B — Explicit camera state machine

**Problem.** Three modes (rail-cruise, fly-to-pedestal, free mouse-look) sharing a single camera object produces transition jank.

**Fix.**

`frontend/src/viewer/locomotion/CameraStateMachine.ts` replaces ad-hoc `ModeManager.ts`. Explicit states:

```typescript
type CameraState =
  | { kind: "ON_SPLINE"; t: number }
  | { kind: "FLYING_IN"; from: Pose; to: Pose; t0: number; duration: number; target: ObjectFocus }
  | { kind: "AT_PEDESTAL"; target: ObjectFocus; basePose: Pose }
  | { kind: "FLYING_OUT"; from: Pose; to: Pose; t0: number; duration: number; toSplineT: number };

interface Pose { position: Vec3; quaternion: Quat; }
interface ObjectFocus { objectId: string; pedestalId: string; mesh: THREE.Object3D; }
```

Transitions:

- `advance()` / `retreat()`: only valid in `ON_SPLINE`. Tweens `splineT` (within the same state — no state change).
- `enterFocus(target)`: `ON_SPLINE → FLYING_IN`. Reaches `AT_PEDESTAL` at end of tween.
- `exitFocus()`: `AT_PEDESTAL → FLYING_OUT`. Reaches `ON_SPLINE` at end of tween.
- Pedestal click while `FLYING_IN` or `FLYING_OUT`: ignored.
- Pedestal click while `ON_SPLINE` mid-rail-tween: rail tween interrupted, transition to `FLYING_IN` immediately.

**Mouse-look** is a `(yaw_delta, pitch_delta)` pair stored separately in `LocomotionStore`. Each frame, the FSM computes a `basePose` from the state and `effectivePose = basePose.compose(mouseLookDelta)`. The camera quaternion is **always derived**, never owned by mouse-look directly. This prevents pose drift across transitions.

**Affected plan tasks:** Tasks 20 (RailController — produces basePose for ON_SPLINE state), 21 (MouseLook — emits delta only, never writes camera), 23 (FocusController + ModeManager renamed to CameraStateMachine).

---

## Amendment C — Latency benchmark (new Task 9.5)

Insert immediately after Spike A passes, before Spike B.

**New task:**

### Task 9.5 — End-to-end latency benchmark

**Files:** `modal/benchmark.py`, `BENCHMARK_RESULT.md`.

- [ ] **Step 1: Pick 10 representative prompts** (mix of simple objects, complex scenes, anatomy, mechanical, food, abstract).
- [ ] **Step 2: Write `modal/benchmark.py`** that, for each prompt, calls orchestrator → scene_gen → 3-object parallel object_gen end-to-end and records per-stage timing.
- [ ] **Step 3: Run on Modal H100, 10 prompts.**
- [ ] **Step 4: Compute p50, p90, p95, p99** of total wall-clock.
- [ ] **Step 5: Write `BENCHMARK_RESULT.md`** with the distribution + the **UX directive**:
  - p95 ≤ 3 min → prompt box is primary CTA on landing.
  - 3 min < p95 ≤ 6 min → curated gallery is primary, prompt box secondary with "takes a few minutes" copy.
  - p95 > 6 min → curated gallery is the only landing CTA; prompt box on a separate page with "leave us your email" affordance (email is a v2 feature — for v1, just honest copy "this might take up to N minutes").
- [ ] **Step 6: Commit.**

**Affected plan tasks:** Tasks 29 (landing page) and 30 (generate page) inherit the UX directive from this benchmark.

---

## Amendment D — Per-object capping fallback (ghost shader)

**Problem.** TRELLIS.2 sometimes outputs meshes with non-manifold geometry, internal holes, or thin walls that defeat stencil-buffer capping (cap renders with holes, looks worse than no cap).

**Fix.**

`CappedClipMaterial.attach()` becomes a try-once-detect-then-decide:

1. Attach the stencil cap as designed.
2. Render one frame to an offscreen target.
3. Sample the cap area: if > 10% of expected cap pixels are background (not stencil-1), capping failed.
4. Fall back to `GhostClipMaterial`: the clipped half (the part the plane is "looking at") renders with `material.transparent = true; material.opacity = 0.25` — interior content remains visible through a translucent shell.

`GhostClipMaterial` is a sibling class that simply mutates the focused-object materials when capping fails. No stencil work.

**Spike C amendment:** Spike C runs against the **best-scoring GLB from Spike A** (not a Sketchfab reference). Pass criterion includes: at least 2 of the top-3 Spike A objects must cap cleanly. If only 0–1 cap cleanly, capping ships disabled by default with the ghost-shader fallback as the v1 mode.

**Affected plan tasks:** Task 11 (Spike C uses Spike A output), Task 25 (CappedClipMaterial gets the detection + ghost fallback), Task 27 (DissectionPanel gets no UI change — the fallback is transparent to the user).

---

## Amendment E — Output-level NSFW gate

**Problem.** Prompt-injection bypasses Claude's refusal within hours of launch.

**Fix.**

New Modal function `modal/safety_check.py`:

```python
"""CLIP-based NSFW classifier — second-gate output check."""
import modal
from .shared import app, base_image

safety_image = base_image.pip_install(
    "transformers", "torch", "safetensors", "huggingface_hub"
)

@app.function(image=safety_image, gpu="T4", timeout=30)
def is_safe_image(image_url: str) -> dict:
    """Return {'safe': bool, 'score': float, 'category': str | None}.

    Uses a CLIP-based NSFW classifier (Falconsai/nsfw_image_detection or similar).
    Threshold: score > 0.5 -> unsafe.
    """
    import torch
    from transformers import AutoModelForImageClassification, ViTImageProcessor
    from PIL import Image
    import requests
    from io import BytesIO

    model_name = "Falconsai/nsfw_image_detection"
    model = AutoModelForImageClassification.from_pretrained(model_name)
    processor = ViTImageProcessor.from_pretrained(model_name)
    img = Image.open(BytesIO(requests.get(image_url).content))

    with torch.no_grad():
        inputs = processor(images=img, return_tensors="pt")
        outputs = model(**inputs)
        probs = outputs.logits.softmax(-1)[0]
    unsafe_score = float(probs[1])  # label 1 is "nsfw" in this classifier
    return {
        "safe": unsafe_score < 0.5,
        "score": unsafe_score,
        "category": "nsfw" if unsafe_score >= 0.5 else None,
    }
```

**Pipeline insertion.** In `modal/object_gen.py`:

```
flux generates image
check = is_safe_image.remote(image_url)
if not check["safe"]:
    raise UnsafeOutputError(category=check["category"])
trellis runs on image
```

When `UnsafeOutputError` propagates up, the orchestrator re-runs with a `force_safe_prompt=true` flag that adds a stricter system prompt prefix. If three objects in one job trip the filter, the whole exhibit fails with a refusal.

Cost: T4 inference ~$0.0005/s × 0.2s = $0.0001/object. Negligible.

**Affected plan tasks:** Task 15 (object_gen pipeline gets the safety check), new module `modal/safety_check.py`.

---

## Amendment F — Refusal categorization

`OrchestratorRefused` now carries `category` in addition to `reason`:

```python
class OrchestratorRefused(Exception):
    def __init__(self, reason: str, category: str = "other"):
        super().__init__(reason)
        self.reason = reason
        self.category = category  # nsfw | real_person | gore | other
```

System prompt updated:

```
If refusing, return:
{"refused": true, "category": "nsfw" | "real_person" | "gore" | "other", "reason": "<short>"}
```

Backend logs `{"prompt_hash": "<sha256-hex>", "category": "...", "timestamp": "<ISO-8601 UTC>"}` to a flat append-only file `data/refusal_log.jsonl`. User sees category-tailored copy:

- `nsfw` → "Adult content isn't supported in this demo."
- `real_person` → "We can't create exhibits about specific real people."
- `gore` → "Graphic content isn't supported."
- `other` → the raw `reason` from Claude.

**Affected plan tasks:** Task 13 (orchestrator), Task 30 (generate page error display).

---

## Amendment G — OG card as stylized poster

OG endpoint renders a **museum-poster-style composition**, not a screenshot:

- 1200×630 PNG
- Dark gradient background (top-left `#0a0a0a` to bottom-right `#1f1610`, warm tint)
- Serif title (Cormorant or fall-back to DejaVuSerif), 84 pt, top-left
- Description below in sans, 28 pt, 1.5 line-height, 60% opacity
- Bottom-right: one rendered thumbnail of the highest-`dissectable` object on a transparent background (PIL paste with alpha; thumbnail pre-rendered when the object is generated and cached at `exhibits/{id}/objects/{object_id}.thumb.png`)
- Bottom-left: small wordmark "Interactive AI Museums" + a thin brass-colored horizontal rule

**Object thumbnail pre-rendering.** When `object_gen` completes, render a 400×400 PNG of the GLB on transparent background (trimesh + pyrender headless, or a `gltf-pipeline` CLI call). Cached for OG card.

**Affected plan tasks:** Task 15 (object_gen writes thumbnail too), Task 31 (OG renderer rewritten).

---

## Amendment H — ULID exhibit IDs

Replace `uuid.uuid4().hex` with ULIDs (26 chars, base32, time-ordered, no enumeration risk):

```python
from ulid import ULID

def _new_exhibit_id() -> str:
    return str(ULID())
```

Add `python-ulid>=2.7` to `backend/pyproject.toml`.

**Affected plan tasks:** Task 17 (routing).

---

## Amendment I — HUD as museum signage

`StationIndicator.tsx`, `ProgressBar.tsx`, `HintChip.tsx` styled with:

- Serif font for label text (Cormorant Garamond or fallback to system Serif)
- Brass-tone bottom border on the StationIndicator (`#8a6b3a`, 1px)
- Description text in `italic small-caps` for the station name
- Background blur (`backdrop-blur-sm`) instead of solid black/40

Small visual change, large feel change.

**Affected plan tasks:** Task 22 (HUD).

---

## Summary — task additions and renumbering

The plan's task list gains:

- **Task 9.5:** Latency benchmark (after Spike A, before Spike B).
- **Task 14.5:** Coordinate frame fitting module (`scene_geometry/frame_fit.py`) after scene_gen, before routing wires it.
- **Task 15.5:** Safety check Modal function (`modal/safety_check.py`) integrated into object_gen.
- Tasks 3, 13, 14, 15, 16, 17, 21, 22, 23, 25, 27, 30, 31 get amended per the sections above.

The bump is `schema_version: "v4.1" → "v4.2"`.

The total task count goes from 38 to ~41 with three insertions; effort estimate adjusts from 6 weeks to **6.5–7 weeks** honestly. If portfolio-mode pressure demands 6 weeks flat, Amendment G (OG card stylization) can ship a simpler v1 (plain text-on-dark) and the poster version becomes a week-7 polish task.

---

## Execution order

1. Subagents read this file first.
2. Each subagent for tasks 3, 13, 14, 15, 16, 17, 21, 22, 23, 25, 27, 30, 31 must check the corresponding Amendment and incorporate the change.
3. New tasks 9.5, 14.5, 15.5 are inserted with the content above.
4. The pinned week-1 done sentence governs all spike work.
