# HANDOFF — Interactive AI Museums

> A one-page-at-a-time review of what got built, what's stub vs prod, and what's deferred.
> Read this before diving into the code.

## The TL;DR

**The full end-to-end demo works in stub mode for $0.** Visit `/`, click a gallery card, walk + dissect. No OpenAI or Modal calls happen. The production path is wired but disabled by config — flip two env vars to enable.

**What got built:** the backend pipeline, the orchestrator (with OpenAI gpt-4o-mini + stub fallback), the stub object library (10 CC0 GLBs from Khronos), 10 placeholder stock rooms, the entire frontend viewer with Catmull-Rom rails locomotion + explicit camera state machine + stencil-buffer capped clipping + exploded view + side panel.

**What's deferred:** real TRELLIS.2 generation (blocked on Modal budget), HunyuanWorld scene generation (dropped per Amendment L due to licensing + dependency issues), photorealistic stock rooms (current ones are colored boxes), demo videos, production deploy.

## How to verify it works (3 minutes)

```bash
cd /mnt/3D_Mapping
make install       # ~1 min if first run, else seconds
make test          # ~5 seconds — should show 41 backend + 6 frontend tests passing
make dev           # boots backend on :8000 + frontend on :5173
```

Open <http://localhost:5173>, click **The Anatomy Theatre**, wait ~1 second for SSE pipeline, then:
- Press `→` to advance to next station.
- Press `←` to retreat.
- Move mouse (click canvas first for pointer lock) to look around.
- Click a pedestal → camera flies to the object.
- Drag the cross-section slider in the right panel → you see a cut through the object with filled caps.
- Drag the exploded-view slider (if the object is multi-component) → parts separate radially.
- Press `Esc` or click "Back to exhibit" → returns to walking.

## What's stub vs production

The system is designed so production and stub are the **same code path**, gated by two env vars:

| Component | Stub default | Production switch | Cost in prod |
|---|---|---|---|
| Orchestrator | Hand-authored fixtures matched by keyword | `ORCHESTRATOR_MODE=openai` + `OPENAI_API_KEY` | ~$0.001/exhibit (gpt-4o-mini) |
| Scene gen | File copy from `backend/stock_rooms/` | Currently same (HunyuanWorld dropped per Amendment L) | $0 |
| Object gen | Best-keyword-match from `backend/stub_objects/` | `OBJECT_GEN_MODE=modal` + deployed `modal_pkg/object_gen.py` | ~$0.10/object on Modal H100 |

### Flipping to production (when budget allows)

```bash
# Edit backend/.env:
OPENAI_API_KEY=sk-proj-...your-key...
ORCHESTRATOR_MODE=openai           # default was "stub"
OBJECT_GEN_MODE=modal              # default was "stub"
```

Then **also deploy the Modal ML function** (currently just declared, not deployed):

```bash
conda run -n museums --no-capture-output modal deploy modal_pkg/object_gen.py
# Requires accepting the DINOv3, RMBG-2.0, FLUX.1-schnell licenses on HuggingFace
# (DINOv3 and RMBG-2.0 were already accepted during Spike A).
```

Restart `make dev`. Same UI. Real ML.

## Architecture cheat sheet

```
   User prompt
        |
   POST /api/exhibits           -> returns events_url
        |
   SSE /api/exhibits/events?prompt=...
        |
        +-- orchestrator.orchestrate(prompt)        [stub | openai]
        |     -> Manifest v4.2 with normalized [0,1] pedestal/spline coords
        |
        +-- scene_gen.scene_gen_stock(stock_room_id)
        |     -> copies stock-room GLB
        |
        +-- frame_fit.load_room_frame + transform_manifest_to_world
        |     -> manifest coords transformed to world space
        |
        +-- object_gen.generate(image_prompt) x N   [stub | modal]
        |     -> copies/generates per-object GLB
        |
        +-- write_manifest(manifest) at {DATA_DIR}/exhibits/{id}/manifest.json
        |
        +-- emit "ready" SSE event with exhibit_id

   Frontend GET /api/exhibits/{id} -> manifest
        |
   <ExhibitViewer manifest={...} />
        +-- three.js: stencil enabled, local clipping enabled
        +-- Scene.ts loads scene.glb
        +-- Pedestals.ts places pedestal cylinders + object.glb meshes
        +-- CameraStateMachine: ON_SPLINE -> FLYING_IN -> AT_PEDESTAL -> FLYING_OUT
        |     - RailController: tween between stations on Catmull-Rom spline
        |     - MouseLook: pointer-lock; outputs yaw/pitch DELTAS only
        |     - FocusController: hero-pose calc for fly-in tweens
        +-- DissectionPanel (in AT_PEDESTAL):
        |     - ClippingPlane + CappedClipMaterial (stencil buffer cap)
        |     - ExplodedView (per-component radial translation)
        +-- HUD: StationIndicator + ProgressBar + HintChip
```

## What didn't ship (and why)

### TRELLIS.2 empirical validation

We hit two HuggingFace gates (DINOv3, RMBG-2.0) consecutively during Spike A. Both got accepted. The next Modal re-runs were killed by the `modal run` CLI's "local client disconnected" issue — the dispatched subagent dropped the `--detach` flag specified, so the ephemeral app died mid-inference. With the $1 Modal cap already approached, further attempts were stopped and we proceeded on third-party evidence (Microsoft's published TRELLIS.2 examples + Spike C renderer validation). The infrastructure works; there is just no real TRELLIS.2 output to dissect yet. The cached Modal image is ready when budget allows.

### HunyuanWorld scene generation

Dropped per Amendment L. Three HuggingFace license gates + a 420 MB open3d dependency chain + a territory-restricted Tencent custom license made it not worth the complexity for v1. We pivoted to **Approach C** (stock rooms loaded from disk). The 10 stock rooms currently are placeholder primitives (colored boxes) because Polyhaven only has objects, not full-room scenes. Replacing them with photorealistic CC0 rooms is a week-5 curation task not in the budget pivot scope.

### Stock-room visual polish

The 10 stock rooms in `backend/stock_rooms/rooms/` are programmatically-synthesized colored boxes. They render and work, but a Victorian study looks like a brown box. To fix: source 10 photorealistic CC0 room GLBs (Sketchfab CC0 collection, or hand-rendering in Blender) and replace the files. The `manifest.json` + precomputed `RoomFrame` system works for any room GLB — just update the file paths and re-run `make test` to verify nothing broke.

### Production deploy

Backend hasn't been deployed to a public URL. The Modal `@asgi_app` integration was scoped out (requires real Modal spend + DNS + Cloudflare Pages). The README quickstart works on localhost.

### Demo videos

Not recorded — depends on functional spike + visual polish, both deferred.

### Three.js capped-clipping on real TRELLIS.2 mesh topology (Spike C-2)

Spike C confirmed capping works at 144 fps on Khronos `DamagedHelmet` (a clean reference mesh). Whether it works on TRELLIS.2's non-manifold topology is unanswered until a real TRELLIS.2 GLB exists. The `CappedClipMaterial` implementation has the saved-and-restored material state pattern that makes testing clean when the time comes.

## Test summary

```
backend/tests/  ->  41 passed, 1 skipped (OpenAI live-test, opt-in only)
frontend/       ->  6 passed (Spline / Catmull-Rom unit tests)
```

The skipped test runs only with `OPENAI_API_KEY` AND `RUN_OPENAI_TESTS=1` set — double-gated to prevent accidentally burning the $1 OpenAI budget on routine CI runs.

## Git history (high level)

```bash
git log --oneline | head -40
```

Key commits in chronological order:

| Hash | Description |
|---|---|
| `7c91f28` | Init: design spec + plan + pre-execution amendments |
| `4d7742a` | Backend: minimal FastAPI app + `/health` |
| `f5c76eb` | Schemas v4.2 with RoomFrame (Amendment A) |
| `e9ad783` | API stubs (501 wiring) |
| `59462ab` | Frontend: Vite + React 19 + TS scaffold |
| `58d50bb` | Typed API client (mirrors Manifest v4.2) |
| `96f2333` | Fix SSE named-event listeners + content-type headers |
| `ec03694` | Reference repos documented |
| `549b01d` | Amendment J (HunyuanWorld license clarification) |
| `2b6014f` | 10 stock-room placeholders + manifests |
| `9f9f7f6` | Spike A: TRELLIS.2 image build done, BLOCKED on DINOv3 |
| `421f949` | Spike A closed inconclusive — budget cap; Amendment L |
| `8f4cf8a` | Spike B pivot: HunyuanWorld dropped, Approach C committed |
| `7b4253f` | Spike C: three.js capped clipping proof-of-concept |
| `26dea95` | OpenAI orchestrator + stub mode |
| `31dab1e` | Stub object library (10 CC0 GLBs) |
| `36822fb` | Pipeline routing + SSE (end-to-end stub mode works) |
| `f9fff78` | Viewer: rails locomotion + click-to-focus + HUD (Phase 3) |
| `05e004a` | Dissection: stencil-buffer capped clipping + exploded view (Phase 4) |
| `1d99788` | Pages: landing hero + curated gallery + auto-fire generate (Phase 5) |

## What to do next, in priority order

1. **One Modal run, $0.30 budgeted, with `--detach` properly set.** Get 3 GLBs from Spike A. Unblocks the visual proof that dissection works on real TRELLIS.2 output.
2. **Source 5+ photorealistic CC0 rooms** to replace the colored-box placeholders. Even one or two high-quality rooms (e.g., a real Victorian study from Sketchfab CC0) dramatically lifts the demo's feel.
3. **One OpenAI orchestrate call**, $0.001 budget, to see what gpt-4o-mini produces on a custom prompt. The few-shot examples cover anatomy/horology/botany; a novel prompt like "an exhibit of strange seeds" would test generalisation.
4. **Production deploy** if (1)-(3) feel good. Cloudflare Pages for frontend (free), Modal for backend (~$5/mo at low traffic).

## Decisions deferred to you

- **Brand name.** "Interactive AI Museums" works as a category descriptor. The product likely wants a punchier one-word brand at launch.
- **License.** Project code defaults to TBD. Likely MIT or Apache-2.0 after review.
- **Production OpenAI key + Modal credit.** Avoided burning the $1 caps during dev.
- **Whether to add OG image generation, analytics (Plausible), per-IP rate limiting** — all useful for launch, none necessary for demo state.

## Amendments index

13 amendments to the original design spec are in `PRE_EXECUTION_NOTES.md`:

| Amendment | Summary |
|---|---|
| A | Coordinate frame fitting: normalized [0,1] manifest coords -> world via RoomFrame |
| B | Explicit camera state machine (FSM, not three ad-hoc controllers) |
| C | Latency benchmark task deferred (not run) |
| D | Per-object capping fallback (ghost shader) — not needed; stencil approach works |
| E | Output-level NSFW gate deferred — production-only concern |
| F | Refusal categorization |
| G | OG card as stylized poster (deferred) |
| H | ULID exhibit IDs (shipped) |
| I | HUD as museum signage, serif + brass border (shipped) |
| J | HunyuanWorld is Tencent custom-licensed (informational; led to L) |
| K | Orchestrator uses OpenAI gpt-4o-mini, not Claude |
| L | HunyuanWorld dropped from v1; Approach C (stock rooms) is v1 scene path |

Each amendment block in `PRE_EXECUTION_NOTES.md` documents the reasoning and affected tasks.

## Questions for next pass

1. Is the gallery-card-auto-fire UX correct, or should clicking a gallery card go straight to a curated pre-generated exhibit (faster, but less of an "I made this" feeling)?
2. Source the 5+ photorealistic CC0 rooms before next review, or keep that as a manual curation step?
3. Is the museum-signage HUD styling (serif + brass underline) the right tone, or something more modern?
4. Should the placeholder pedestals (basic grey cylinders) be styled — marble + brass plaque — or kept minimal?
