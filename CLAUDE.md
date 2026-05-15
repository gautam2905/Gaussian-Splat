# CLAUDE.md — onboarding for a new agent

> Read this file first. Then read `HANDOFF.md`. You'll be productive in ~10 minutes.

## What this project is

**Interactive AI Museums** — a web app where the user types a one-line prompt and gets back a 3D museum room they walk through (on rails), with objects on pedestals that can be dissected (clipping plane + exploded view) in the browser.

Stack: **FastAPI + Vite/React 19 + three.js + OpenAI gpt-4o-mini + Modal (TRELLIS.2)**, all paused on the ML side for v1 — see "Current mode" below.

## Current mode (CRITICAL — read before doing anything)

**The project ships in stub mode** — runs end-to-end for **$0** using a hand-authored Manifest stub + a vendored CC0 GLB library + placeholder stock rooms. Production ML is wired but disabled via two env vars.

| Component | Default (v1) | Production switch |
|---|---|---|
| Orchestrator | `ORCHESTRATOR_MODE=stub` (hand-authored fixtures) | `=openai` + `OPENAI_API_KEY` |
| Object generation | `OBJECT_GEN_MODE=stub` (CC0 library copy) | `=modal` + deployed `modal_pkg/object_gen.py` |
| Scene generation | Stock-room file copy (always; HunyuanWorld dropped per Amendment L) | — |

**Do not** flip these switches without explicit user authorization. The user has tight API budgets:
- **OpenAI: $1 cap** — already near zero on dev.
- **Modal: ~$0.50 spent, near $1 informal cap** — no further Modal runs in v1.

## Run it

```bash
cd /mnt/3D_Mapping
make install          # conda env + npm install; ~1 min first time
make test             # 41 backend + 6 frontend tests, all pass, $0
make dev              # backend :8000 + frontend :5173
```

Open <http://localhost:5173>, click any of the 3 gallery cards (Anatomy / Horology / Botany), wait ~1 second for the SSE pipeline, then:
- **Walk:** `→`/`Space` to advance station, `←` to retreat. Mouse drag for free look (pointer lock — click canvas first).
- **Dissect:** click any pedestal → camera flies in → right-side panel shows clipping-plane axis dropdown + cross-section slider + exploded view slider.
- **Exit focus:** `Esc` or "Back to exhibit" button.

## Repo layout (one-paragraph each)

```
backend/         FastAPI app. Pipeline: orchestrator → scene_gen → object_gen → manifest.
                 Stub-mode default. See backend/app/routing.py for the spine.

frontend/        Vite + React 19 + TS strict + three.js (raw, not R3F).
                 Pages: Landing (hero + gallery), Generate (SSE), Exhibit (viewer).
                 Viewer is at frontend/src/viewer/ with locomotion/, dissection/, hud/.

modal_pkg/       Modal ML function definitions. NOT DEPLOYED in v1. spike_a_trellis.py
                 has the FLUX+TRELLIS.2 chain ready to run when budget allows.

docs/superpowers/specs/   Design spec (the canonical "what we're building").
docs/superpowers/plans/   Original 38-task implementation plan (now compressed).

PRE_EXECUTION_NOTES.md    13 binding amendments (A-L) made during execution.
SPIKE_RESULT.md           Honest spike outcomes + budget-pivot decisions.
HANDOFF.md                Full review doc — what's stub vs prod, what's deferred, why.
```

## Critical conventions

### Python
- **Conda env `museums`** (Python 3.12). All Python work goes through `conda run -n museums --no-capture-output <cmd>`. No system pip installs. No uv.
- pyproject deps in `backend/pyproject.toml`. Pip install via `conda run -n museums --no-capture-output pip install -e 'backend/[dev]'`.
- Pydantic v2. Pydantic-settings for config.

### TypeScript
- **Strict mode is on** (`strict: true` + `erasableSyntaxOnly: true` in `frontend/tsconfig.app.json`).
- `erasableSyntaxOnly: true` forbids: TS enums, constructor parameter properties (`constructor(private x: T)`), namespaces, decorators. Use explicit field declarations instead.
- No `any`. Use `unknown` and narrow.
- **WebGLRenderer must be `new WebGLRenderer({ ..., stencil: true })`** — without this, stencil-buffer capping silently fails. Already set in `frontend/src/viewer/ExhibitViewer.tsx`.

### Git
- **Global attribution is DISABLED** in `~/.claude/settings.json`. **DO NOT add `Co-Authored-By` lines to commit messages.**
- Conventional commits: `feat(scope): ...`, `fix(scope): ...`, `docs(scope): ...`, `chore: ...`, `spike(letter): ...`.
- Don't force-push to main without explicit user authorization (granted previously for the initial push).

### Subagent workflow (when implementing features)
The project was built using `superpowers:subagent-driven-development`. Per-task pattern:
1. **Implementer** subagent writes the code.
2. **Spec-compliance reviewer** subagent verifies file-by-file against the brief.
3. **Code-quality reviewer** subagent reviews for design / bugs.
4. Fix any issues inline (< 10 lines) or re-dispatch implementer (≥ 10 lines).

This was the user's explicit preference. Honor it for new features unless they tell you otherwise.

## Manifest schema v4.2

Single source of truth: `backend/app/schemas.py` (Python) and `frontend/src/api/types.ts` (TS mirror). Per **Amendment A**:

- Pedestal `position` and Spline `control_points` are emitted by the orchestrator in **normalized [0,1] room-local coordinates**.
- After scene_gen, `backend/app/scene_geometry/frame_fit.py` loads the stock-room's precomputed `RoomFrame` and transforms manifest positions to **world coordinates**.
- `scene.room_frame === null` means pre-transform; populated means post-transform.

Per **Amendment B** the camera is driven by an **explicit FSM** (`frontend/src/viewer/locomotion/CameraStateMachine.ts`) with 4 states: `ON_SPLINE → FLYING_IN → AT_PEDESTAL → FLYING_OUT`. Mouse-look outputs **yaw/pitch deltas only**; the FSM composes them on top of each state's base pose. Never write the camera directly from mouse-look.

## The amendments (read PRE_EXECUTION_NOTES.md for full text)

13 binding amendments (A–L; one skipped letter intentional). Most-cited:
- **A** — normalized [0,1] manifest coords + RoomFrame transform
- **B** — explicit camera FSM
- **K** — orchestrator uses OpenAI gpt-4o-mini (not Claude)
- **L** — HunyuanWorld dropped; Approach C (stock rooms) only
- **H** — ULID exhibit IDs (26-char, time-ordered)

## What's NOT done / open questions

From `HANDOFF.md`. The user is reviewing the project and will decide these:

1. **Empirical TRELLIS.2 validation** — never got a real GLB out due to Modal disconnect + budget cap. Cached Modal image is ready when budget allows. The user has accepted DINOv3 + RMBG-2.0 licenses on HuggingFace.
2. **Photorealistic stock rooms** — current 10 are colored-box primitives. Polyhaven only has objects; need Sketchfab CC0 hunt or Blender hand-render.
3. **Production deploy** — not done. Modal `@asgi_app` planned (free tier sufficient).
4. **Demo videos** — not recorded.
5. **Brand name / license** — TBD.
6. **OG card generation, analytics, rate limiting** — Amendment G/related, deferred.

Four UX/style decisions waiting on the user (from `HANDOFF.md` §questions):
1. Gallery card UX: auto-fire generate (current) vs pre-baked curated exhibits?
2. Source photorealistic CC0 rooms before next review, or leave as manual curation?
3. Museum-signage HUD (serif + brass) right tone, or modern?
4. Pedestal styling — keep minimal grey cylinders, or marble + brass plaque?

## When the user asks you to add a feature

1. Read `HANDOFF.md` for current state.
2. Check `PRE_EXECUTION_NOTES.md` for any binding amendment that constrains the work.
3. Check `docs/superpowers/specs/` for the design intent.
4. Make a plan (consider invoking `superpowers:writing-plans`).
5. Execute via subagent-driven-development if the task is non-trivial.
6. Keep stub mode working. **Default behavior must not require any API key.**

## When the user asks you to flip on real ML

Two changes:

```bash
# 1. backend/.env (create if missing):
OPENAI_API_KEY=sk-...
ORCHESTRATOR_MODE=openai
OBJECT_GEN_MODE=modal           # only after step 2

# 2. Deploy the Modal function:
conda run -n museums --no-capture-output modal deploy modal_pkg/object_gen.py
```

Caveats:
- TRELLIS.2 needs DINOv3 + RMBG-2.0 + ideally FLUX.1-schnell licenses accepted on HF. User has done the first two.
- Modal H100 time costs ~$0.10/object. A 3-object exhibit ≈ $0.30. Check user's current budget before triggering.
- The `modal run` CLI has a known disconnect issue — always use `--detach` flag, or use `modal deploy` + Python SDK invocation instead.

## Common pitfalls (things that wasted time in v1)

- **`modal run` without `--detach`** killed multiple spike runs mid-inference. Always `--detach` for any non-trivial run.
- **Co-Authored-By lines in commit messages** trip the project's global-attribution-disabled rule. Don't add them.
- **TypeScript `erasableSyntaxOnly: true`** rejects constructor parameter properties. Subagents have been caught by this multiple times; use explicit field declarations.
- **`WebGLRenderer` without `stencil: true`** = capped clipping silently fails (Chromium allocates 0 stencil bits by default).
- **`<a href>`** instead of `<Link>` causes full page reloads. Use `react-router-dom`'s `Link` for client-side nav.
- **Scope warning hook** fires after ~5+ file edits. It's non-fatal noise — the warning measures cumulative session changes, not single-task scope. Acknowledge and continue.

## Tests are the source of ground truth

```bash
make test
```

41 backend tests + 6 frontend tests, all green, $0 spend. If you change anything, this should still be all green. Don't merge with a red test.

## End of onboarding

You should now be able to:
- Run the demo (`make dev`).
- Find the design intent (`docs/superpowers/specs/`).
- Find the implementation contract (`backend/app/schemas.py` + `frontend/src/api/types.ts`).
- Find recent decisions (`PRE_EXECUTION_NOTES.md`).
- Find the honest review of state (`HANDOFF.md`).
- Understand why TRELLIS.2 isn't running yet (Modal budget + disconnect issue).
- Understand why HunyuanWorld is gone (Amendment L).

When the user gives you a task, ask one clarifying question if anything's ambiguous, then proceed.
