# Interactive AI Museums

> Type a prompt. Step inside. Take it apart.

A web app where the user types a one-line prompt and gets back a 3D museum room — generated, walkable via on-rails camera, with objects on pedestals that can be dissected (cross-section + exploded view) in the browser.

**Stack:** FastAPI + Modal (paused for v1) + OpenAI gpt-4o-mini (paused for v1) + Vite/React 19 + three.js. Demo currently ships in **stub mode** — runs end-to-end for $0 using a pre-vendored CC0 GLB library + a hand-authored manifest stub. Production ML (TRELLIS.2 object generation, HunyuanWorld scene generation) is wired but disabled by config.

## Quickstart

Prerequisites: Python 3.12 (conda env `museums` provisioned during the build), Node 20.18+, `npm`, `git`.

```bash
# Backend
cd backend
conda run -n museums --no-capture-output pip install -e '.[dev]'

# Frontend
cd ../frontend
npm install

# Run both (in two terminals or via make dev)
cd ../
make dev
```

Then open <http://localhost:5173/>.

## How it works in stub mode

1. Visit `/`, click any of the 3 curated gallery cards (Anatomy / Horology / Botany).
2. The landing redirects to `/generate?prompt=...&auto=true` which auto-fires the SSE pipeline.
3. Backend pipeline runs in ~150 ms:
   - **Orchestrator** (`backend/app/orchestrator/stub.py`): keyword-matches the prompt to one of 3 hand-authored Manifest fixtures.
   - **Scene gen** (`backend/app/scene_gen.py`): copies a stock-room GLB from `backend/stock_rooms/rooms/`.
   - **Object gen** (`backend/app/object_gen/stub.py`): for each object in the manifest, fuzzy-matches its `image_prompt` against the vendored CC0 GLB library and copies the best match.
   - **Manifest persisted** to `${DATA_DIR}/exhibits/{ulid}/manifest.json`.
4. SSE stream emits `ready` event with the exhibit_id → frontend navigates to `/exhibit/{id}`.
5. Three.js viewer loads the scene + pedestals + objects.
6. User presses `→` / `←` to advance/retreat between stations. Free mouse-look. Click an object → camera flies to a hero pose. Side panel exposes clipping-plane + exploded-view controls.

## Layout

```
backend/
  app/
    api/                  FastAPI routes (exhibits, health)
    orchestrator/         OpenAI + stub planner; emits Manifest v4.2
    object_gen/           Stub library dispatcher + Modal placeholder
    scene_gen.py          Stock-room file-copy
    scene_geometry/       Precomputed RoomFrame loader + coord transform
    storage/              Per-exhibit volume I/O
    routing.py            Pipeline orchestration with SSE events
    schemas.py            Manifest v4.2 + RoomFrame Pydantic models
  stock_rooms/            10 placeholder primitive rooms + manifest
  stub_objects/           10 CC0 Khronos GLBs + library.json (label keywords)
  tests/                  41 tests, all green, no API calls

frontend/
  src/
    api/                  Typed client mirroring Manifest v4.2
    pages/                Landing + Generate (SSE) + Exhibit (viewer host)
    viewer/
      ExhibitViewer.tsx   three.js root
      Scene.ts / Pedestals.ts     asset loaders
      locomotion/         Spline + RailController + MouseLook + CameraStateMachine (Amendment B explicit FSM)
      dissection/         ClippingPlane + CappedClipMaterial (stencil) + ExplodedView + DissectionPanel
      hud/                Station indicator + progress bar + hint chip
      store/              Zustand for exhibit/locomotion state

modal_pkg/                Modal ML function definitions (deployed but currently bypassed)
references/               Source code for TRELLIS.2, HunyuanWorld, supersplat, splatviz (gitignored)
spikes/                   Spike A/B/C exploratory work (gitignored)

docs/
  superpowers/specs/      Design spec
  superpowers/plans/      Original 38-task implementation plan

PRE_EXECUTION_NOTES.md    13 amendments (A-L) made during execution; binding on the design
SPIKE_RESULT.md           Honest spike outcomes + budget pivot decisions
HANDOFF.md                Comprehensive review doc — start here
```

## Flipping switches (when budget allows)

Edit `backend/.env`:

```
OPENAI_API_KEY=sk-...
ORCHESTRATOR_MODE=openai     # was "stub"
OBJECT_GEN_MODE=modal        # was "stub" — also requires deploying modal_pkg/object_gen.py
```

Restart the backend. Same pipeline; real ML now runs.

## Tests

```bash
make test
```

Runs 41 backend tests + 6 frontend (Spline) tests. All pass in stub mode without any API key. The single OpenAI live-test is gated behind `RUN_OPENAI_TESTS=1` and `OPENAI_API_KEY` for opt-in.

## What's incomplete / known issues

See [`HANDOFF.md`](./HANDOFF.md) for the full review.

## License

Project code: TBD (likely MIT after review). Bundled assets in `backend/stub_objects/` are CC0 from Khronos GLTF Sample Assets. Stock rooms are synthesized primitives (CC0).
