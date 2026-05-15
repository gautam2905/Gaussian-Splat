"""End-to-end pipeline test in stub mode.

Drives run_pipeline directly (not via SSE) and asserts:
- All stages emit events
- The final 'ready' event has the exhibit_id
- The manifest is persisted on disk
- Object GLBs are copied from the stub library into the exhibit's objects/ dir
"""
import asyncio
import json
from pathlib import Path

import pytest

from app.config import get_settings
from app.routing import run_pipeline
from app.schemas import JobStage, Manifest
from app.storage.volume import read_manifest


@pytest.fixture(autouse=True)
def _force_stub_mode(monkeypatch, tmp_path):
    monkeypatch.setenv("ORCHESTRATOR_MODE", "stub")
    monkeypatch.setenv("OBJECT_GEN_MODE", "stub")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_pipeline_end_to_end_anatomy(tmp_path: Path) -> None:
    events: list[dict] = []
    async for ev in run_pipeline("An anatomy exhibit"):
        events.append({"event": ev["event"], "data": json.loads(ev["data"])})

    event_types = [e["event"] for e in events]
    assert "stage" in event_types
    assert "object_ready" in event_types  # at least one
    assert event_types[-1] == "ready"

    exhibit_id = events[-1]["data"]["exhibit_id"]
    assert exhibit_id

    # Manifest persisted
    m = read_manifest(exhibit_id)
    assert m is not None
    assert m.stage == JobStage.ready
    assert m.scene.scene_url.endswith("/scene/scene.glb")
    assert m.scene.room_frame is not None
    assert len(m.objects) >= 1

    # Coords transformed: control points should NOT all be inside [0,1] anymore
    assert any(any(abs(c) > 1.1 for c in cp) for cp in m.spline.control_points), (
        "Expected world-coord control points (post frame transform), got normalized"
    )

    # GLBs on disk
    scene_glb = tmp_path / "exhibits" / exhibit_id / "scene" / "scene.glb"
    assert scene_glb.exists()
    for obj in m.objects:
        if obj.glb_url:
            obj_path = tmp_path / "exhibits" / exhibit_id / "objects" / f"{obj.id}.glb"
            assert obj_path.exists(), f"object GLB missing: {obj_path}"


@pytest.mark.asyncio
async def test_pipeline_horology(tmp_path: Path) -> None:
    events: list[dict] = []
    async for ev in run_pipeline("A horology exhibit with three pocket watches"):
        events.append({"event": ev["event"], "data": json.loads(ev["data"])})

    assert events[-1]["event"] == "ready"
    exhibit_id = events[-1]["data"]["exhibit_id"]
    m = read_manifest(exhibit_id)
    assert m is not None
    assert m.title.lower() != ""


@pytest.mark.asyncio
async def test_pipeline_emits_timing(tmp_path: Path) -> None:
    events = []
    async for ev in run_pipeline("an anatomy exhibit"):
        events.append({"event": ev["event"], "data": json.loads(ev["data"])})
    exhibit_id = events[-1]["data"]["exhibit_id"]
    m = read_manifest(exhibit_id)
    assert m.timing.orchestrator_ms >= 0
    assert m.timing.scene_gen_ms >= 0
    assert m.timing.total_ms > 0
