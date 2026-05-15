"""Pipeline orchestration: orchestrator -> scene_gen -> object_gen -> ready.

Emits SSE events to the frontend. Errors degrade gracefully — one failed object
does not abort the exhibit.

Stub mode (default per config) runs the entire pipeline for $0. Production mode
calls OpenAI + Modal under the hood (transparent to this module).
"""
import json
import time
from typing import AsyncIterator

from ulid import ULID

from .config import get_settings
from .object_gen import generate as object_generate
from .object_gen import hints_for_object
from .orchestrator.openai_client import (
    OrchestratorFailed,
    OrchestratorRefused,
    orchestrate,
)
from .scene_gen import scene_gen_stock
from .scene_geometry.frame_fit import load_room_frame, transform_manifest_to_world
from .schemas import JobStage, Manifest, ObjectSpec, StageError
from .storage.volume import data_dir as get_data_dir
from .storage.volume import write_manifest


def new_exhibit_id() -> str:
    """ULID — 26 chars, time-ordered, base32, no enumeration risk."""
    return str(ULID())


def _sse(event: str, data: dict) -> dict:
    """Format for sse_starlette's EventSourceResponse."""
    return {"event": event, "data": json.dumps(data)}


async def run_pipeline(user_prompt: str) -> AsyncIterator[dict]:
    """Drive the full pipeline; yield SSE events for the frontend.

    Final events:
      - 'ready' with {exhibit_id} on success
      - 'refused' with {category, reason} on Claude/OpenAI refusal
      - 'failed' with {detail} on orchestrator or scene failure
    """
    exhibit_id = new_exhibit_id()
    t_start = time.perf_counter()
    data_root = get_data_dir()

    yield _sse("stage", {"stage": "orchestrating", "exhibit_id": exhibit_id})

    # ---- Stage 1: Orchestrator ----
    try:
        manifest: Manifest = orchestrate(user_prompt, exhibit_id=exhibit_id)
    except OrchestratorRefused as e:
        yield _sse("refused", {"category": e.category, "reason": e.reason, "exhibit_id": exhibit_id})
        return
    except OrchestratorFailed as e:
        yield _sse("failed", {"detail": e.detail, "exhibit_id": exhibit_id})
        return

    manifest.stage = JobStage.scene_gen
    manifest.timing.orchestrator_ms = int((time.perf_counter() - t_start) * 1000)
    write_manifest(manifest)
    yield _sse("stage", {"stage": "scene_gen", "exhibit_id": exhibit_id})

    # ---- Stage 2: scene_gen (file-copy from stock_rooms) ----
    try:
        scene_result = scene_gen_stock(
            manifest.scene.stock_room_id, exhibit_id, data_root
        )
        manifest.scene.scene_url = scene_result["scene_url"]
        manifest.timing.scene_gen_ms = scene_result["elapsed_ms"]

        # Load the precomputed RoomFrame and transform manifest coords to world space
        frame = load_room_frame(manifest.scene.stock_room_id)
        manifest.scene.room_frame = frame
        manifest = transform_manifest_to_world(manifest, frame)
    except Exception as e:
        yield _sse("failed", {"detail": f"scene_gen: {e}", "exhibit_id": exhibit_id})
        return

    write_manifest(manifest)
    yield _sse("stage", {"stage": "objects_gen", "exhibit_id": exhibit_id})

    # ---- Stage 3: object_gen (per-object; sequential in stub, parallelisable later) ----
    new_objects: list[ObjectSpec] = []
    for obj in manifest.objects:
        try:
            result = object_generate(
                image_prompt=obj.image_prompt,
                exhibit_id=exhibit_id,
                object_id=obj.id,
                data_dir=data_root,
            )
            d = obj.model_dump()
            d["image_url"] = result.get("image_url")
            d["glb_url"] = result["glb_url"]
            manifest.timing.objects_gen_ms[obj.id] = result["elapsed_ms"]

            # In stub mode, override per-object hints from the stub library so
            # dissection axis / t / rotation / scale match what was actually loaded.
            settings = get_settings()
            if settings.object_gen_mode == "stub":
                hints = hints_for_object(obj.image_prompt)
                d["dissectable"] = hints["dissectable"]
                d["preferred_clip_axis"] = hints["preferred_clip_axis"]
                d["preferred_clip_t"] = hints["preferred_clip_t"]
                d["rotation_y_deg"] = hints["rotation_y_deg"]
                d["scale"] = hints["scale_hint"]

            new_objects.append(ObjectSpec.model_validate(d))
            yield _sse("object_ready", {"object_id": obj.id, "exhibit_id": exhibit_id})
        except Exception as e:
            manifest.errors.append(
                StageError(stage="object_gen", object_id=obj.id, detail=str(e))
            )
            # Keep the un-generated object in the manifest with null URLs so the
            # frontend can show a "missing" placard.
            new_objects.append(obj)
            yield _sse("object_failed", {"object_id": obj.id, "reason": str(e), "exhibit_id": exhibit_id})

    manifest.objects = new_objects

    # ---- Stage 4: ready ----
    manifest.stage = JobStage.ready
    manifest.timing.total_ms = int((time.perf_counter() - t_start) * 1000)
    write_manifest(manifest)
    yield _sse("ready", {"exhibit_id": exhibit_id, "total_ms": manifest.timing.total_ms})
