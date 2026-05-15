"""Exhibits API — real pipeline behind SSE.

POST /api/exhibits validates the prompt and returns an events_url the client
subscribes to. The pipeline runs inside the SSE generator so progress events
flow naturally without a separate task queue.

GET /api/exhibits/events?prompt=...  drives the pipeline and streams events.
GET /api/exhibits/{exhibit_id}       returns the persisted manifest.
"""
from urllib.parse import quote

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from ..routing import run_pipeline
from ..storage.volume import read_manifest

router = APIRouter(prefix="/api", tags=["exhibits"])


class CreateExhibitRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=500)


@router.post("/exhibits", status_code=202)
def create_exhibit(req: CreateExhibitRequest) -> dict:
    """Validate prompt; return events_url the client should subscribe to."""
    return {
        "events_url": f"/api/exhibits/events?prompt={quote(req.prompt)}",
    }


@router.get("/exhibits/events")
async def exhibit_events_with_prompt(prompt: str) -> EventSourceResponse:
    """SSE-driven pipeline. Subscribe here with ?prompt=... and consume
    stage events until 'ready' / 'refused' / 'failed'."""
    return EventSourceResponse(run_pipeline(prompt))


@router.get("/exhibits/{exhibit_id}")
def get_exhibit(exhibit_id: str) -> dict:
    m = read_manifest(exhibit_id)
    if m is None:
        raise HTTPException(404, detail={"exhibit_id": exhibit_id, "detail": "not found"})
    return m.model_dump(mode="json")
