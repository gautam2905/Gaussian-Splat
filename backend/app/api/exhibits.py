"""M0 stub — every ML stage returns 501."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

router = APIRouter(prefix="/api", tags=["exhibits"])


class CreateExhibitRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=500)


@router.post("/exhibits", status_code=501)
def create_exhibit(req: CreateExhibitRequest) -> None:
    raise HTTPException(
        status_code=501,
        detail={"stage": "orchestrator", "detail": "Not yet implemented (M0)"},
    )


@router.get("/exhibits/{exhibit_id}")
def get_exhibit(exhibit_id: str) -> None:
    raise HTTPException(status_code=404, detail={"exhibit_id": exhibit_id, "detail": "not found"})


@router.get("/exhibits/{exhibit_id}/events")
async def exhibit_events(exhibit_id: str) -> EventSourceResponse:
    async def gen():
        yield {"event": "not_implemented", "data": exhibit_id}
    return EventSourceResponse(gen())
