"""Canonical Manifest schema — v4.2.

The Manifest is the single source of truth for an exhibit. Every downstream module
(scene_gen, object_gen, viewer) reads from it; the orchestrator (Claude) emits it.

Coordinate convention:
- The orchestrator emits `pedestals[*].position` and `spline.control_points`
  in NORMALIZED room-local coordinates (each component in [0, 1]).
- After scene generation, `backend.scene_geometry.frame_fit` fits a `RoomFrame` to
  the generated scene GLB and transforms manifest positions into world coordinates.
- The schema does NOT enforce the [0,1] range — both pre- and post-transform manifests
  are structurally valid. `scene.room_frame is None` indicates pre-transform state.

Schema version bumps when fields are removed/renamed/retyped. Additive changes are
allowed within the same version.
"""
from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

SchemaVersion = Literal["v4.2"]
ClipAxis = Literal["x", "y", "z"]
SceneGenerator = Literal["hunyuanworld", "stock"]


class JobStage(str, Enum):
    queued = "queued"
    orchestrating = "orchestrating"
    scene_gen = "scene_gen"
    objects_gen = "objects_gen"
    ready = "ready"
    failed = "failed"


class Color(BaseModel):
    color: str
    intensity: float = Field(ge=0.0, le=10.0)


class Fog(BaseModel):
    color: str
    near: float = Field(ge=0.0)
    far: float = Field(ge=0.0)


class RoomFrame(BaseModel):
    """Coordinate frame fit to a generated scene GLB.

    Used by `backend.scene_geometry.frame_fit` to transform normalized
    manifest positions into the scene's world coordinates.
    """
    origin: list[float] = Field(min_length=3, max_length=3)
    basis_x: list[float] = Field(min_length=3, max_length=3)
    basis_y: list[float] = Field(min_length=3, max_length=3)
    basis_z: list[float] = Field(min_length=3, max_length=3)
    extent: list[float] = Field(min_length=3, max_length=3)  # room size in metres


class Scene(BaseModel):
    generator: SceneGenerator
    scene_prompt: str
    stock_room_id: str | None = None
    scene_url: str
    ambient_light: Color
    fog: Fog | None = None
    room_frame: RoomFrame | None = None  # populated after frame fit


class Station(BaseModel):
    t: float = Field(ge=0.0, le=1.0)
    label: str
    pedestal_id: str | None
    preferred_yaw_deg: float


class Spline(BaseModel):
    """Camera path through the scene.

    `control_points` are in normalized [0,1] room-local coordinates before frame fit,
    in world coordinates after. See module docstring.
    """
    control_points: list[list[float]] = Field(min_length=2)
    stations: list[Station] = Field(min_length=2)
    travel_time_s: float = Field(gt=0.0)

    @field_validator("control_points")
    @classmethod
    def _three_d(cls, v: list[list[float]]) -> list[list[float]]:
        for p in v:
            if len(p) != 3:
                raise ValueError(f"control point must be [x,y,z], got {p}")
        return v


class PedestalSpec(BaseModel):
    """Pedestal placed on the scene floor.

    `position` is in normalized [0,1] room-local coordinates before frame fit,
    world coordinates after. See module docstring.
    """
    id: str
    position: list[float]
    height: float = Field(gt=0.0)
    object_id: str

    @field_validator("position")
    @classmethod
    def _three_d(cls, v: list[float]) -> list[float]:
        if len(v) != 3:
            raise ValueError(f"position must be [x,y,z], got {v}")
        return v


class ObjectSpec(BaseModel):
    id: str
    label: str
    placard_description: str
    image_prompt: str
    image_url: str | None = None
    glb_url: str | None = None
    rotation_y_deg: float = 0.0
    scale: float = Field(gt=0.0, default=0.1)
    dissectable: bool = True
    preferred_clip_axis: ClipAxis = "y"
    preferred_clip_t: float = Field(ge=0.0, le=1.0, default=0.5)


class Timing(BaseModel):
    orchestrator_ms: int = 0
    scene_gen_ms: int = 0
    objects_gen_ms: dict[str, int] = Field(default_factory=dict)
    total_ms: int = 0


class StageError(BaseModel):
    stage: str
    object_id: str | None = None
    detail: str


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: SchemaVersion = "v4.2"
    exhibit_id: str
    created_at: datetime
    user_prompt: str
    title: str
    description: str
    scene: Scene
    spline: Spline
    pedestals: list[PedestalSpec]
    objects: list[ObjectSpec]
    stage: JobStage
    timing: Timing = Field(default_factory=Timing)
    errors: list[StageError] = Field(default_factory=list)
