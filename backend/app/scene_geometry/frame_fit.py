"""Coordinate frame helpers for stock rooms.

Per Amendment L: stock rooms have RoomFrames precomputed at asset-prep time
(stored in backend/stock_rooms/manifest.json). This module loads them
into the in-memory RoomFrame Pydantic model.

If/when HunyuanWorld returns, frame_fit() would be the runtime-fit function
operating on a fresh GLB. For v1 we only load precomputed frames.
"""
import json
from pathlib import Path

from ..schemas import RoomFrame

_STOCK_MANIFEST = (
    Path(__file__).resolve().parent.parent.parent / "stock_rooms" / "manifest.json"
)

_INDEX: dict[str, dict] | None = None


def _lazy_load_index() -> dict[str, dict]:
    """Map stock_room_id -> entry dict."""
    global _INDEX
    if _INDEX is None:
        if not _STOCK_MANIFEST.exists():
            raise RuntimeError(f"Stock-rooms manifest missing: {_STOCK_MANIFEST}")
        data = json.loads(_STOCK_MANIFEST.read_text())
        # Manifest is a JSON array; key by id
        _INDEX = {entry["id"]: entry for entry in data}
    return _INDEX


def load_room_frame(stock_room_id: str) -> RoomFrame:
    idx = _lazy_load_index()
    entry = idx.get(stock_room_id)
    if entry is None:
        raise RuntimeError(f"Unknown stock_room_id: {stock_room_id!r}")
    return RoomFrame.model_validate(entry["room_frame"])


def stock_glb_path(stock_room_id: str) -> Path:
    """Absolute path to the stock-room GLB file."""
    idx = _lazy_load_index()
    entry = idx.get(stock_room_id)
    if entry is None:
        raise RuntimeError(f"Unknown stock_room_id: {stock_room_id!r}")
    base = Path(__file__).resolve().parent.parent.parent / "stock_rooms"
    return base / entry["glb_path"]


def transform_manifest_to_world(manifest, frame: RoomFrame):
    """Transform normalized [0,1] coordinates in manifest into world coordinates
    using the room's RoomFrame.

    For axis-aligned stock rooms (basis = identity, extent in metres), the
    transform reduces to:
        world = origin + (normalized - center_offset) * extent

    x/z are centered (0.5 → 0 in local space), y is bottom-aligned (floor at y=0).
    """
    import numpy as np

    origin = np.array(frame.origin)
    bx = np.array(frame.basis_x)
    by = np.array(frame.basis_y)
    bz = np.array(frame.basis_z)
    extent = np.array(frame.extent)

    def to_world(p: list[float]) -> list[float]:
        # x/z centered around room midpoint; y bottom-aligned at floor
        n = np.array(p) - np.array([0.5, 0.0, 0.5])
        scaled = n * extent
        local = bx * scaled[0] + by * scaled[1] + bz * scaled[2]
        return (origin + local).tolist()

    # Pedestals
    new_pedestals = []
    for ped in manifest.pedestals:
        d = ped.model_dump()
        d["position"] = to_world(d["position"])
        new_pedestals.append(d)

    # Spline control points
    new_cps = [to_world(p) for p in manifest.spline.control_points]

    # Rebuild manifest with transformed coords
    m_dict = manifest.model_dump()
    m_dict["pedestals"] = new_pedestals
    m_dict["spline"]["control_points"] = new_cps

    from ..schemas import Manifest
    return Manifest.model_validate(m_dict)
