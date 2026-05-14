"""Tests for the canonical Manifest contract (schema v4.2)."""
import pytest
from pydantic import ValidationError

from app.schemas import Manifest, RoomFrame


def _valid_payload() -> dict:
    """A valid v4.2 manifest in normalized-coord (pre-frame-fit) state."""
    return {
        "schema_version": "v4.2",
        "exhibit_id": "01HZ8X9A2B0000000000000001",
        "created_at": "2026-05-14T18:00:00Z",
        "user_prompt": "A Victorian study with a pocket watch",
        "title": "The Inventor's Study",
        "description": "An evening in a 19th-century inventor's room.",
        "scene": {
            "generator": "hunyuanworld",
            "scene_prompt": "Victorian study interior, oak panelling",
            "stock_room_id": None,
            "scene_url": "/exhibits/01HZ8X9A2B0000000000000001/scene/scene.glb",
            "ambient_light": {"color": "#f4e2c4", "intensity": 0.6},
            "fog": {"color": "#1a1208", "near": 8.0, "far": 25.0},
            "room_frame": None,
        },
        "spline": {
            "control_points": [[0.5, 0.5, 0.9], [0.3, 0.5, 0.6], [0.7, 0.5, 0.2]],
            "stations": [
                {"t": 0.0, "label": "entrance", "pedestal_id": None, "preferred_yaw_deg": 0},
                {"t": 1.0, "label": "the watch", "pedestal_id": "pedestal_1", "preferred_yaw_deg": -90},
            ],
            "travel_time_s": 6.0,
        },
        "pedestals": [
            {
                "id": "pedestal_1",
                "position": [0.25, 0.0, 0.55],
                "height": 0.95,
                "object_id": "watch_1",
            }
        ],
        "objects": [
            {
                "id": "watch_1",
                "label": "Antique pocket watch",
                "placard_description": "Brass, c. 1880.",
                "image_prompt": "antique brass pocket watch, hero shot, white bg",
                "image_url": "/exhibits/01HZ8X9A2B0000000000000001/objects/watch_1.png",
                "glb_url": "/exhibits/01HZ8X9A2B0000000000000001/objects/watch_1.glb",
                "rotation_y_deg": 30,
                "scale": 0.08,
                "dissectable": True,
                "preferred_clip_axis": "y",
                "preferred_clip_t": 0.5,
            }
        ],
        "stage": "ready",
        "timing": {
            "orchestrator_ms": 1850,
            "scene_gen_ms": 240000,
            "objects_gen_ms": {"watch_1": 31200},
            "total_ms": 273050,
        },
        "errors": [],
    }


def _valid_room_frame() -> dict:
    """A valid RoomFrame for a unit room aligned to world axes."""
    return {
        "origin": [0.0, 0.0, 0.0],
        "basis_x": [1.0, 0.0, 0.0],
        "basis_y": [0.0, 1.0, 0.0],
        "basis_z": [0.0, 0.0, 1.0],
        "extent": [8.0, 3.0, 6.0],
    }


def test_valid_manifest_parses() -> None:
    m = Manifest.model_validate(_valid_payload())
    assert m.exhibit_id == "01HZ8X9A2B0000000000000001"
    assert m.schema_version == "v4.2"
    assert m.scene.generator == "hunyuanworld"
    assert m.scene.room_frame is None
    assert len(m.objects) == 1


def test_manifest_with_room_frame_parses() -> None:
    payload = _valid_payload()
    payload["scene"]["room_frame"] = _valid_room_frame()
    m = Manifest.model_validate(payload)
    assert m.scene.room_frame is not None
    assert m.scene.room_frame.extent == [8.0, 3.0, 6.0]


def test_room_frame_rejects_wrong_dimension() -> None:
    bad = _valid_room_frame()
    bad["origin"] = [0.0, 0.0]  # only 2 dims
    with pytest.raises(ValidationError):
        RoomFrame.model_validate(bad)


def test_rejects_old_schema_version() -> None:
    p = _valid_payload()
    p["schema_version"] = "v4.1"
    with pytest.raises(ValidationError):
        Manifest.model_validate(p)


def test_rejects_negative_travel_time() -> None:
    p = _valid_payload()
    p["spline"]["travel_time_s"] = -1.0
    with pytest.raises(ValidationError):
        Manifest.model_validate(p)


def test_rejects_spline_with_one_station() -> None:
    p = _valid_payload()
    p["spline"]["stations"] = [{"t": 0.0, "label": "lone", "pedestal_id": None, "preferred_yaw_deg": 0}]
    with pytest.raises(ValidationError):
        Manifest.model_validate(p)


def test_rejects_unknown_stage() -> None:
    p = _valid_payload()
    p["stage"] = "not_a_stage"
    with pytest.raises(ValidationError):
        Manifest.model_validate(p)


def test_rejects_bad_clip_axis() -> None:
    p = _valid_payload()
    p["objects"][0]["preferred_clip_axis"] = "w"
    with pytest.raises(ValidationError):
        Manifest.model_validate(p)


def test_boundary_normalized_coords_accepted() -> None:
    """Coords at the exact [0,1] boundaries are accepted (pre-frame-fit state)."""
    p = _valid_payload()
    p["spline"]["control_points"] = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
    p["pedestals"][0]["position"] = [0.0, 0.0, 0.0]
    Manifest.model_validate(p)
    p["pedestals"][0]["position"] = [1.0, 1.0, 1.0]
    Manifest.model_validate(p)


def test_world_coords_also_accepted() -> None:
    """Post-frame-fit coords can be any magnitude; schema only validates shape."""
    p = _valid_payload()
    p["pedestals"][0]["position"] = [-1.0, 0.0, 0.5]  # negative + > 1, valid
    p["spline"]["control_points"] = [[0.0, 1.7, 4.0], [-1.0, 1.7, 2.5]]
    Manifest.model_validate(p)


def test_fog_rejects_near_ge_far() -> None:
    p = _valid_payload()
    p["scene"]["fog"] = {"color": "#1a1208", "near": 25.0, "far": 8.0}
    with pytest.raises(ValidationError):
        Manifest.model_validate(p)


def test_pedestal_rejects_two_dim_position_via_kwargs() -> None:
    """Direct construction must reject 2-dim position (the Point3D alias fires)."""
    from app.schemas import PedestalSpec
    with pytest.raises(ValidationError):
        PedestalSpec(id="x", position=[0.1, 0.2], height=1.0, object_id="o")


def test_station_defaults_pedestal_id_to_none() -> None:
    from app.schemas import Station
    s = Station(t=0.0, label="entrance", preferred_yaw_deg=0.0)
    assert s.pedestal_id is None
