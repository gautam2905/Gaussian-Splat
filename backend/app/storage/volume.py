"""Per-exhibit storage on local filesystem (Modal Volume in production).

In v1: data_dir is a local path (e.g., /tmp/museums-data or ./data).
The per-exhibit layout:

    {data_dir}/exhibits/{exhibit_id}/
        manifest.json
        scene/
            scene.glb         (copied from backend/stock_rooms/rooms/)
        objects/
            {object_id}.glb   (copied from backend/stub_objects/glbs/)
"""
from pathlib import Path

from ..config import get_settings
from ..schemas import Manifest


def _configured_root() -> Path:
    """Return the configured data root path (does NOT mkdir on reads)."""
    return Path(get_settings().data_dir)


def data_dir() -> Path:
    """Configured data directory; auto-created if missing (write path only)."""
    p = _configured_root()
    p.mkdir(parents=True, exist_ok=True)
    return p


def exhibit_dir(exhibit_id: str) -> Path:
    p = data_dir() / "exhibits" / exhibit_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def _manifest_path_ro(exhibit_id: str) -> Path:
    """Pure path resolution — no side effects (safe for reads)."""
    return _configured_root() / "exhibits" / exhibit_id / "manifest.json"


def write_manifest(m: Manifest) -> None:
    p = _manifest_path_ro(m.exhibit_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(m.model_dump_json(indent=2))


def read_manifest(exhibit_id: str) -> Manifest | None:
    p = _manifest_path_ro(exhibit_id)
    if not p.exists():
        return None
    return Manifest.model_validate_json(p.read_text())


def list_exhibits() -> list[str]:
    """All exhibit_ids that have manifests on disk."""
    base = _configured_root() / "exhibits"
    if not base.exists():
        return []
    return sorted([p.name for p in base.iterdir() if (p / "manifest.json").exists()])
