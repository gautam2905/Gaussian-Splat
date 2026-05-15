"""Scene generation: file-copy from stock_rooms (per Amendment L; HunyuanWorld dropped from v1).

For production with HunyuanWorld restored, this becomes a Modal function call.
"""
import shutil
import time
from pathlib import Path

from .scene_geometry.frame_fit import stock_glb_path


def scene_gen_stock(stock_room_id: str, exhibit_id: str, data_dir: Path | str) -> dict:
    """Copy the stock-room GLB into the exhibit's scene/ dir.

    Returns:
        {"scene_url": str, "elapsed_ms": int, "source_glb": str}
    """
    t0 = time.perf_counter()
    data_dir = Path(data_dir)
    src = stock_glb_path(stock_room_id)
    if not src.exists():
        raise RuntimeError(f"Stock GLB not on disk: {src}")
    out_dir = data_dir / "exhibits" / exhibit_id / "scene"
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / "scene.glb"
    shutil.copyfile(src, dst)
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    return {
        "scene_url": f"/exhibits/{exhibit_id}/scene/scene.glb",
        "elapsed_ms": elapsed_ms,
        "source_glb": str(src.name),
    }
