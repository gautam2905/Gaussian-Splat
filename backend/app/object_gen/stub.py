"""Stub object_gen — returns pre-vendored CC0 GLBs by label-fuzzy-match.

Used when settings.object_gen_mode == 'stub' so the demo runs for $0.

The stub library lives at backend/stub_objects/. Each request is a
(prompt, exhibit_id, object_id) tuple; we return a dict matching the
real object_gen response shape:

    {"image_url": str | None, "glb_url": str, "elapsed_ms": int, "stub_id": str}

Matching logic: lowercase the image_prompt, tokenize, score each library
entry by number of label-keywords present, pick the highest. Ties broken
by 'dissectable=true' first (we prefer dissection-capable stubs).
"""
import json
import re
import shutil
import time
from pathlib import Path
from typing import Iterable

_LIB_ROOT = Path(__file__).resolve().parent.parent.parent / "stub_objects"
_LIB_INDEX_PATH = _LIB_ROOT / "library.json"

_LIBRARY: list[dict] | None = None


def _lazy_load() -> list[dict]:
    global _LIBRARY
    if _LIBRARY is None:
        if not _LIB_INDEX_PATH.exists():
            raise RuntimeError(f"Stub object library missing at {_LIB_INDEX_PATH}")
        _LIBRARY = json.loads(_LIB_INDEX_PATH.read_text())["objects"]
    return _LIBRARY


_WORD_SPLIT_RE = re.compile(r"[^a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    return {t for t in _WORD_SPLIT_RE.split(text.lower()) if t}


def _score(prompt_tokens: set[str], labels: Iterable[str]) -> int:
    label_set = {la.lower() for la in labels}
    return len(prompt_tokens & label_set)


def best_match(image_prompt: str) -> dict:
    """Return the library entry with the highest label-overlap score.

    Tie-break: prefer dissectable=true, then alphabetic id.
    Worst-case fallback: first entry in the library.
    """
    lib = _lazy_load()
    if not lib:
        raise RuntimeError("Stub library is empty")
    tokens = _tokenize(image_prompt)

    scored = [
        (_score(tokens, entry["labels"]), 0 if entry.get("dissectable") else 1, entry["id"], entry)
        for entry in lib
    ]
    # Highest score wins; then dissectable; then alphabetic for stable ordering
    scored.sort(key=lambda t: (-t[0], t[1], t[2]))
    return scored[0][3]


def stub_object_gen(
    image_prompt: str,
    exhibit_id: str,
    object_id: str,
    data_dir: Path | str,
) -> dict:
    """Copy the best-matching stub GLB into the exhibit's objects dir.

    Returns the response shape expected by routing.py:
        {"image_url": None, "glb_url": str, "elapsed_ms": int, "stub_id": str}
    """
    t0 = time.perf_counter()
    data_dir = Path(data_dir)
    entry = best_match(image_prompt)

    src_glb = _LIB_ROOT / entry["glb_path"]
    if not src_glb.exists():
        raise RuntimeError(f"Stub GLB not on disk: {src_glb}")

    out_dir = data_dir / "exhibits" / exhibit_id / "objects"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{object_id}.glb"
    shutil.copyfile(src_glb, out_path)

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    return {
        "image_url": None,
        "glb_url": f"/exhibits/{exhibit_id}/objects/{object_id}.glb",
        "elapsed_ms": elapsed_ms,
        "stub_id": entry["id"],
    }


def hints_for_object(object_label: str) -> dict:
    """Return per-object hints (rotation, scale, clip axis) from the stub library.

    Used by routing.py to populate ObjectSpec fields when in stub mode, so the
    placed object matches what was actually generated.
    """
    entry = best_match(object_label)
    return {
        "dissectable": entry.get("dissectable", True),
        "preferred_clip_axis": entry.get("preferred_clip_axis", "y"),
        "preferred_clip_t": entry.get("preferred_clip_t", 0.5),
        "rotation_y_deg": entry.get("rotation_y_deg", 0),
        "scale_hint": entry.get("scale_hint", 0.1),
    }
