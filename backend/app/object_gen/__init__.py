"""Object generation dispatcher.

Routes by settings.object_gen_mode:
- 'stub': returns pre-vendored CC0 GLB by label-fuzzy-match ($0 cost).
- 'modal': calls the production Modal FLUX -> TRELLIS.2 chain (not deployed in v1).
"""
from pathlib import Path
from typing import Any

from ..config import get_settings
from .modal_client import modal_object_gen
from .stub import hints_for_object, stub_object_gen


def generate(
    image_prompt: str,
    exhibit_id: str,
    object_id: str,
    data_dir: str | Path,
) -> dict:
    """Generate or fetch an object GLB. Returns the routing-friendly response shape."""
    settings = get_settings()
    if settings.object_gen_mode == "stub":
        return stub_object_gen(image_prompt, exhibit_id, object_id, data_dir)
    return modal_object_gen(image_prompt, exhibit_id, object_id, data_dir=data_dir)


__all__ = ["generate", "hints_for_object"]
