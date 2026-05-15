"""Production object_gen — calls Modal FLUX → TRELLIS.2 chain.

Currently a stub-of-a-stub: the real Modal function isn't deployed in v1
(per the budget-pivot decision in SPIKE_RESULT.md). When you eventually
deploy modal_pkg/object_gen.py and have Modal credit, this wrapper takes
over from stub_object_gen.

Flip the switch: set OBJECT_GEN_MODE=modal in backend/.env.
"""
from typing import Any


def modal_object_gen(image_prompt: str, exhibit_id: str, object_id: str, **kwargs: Any) -> dict:
    """Real Modal call — NOT IMPLEMENTED in v1.

    When the project owner has Modal budget and a deployed object_gen function,
    replace this body with:

        import modal
        from ..config import get_settings

        _fn = modal.Function.lookup(get_settings().modal_app_name, "object_gen")
        result = _fn.remote(image_prompt, exhibit_id, object_id)
        return result
    """
    raise NotImplementedError(
        "Modal object_gen not deployed in v1 (budget-pivot — see SPIKE_RESULT.md). "
        "Switch OBJECT_GEN_MODE=stub in backend/.env to use the stub library, "
        "or deploy modal_pkg/object_gen.py and uncomment the modal.Function.lookup "
        "block in this file."
    )
