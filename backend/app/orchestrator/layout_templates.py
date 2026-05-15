"""Layout templates for placing pedestals + spline waypoints.

Coordinates are NORMALIZED [0, 1] room-local. Backend transforms to world later.
Eye height (y) is fixed at 0.55 (approx 1.7m on a 3m-tall room).
"""
from typing import Literal

LayoutName = Literal["linear", "hub_and_spokes", "semicircle"]


def describe(name: LayoutName, n_objects: int) -> str:
    """One-line description for inclusion in the orchestrator system prompt."""
    if name == "linear":
        return (
            f"linear gallery: {n_objects} pedestals in a row at z=0.2, "
            f"x-spacing 0.18, centered horizontally. Spline starts at "
            f"(0.5, 0.55, 0.85) and walks past each pedestal."
        )
    if name == "hub_and_spokes":
        return (
            f"hub-and-spokes: {n_objects} pedestals at radius 0.35 from "
            f"room center, evenly angled. Spline starts at (0.5, 0.55, 0.85) "
            f"and visits each via the center."
        )
    if name == "semicircle":
        return (
            f"semicircle: {n_objects} pedestals on a 0.35-radius arc on the "
            f"far half of the room (z < 0.5), facing the viewer. Spline at "
            f"radius 0.15 walks past each."
        )
    raise ValueError(name)
