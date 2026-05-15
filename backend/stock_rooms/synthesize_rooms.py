"""
Synthesize stock room GLB assets using trimesh primitives.
All rooms are placeholder geometry — correct interior geometry for v1 launch.
Run with: conda run -n museums --no-capture-output python synthesize_rooms.py
"""

import json
import os
import numpy as np
import trimesh
from trimesh.transformations import translation_matrix

OUTPUT_DIR = "/mnt/3D_Mapping/backend/stock_rooms/rooms"
MANIFEST_PATH = "/mnt/3D_Mapping/backend/stock_rooms/manifest.json"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def make_room_scene(
    width: float,
    height: float,
    depth: float,
    floor_color: list,
    wall_color: list,
    ceiling_color: list,
    has_window_slots: bool = False,
) -> trimesh.Scene:
    """
    Build a simple room as a collection of 6 thin box panels:
    floor, ceiling, left wall, right wall, back wall, front wall (open/thin).
    Y-up convention. Room is centered at origin horizontally, floor at Y=0.
    """
    thickness = 0.05  # panel thickness in metres

    # Floor
    floor = trimesh.creation.box(extents=[width, thickness, depth])
    floor.apply_translation([0, thickness / 2, 0])
    floor.visual.face_colors = floor_color + [255]

    # Ceiling
    ceiling = trimesh.creation.box(extents=[width, thickness, depth])
    ceiling.apply_translation([0, height - thickness / 2, 0])
    ceiling.visual.face_colors = ceiling_color + [255]

    # Back wall (at Z = -depth/2)
    back_wall = trimesh.creation.box(extents=[width, height, thickness])
    back_wall.apply_translation([0, height / 2, -depth / 2 + thickness / 2])
    back_wall.visual.face_colors = wall_color + [255]

    # Left wall (at X = -width/2)
    left_wall = trimesh.creation.box(extents=[thickness, height, depth])
    left_wall.apply_translation([-width / 2 + thickness / 2, height / 2, 0])
    left_wall.visual.face_colors = wall_color + [255]

    # Right wall (at X = +width/2)
    right_wall = trimesh.creation.box(extents=[thickness, height, depth])
    right_wall.apply_translation([width / 2 - thickness / 2, height / 2, 0])
    right_wall.visual.face_colors = wall_color + [255]

    # Front wall (partially open — thin strip at top so room feels enclosed from camera)
    front_top = trimesh.creation.box(extents=[width, height * 0.2, thickness])
    front_top.apply_translation([0, height - height * 0.1, depth / 2 - thickness / 2])
    front_top.visual.face_colors = wall_color + [255]

    scene = trimesh.Scene()
    scene.add_geometry(floor, node_name="floor")
    scene.add_geometry(ceiling, node_name="ceiling")
    scene.add_geometry(back_wall, node_name="back_wall")
    scene.add_geometry(left_wall, node_name="left_wall")
    scene.add_geometry(right_wall, node_name="right_wall")
    scene.add_geometry(front_top, node_name="front_top")

    return scene


def compute_room_frame(glb_path: str) -> dict:
    """Load a GLB and compute RoomFrame: floor_y, bbox, origin, extent."""
    scene = trimesh.load(glb_path, force="scene")

    # Collect all vertices from all geometries
    all_verts = []
    if hasattr(scene, "geometry"):
        for geom in scene.geometry.values():
            if hasattr(geom, "vertices") and len(geom.vertices) > 0:
                # Apply node transform if available
                all_verts.append(geom.vertices)

    if not all_verts:
        # Fallback
        return {
            "origin": [0.0, 0.0, 0.0],
            "basis_x": [1.0, 0.0, 0.0],
            "basis_y": [0.0, 1.0, 0.0],
            "basis_z": [0.0, 0.0, 1.0],
            "extent": [6.0, 3.0, 6.0],
        }

    verts = np.vstack(all_verts)

    # Determine up-axis: the axis with minimum variance is likely the vertical
    # (floor/ceiling are flat planes). For synthesized rooms Y-up is correct.
    variances = np.var(verts, axis=0)
    # Y-up assumption (our synthesized rooms are Y-up)
    y_coords = verts[:, 1]

    # Floor: bottom 10% by Y
    threshold = np.percentile(y_coords, 10)
    floor_verts = verts[y_coords <= threshold]
    floor_y = float(np.mean(floor_verts[:, 1]))

    # AABB
    mins = verts.min(axis=0)
    maxs = verts.max(axis=0)

    center_x = float((mins[0] + maxs[0]) / 2)
    center_z = float((mins[2] + maxs[2]) / 2)

    extent_x = float(maxs[0] - mins[0])
    extent_y = float(maxs[1] - mins[1])
    extent_z = float(maxs[2] - mins[2])

    return {
        "origin": [round(center_x, 4), round(floor_y, 4), round(center_z, 4)],
        "basis_x": [1.0, 0.0, 0.0],
        "basis_y": [0.0, 1.0, 0.0],
        "basis_z": [0.0, 0.0, 1.0],
        "extent": [round(extent_x, 4), round(extent_y, 4), round(extent_z, 4)],
    }


# Room definitions: (id, theme, description, width, height, depth, floor_rgb, wall_rgb, ceiling_rgb, themes_list)
ROOMS = [
    {
        "id": "victorian_study_001",
        "name": "Victorian Study",
        "themes": ["historical", "study", "warm-light", "interior", "victorian"],
        "width": 6.0, "height": 3.2, "depth": 6.0,
        # warm wood-brown floor, cream walls, off-white ceiling
        "floor_color": [101, 67, 33],
        "wall_color": [210, 195, 170],
        "ceiling_color": [235, 228, 215],
    },
    {
        "id": "library_warm_001",
        "name": "Warm Library",
        "themes": ["library", "warm-light", "interior", "books", "reading"],
        "width": 8.0, "height": 4.0, "depth": 10.0,
        # dark oak floor, warm ivory walls, cream ceiling
        "floor_color": [80, 50, 25],
        "wall_color": [220, 205, 180],
        "ceiling_color": [240, 235, 220],
    },
    {
        "id": "workshop_industrial_001",
        "name": "Industrial Workshop",
        "themes": ["workshop", "industrial", "interior", "mechanics", "tools"],
        "width": 10.0, "height": 4.5, "depth": 12.0,
        # concrete grey floor, steel-grey walls, bare ceiling
        "floor_color": [130, 130, 130],
        "wall_color": [160, 160, 155],
        "ceiling_color": [180, 180, 180],
    },
    {
        "id": "gallery_white_001",
        "name": "White Cube Gallery",
        "themes": ["gallery", "modern", "interior", "neutral", "art"],
        "width": 12.0, "height": 4.0, "depth": 15.0,
        # white floor, white walls, white ceiling
        "floor_color": [245, 245, 245],
        "wall_color": [250, 250, 250],
        "ceiling_color": [255, 255, 255],
    },
    {
        "id": "classroom_neutral_001",
        "name": "Neutral Classroom",
        "themes": ["classroom", "educational", "interior", "neutral"],
        "width": 9.0, "height": 3.2, "depth": 10.0,
        # light wood floor, pale green walls, white ceiling
        "floor_color": [185, 155, 110],
        "wall_color": [200, 215, 195],
        "ceiling_color": [245, 245, 240],
    },
    {
        "id": "laboratory_clinical_001",
        "name": "Clinical Laboratory",
        "themes": ["laboratory", "clinical", "interior", "science", "modern"],
        "width": 8.0, "height": 3.0, "depth": 10.0,
        # white tile floor, clinical white walls, white ceiling
        "floor_color": [230, 232, 235],
        "wall_color": [240, 242, 245],
        "ceiling_color": [250, 252, 255],
    },
    {
        "id": "kitchen_period_001",
        "name": "Period Kitchen",
        "themes": ["kitchen", "period", "interior", "cooking", "domestic"],
        "width": 5.0, "height": 2.8, "depth": 6.0,
        # terracotta floor, warm ochre walls, warm white ceiling
        "floor_color": [185, 100, 65],
        "wall_color": [215, 195, 145],
        "ceiling_color": [240, 232, 210],
    },
    {
        "id": "conservatory_botanic_001",
        "name": "Botanic Conservatory",
        "themes": ["conservatory", "botanic", "interior", "greenhouse", "nature"],
        "width": 10.0, "height": 5.0, "depth": 14.0,
        # stone tile floor, pale grey-green walls, glass-like ceiling
        "floor_color": [160, 155, 140],
        "wall_color": [190, 205, 185],
        "ceiling_color": [220, 235, 225],
    },
    {
        "id": "attic_dim_001",
        "name": "Dusty Attic",
        "themes": ["attic", "curiosities", "interior", "dim", "historical"],
        "width": 7.0, "height": 2.4, "depth": 8.0,
        # old wood floor, aged timber walls, dark ceiling
        "floor_color": [95, 70, 45],
        "wall_color": [120, 100, 75],
        "ceiling_color": [80, 65, 50],
    },
    {
        "id": "pavilion_outdoor_001",
        "name": "Outdoor Pavilion",
        "themes": ["pavilion", "outdoor", "garden", "open", "nature"],
        "width": 8.0, "height": 3.5, "depth": 8.0,
        # stone floor, light stone walls, open sky ceiling (pale blue)
        "floor_color": [170, 165, 155],
        "wall_color": [190, 185, 175],
        "ceiling_color": [180, 205, 235],
    },
]


manifest = []

for room_def in ROOMS:
    room_id = room_def["id"]
    glb_path = os.path.join(OUTPUT_DIR, f"{room_id}.glb")

    print(f"Synthesizing {room_id}...")

    scene = make_room_scene(
        width=room_def["width"],
        height=room_def["height"],
        depth=room_def["depth"],
        floor_color=room_def["floor_color"],
        wall_color=room_def["wall_color"],
        ceiling_color=room_def["ceiling_color"],
    )

    # Export GLB
    glb_bytes = scene.export(file_type="glb")
    with open(glb_path, "wb") as f:
        f.write(glb_bytes)

    size_bytes = len(glb_bytes)
    print(f"  Written {size_bytes:,} bytes -> {glb_path}")

    # Compute RoomFrame
    room_frame = compute_room_frame(glb_path)
    print(f"  RoomFrame: origin={room_frame['origin']}, extent={room_frame['extent']}")

    manifest.append({
        "id": room_id,
        "themes": room_def["themes"],
        "style": "placeholder",
        "credit": f"Synthesized primitive room — trimesh box geometry (placeholder for '{room_def['name']}')",
        "license": "CC0",
        "source_url": "",
        "glb_path": f"rooms/{room_id}.glb",
        "size_bytes": size_bytes,
        "room_frame": room_frame,
    })

# Write manifest
with open(MANIFEST_PATH, "w") as f:
    json.dump(manifest, f, indent=2)

print(f"\nManifest written to {MANIFEST_PATH}")
print(f"Total rooms: {len(manifest)}")
