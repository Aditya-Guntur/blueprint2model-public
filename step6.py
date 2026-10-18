"""Stage 6: build a 3D wall mesh from stage 4's merged wall segments.

This stage runs *inside* Blender's embedded Python, not the system
interpreter - it imports bpy/bmesh, which only exist in that
environment (and are not installable with pip):

    blender --python step6.py -- <path-to-final_skeleton.png>

What it does:
1. Clears the default scene (cube, camera, light) so only the
   generated walls remain.
2. Loads wall_segments.json (written by stage 4, in the same
   directory as the skeleton image) - the exact merged axis-aligned
   wall vectors, as data, not an image to be re-parsed.
3. Converts each segment's pixel coordinates to Blender world
   coordinates (flips the Y axis, since image-space Y grows downward
   and Blender's grows upward).
4. Extrudes each segment into a solid mesh with real height and
   thickness, using bmesh.

An earlier version re-derived wall paths by running
cv2.findContours + approxPolyDP on the skeleton image instead of using
stage 4's segment data directly. That looked reasonable in every 2D
preview image, but cv2.findContours is built to trace the boundary of
a filled region, not a 1px-wide open skeleton line: on a thin line it
walks up one side and back down the other, and approxPolyDP
simplification could then insert a spurious diagonal shortcut between
that "there" and "back" traversal at a branch point - invisible in any
of the flat 2D stage images, but very visible as an extra diagonal
wall once extruded into 3D. Consuming stage 4's segments directly
removes the raster round-trip (and the cv2 dependency inside Blender)
entirely, instead of trying to patch around what findContours does
with thin lines.

A second wall-building approach (create_walls_with_curves) builds the
same segments as beveled curves instead of raw meshes, which bends
more cleanly around corners - left in as an alternative, not wired up
by default.
"""

import json
import math
import os
import sys

import bpy
import bmesh

args = sys.argv[sys.argv.index("--") + 1:]
if not args:
    print("Usage: blender --python step6.py -- <skeleton_image>")
    sys.exit(1)

skel_path = args[0]

# Purge any previously loaded image so Blender doesn't reuse a stale plane
for img in list(bpy.data.images):
    bpy.data.images.remove(img)


def clear_scene():
    """Remove every object from the current scene."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False, confirm=False)


def load_wall_segments(skeleton_image_path):
    """Load stage 4's merged wall segments from wall_segments.json,
    which lives next to the skeleton image. Returns
    (image_height, [(x1, y1, x2, y2), ...])."""
    segments_path = os.path.join(os.path.dirname(os.path.abspath(skeleton_image_path)), "wall_segments.json")
    if not os.path.isfile(segments_path):
        raise FileNotFoundError(
            f"No wall_segments.json next to {skeleton_image_path} - "
            "run stage 4 (step4.extract_and_merge_segments) first."
        )
    with open(segments_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["image_height"], [tuple(seg) for seg in data["segments"]]


def pixel_to_blender_coords(pixel_coords, image_height, scale_factor=0.1):
    """Map (x, y) pixel coordinates to (x, y, z) Blender world coordinates."""
    blender_coords = []
    for x, y in pixel_coords:
        blender_x = x * scale_factor
        blender_y = (image_height - y) * scale_factor  # flip Y: image grows down, Blender grows up
        blender_coords.append((blender_x, blender_y, 0))
    return blender_coords


def create_wall_from_path(path, wall_height=3.0, wall_thickness=0.2, name="Wall"):
    """Build one rectangular box per edge of the path.

    An earlier version built a single zero-thickness ribbon down the
    whole path and tried to give it depth with inset_region. That
    insets *within* the face's own plane, so the wall stayed paper-thin
    and was invisible edge-on from directly above - confirmed by
    actually rendering it in Blender: a top-down shot came back a flat
    uniform gray with zero geometry visible, because every wall quad's
    surface normal was horizontal and the camera was looking straight
    down it. Swapping inset_region for bmesh.ops.solidify was the
    obvious next fix, but it extrudes along face normals assuming a
    continuous shell - a path with sharp turns isn't one, and the
    miter math at each turn blew up into long spikes reaching off the
    edge of the render.

    Building each edge as its own independent box sidesteps both
    problems: no miter join, no shared-normal assumption, at the cost
    of small gaps/overlaps at corners, which don't matter at
    floor-plan scale.
    """
    if len(path) < 2:
        return None

    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)

    bm = bmesh.new()
    half_thickness = wall_thickness / 2

    for (x1, y1, z1), (x2, y2, z2) in zip(path, path[1:]):
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length == 0:
            continue
        # unit vector perpendicular to the segment, in the XY plane
        nx, ny = -dy / length, dx / length

        bottom = [
            (x1 + nx * half_thickness, y1 + ny * half_thickness, z1),
            (x2 + nx * half_thickness, y2 + ny * half_thickness, z2),
            (x2 - nx * half_thickness, y2 - ny * half_thickness, z2),
            (x1 - nx * half_thickness, y1 - ny * half_thickness, z1),
        ]
        top = [(x, y, z + wall_height) for x, y, z in bottom]

        bv = [bm.verts.new(c) for c in bottom]
        tv = [bm.verts.new(c) for c in top]

        bm.faces.new([bv[0], bv[1], bv[2], bv[3]])  # floor
        bm.faces.new([tv[3], tv[2], tv[1], tv[0]])  # ceiling
        for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)):  # four side walls
            bm.faces.new([bv[a], bv[b], tv[b], tv[a]])

    bm.normal_update()
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-4)

    bm.to_mesh(mesh)
    bm.free()
    return obj


def create_walls_from_skeleton(skeleton_image_path, wall_height=3.0, wall_thickness=0.2, scale_factor=0.1):
    """Build one wall mesh object per merged segment in wall_segments.json."""
    clear_scene()

    image_height, segments = load_wall_segments(skeleton_image_path)
    print(f"Loaded {len(segments)} wall segments")

    for i, (x1, y1, x2, y2) in enumerate(segments):
        blender_path = pixel_to_blender_coords([(x1, y1), (x2, y2)], image_height, scale_factor)
        wall_name = f"Wall_{i + 1:03d}"
        wall_obj = create_wall_from_path(blender_path, wall_height, wall_thickness, wall_name)
        if wall_obj:
            print(f"Created {wall_name}")


def create_walls_with_curves(skeleton_image_path, wall_height=3.0, wall_thickness=0.2, scale_factor=0.1):
    """Alternative: build walls as beveled curves instead of raw meshes.

    Not called by default - swap it in below if you want rounder
    corners instead of the box builder's mitred joints.
    """
    clear_scene()

    image_height, segments = load_wall_segments(skeleton_image_path)

    for i, (x1, y1, x2, y2) in enumerate(segments):
        blender_path = pixel_to_blender_coords([(x1, y1), (x2, y2)], image_height, scale_factor)

        curve_data = bpy.data.curves.new(f"WallCurve_{i + 1:03d}", "CURVE")
        curve_data.dimensions = "3D"

        spline = curve_data.splines.new("POLY")
        spline.points.add(len(blender_path) - 1)
        for j, point in enumerate(blender_path):
            spline.points[j].co = (point[0], point[1], point[2], 1.0)

        curve_obj = bpy.data.objects.new(f"Wall_{i + 1:03d}", curve_data)
        bpy.context.collection.objects.link(curve_obj)

        curve_data.bevel_depth = wall_thickness / 2
        curve_data.bevel_resolution = 2
        curve_data.extrude = wall_height


if __name__ == "__main__":
    create_walls_from_skeleton(skel_path, wall_height=3.0, wall_thickness=0.2, scale_factor=0.05)
