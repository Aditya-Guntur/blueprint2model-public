"""Stage 6: build a 3D wall mesh from the final wall skeleton.

This stage runs *inside* Blender's embedded Python, not the system
interpreter - it imports bpy/bmesh, which only exist in that
environment (and are not installable with pip):

    blender --python step6.py -- <path-to-final_skeleton.png>

What it does:
1. Clears the default scene (cube, camera, light) so only the
   generated walls remain.
2. Finds wall paths as contours in the skeleton image.
3. Converts pixel coordinates to Blender world coordinates (flips the
   Y axis, since image-space Y grows downward and Blender's grows
   upward).
4. Extrudes each wall path into a solid mesh with real height and
   thickness, using bmesh.

A second approach (create_walls_with_curves) builds the same walls as
beveled curves instead of raw meshes, which bends more cleanly around
corners - left in as an alternative, not wired up by default.
"""

import sys

import bpy

args = sys.argv[sys.argv.index("--") + 1:]
if not args:
    print("Usage: blender --python step6.py -- <skeleton_image>")
    sys.exit(1)

skel_path = args[0]

# Purge any previously loaded image so Blender doesn't reuse a stale plane
for img in list(bpy.data.images):
    bpy.data.images.remove(img)

import math

import cv2
import bmesh


def clear_scene():
    """Remove every object from the current scene."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False, confirm=False)


def extract_wall_paths(skeleton_image_path):
    """Find wall segments as simplified contours in the skeleton image."""
    skel_img = cv2.imread(skeleton_image_path, cv2.IMREAD_GRAYSCALE)
    contours, _ = cv2.findContours(skel_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    wall_paths = []
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        simplified = cv2.approxPolyDP(contour, epsilon, False)
        points = [tuple(point[0]) for point in simplified]
        if len(points) > 1:
            wall_paths.append(points)
    return wall_paths


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

    The previous version built a single zero-thickness ribbon down the
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
    """Build a wall mesh object per wall path found in the skeleton."""
    clear_scene()

    skel_img = cv2.imread(skeleton_image_path, cv2.IMREAD_GRAYSCALE)
    if skel_img is None:
        print(f"Could not load image: {skeleton_image_path}")
        return

    image_height = skel_img.shape[0]
    wall_paths = extract_wall_paths(skeleton_image_path)
    print(f"Found {len(wall_paths)} wall segments")

    for i, wpath in enumerate(wall_paths):
        blender_path = pixel_to_blender_coords(wpath, image_height, scale_factor)
        wall_name = f"Wall_{i + 1:03d}"
        wall_obj = create_wall_from_path(blender_path, wall_height, wall_thickness, wall_name)
        if wall_obj:
            print(f"Created {wall_name} with {len(wpath)} points")


if __name__ == "__main__":
    create_walls_from_skeleton(skel_path, wall_height=3.0, wall_thickness=0.2, scale_factor=0.05)


def create_walls_with_curves(skeleton_image_path, wall_height=3.0, wall_thickness=0.2, scale_factor=0.1):
    """Alternative: build walls as beveled curves instead of raw meshes.

    Not called by default - swap it in below if you want rounder
    corners instead of the box builder's mitred joints.
    """
    clear_scene()

    skel_img = cv2.imread(skeleton_image_path, cv2.IMREAD_GRAYSCALE)
    image_height = skel_img.shape[0]
    wall_paths = extract_wall_paths(skeleton_image_path)

    for i, wpath in enumerate(wall_paths):
        if len(wpath) < 2:
            continue

        blender_path = pixel_to_blender_coords(wpath, image_height, scale_factor)

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
