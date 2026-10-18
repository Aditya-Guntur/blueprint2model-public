"""Stage 6: build a 3D wall mesh from the final wall skeleton.

This stage runs *inside* Blender's embedded Python, not the system
interpreter - it imports bpy/bmesh, which only exist in that
environment (and are not installable with pip):

    blender --python step6.py -- <path-to-final_skeleton.png>

So far: find wall paths as contours in the skeleton image.
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
    """Extrude a path of points into a quad-faced wall mesh."""
    if len(path) < 2:
        return None

    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)

    bm = bmesh.new()
    bottom_verts = [bm.verts.new((p[0], p[1], p[2])) for p in path]
    top_verts = [bm.verts.new((p[0], p[1], p[2] + wall_height)) for p in path]
    bm.verts.ensure_lookup_table()

    for i in range(len(path) - 1):
        bm.faces.new([bottom_verts[i], bottom_verts[i + 1], top_verts[i + 1], top_verts[i]])

    if wall_thickness > 0:
        bmesh.ops.inset_region(bm, faces=bm.faces[:], thickness=wall_thickness / 2)

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
