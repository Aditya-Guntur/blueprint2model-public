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
