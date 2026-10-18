"""Build walls from a skeleton image and render a top-down preview PNG.

Run with:
    blender --background --python render_preview.py -- <skeleton_image> <output.png>

Kept separate from step6.py's create_walls_from_skeleton() so the wall
builder can also be driven interactively inside the Blender UI without
a render forced on top of it.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bpy
from step6 import create_walls_from_skeleton

args = sys.argv[sys.argv.index("--") + 1:]
if len(args) < 2:
    print("Usage: blender --background --python render_preview.py -- <skeleton_image> <output.png>")
    sys.exit(1)

skel_path, out_path = args[0], args[1]

create_walls_from_skeleton(skel_path, wall_height=3.0, wall_thickness=0.2, scale_factor=0.05)

bpy.ops.object.light_add(type="SUN", location=(0, 0, 20))
bpy.context.object.data.energy = 3

mesh_objs = [o for o in bpy.data.objects if o.type == "MESH"]
if mesh_objs:
    xs = [o.matrix_world @ v.co for o in mesh_objs for v in o.data.vertices]
    cx = sum(v.x for v in xs) / len(xs)
    cy = sum(v.y for v in xs) / len(xs)
    span = max(max(v.x for v in xs) - min(v.x for v in xs), max(v.y for v in xs) - min(v.y for v in xs), 1)
else:
    cx, cy, span = 0, 0, 10

bpy.ops.object.camera_add(location=(cx, cy, span * 1.5))
camera = bpy.context.object
camera.rotation_euler = (0, 0, 0)
camera.data.type = "ORTHO"
camera.data.ortho_scale = span * 1.3
bpy.context.scene.camera = camera

scene = bpy.context.scene
scene.render.resolution_x = 800
scene.render.resolution_y = 800
scene.render.filepath = out_path
scene.render.image_settings.file_format = "PNG"
bpy.ops.render.render(write_still=True)
print(f"Rendered preview to {out_path}")
