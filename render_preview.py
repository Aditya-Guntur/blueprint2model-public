"""Build walls from a skeleton image and render a preview PNG.

Run with:
    blender --background --python render_preview.py -- <skeleton_image> <output.png> [--top]

Renders an angled view by default. A straight top-down orthographic
shot makes any 3D wall structure look identical to a flat 2D line
diagram, since you can't see height from directly above - pass --top
if that's genuinely what you want (e.g. to compare against the
step4/step5 2D outputs).

Kept separate from step6.py's create_walls_from_skeleton() so the wall
builder can also be driven interactively inside the Blender UI without
a render forced on top of it.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bpy
import mathutils
from step6 import create_walls_from_skeleton

args = sys.argv[sys.argv.index("--") + 1:]
if len(args) < 2:
    print("Usage: blender --background --python render_preview.py -- <skeleton_image> <output.png> [--top]")
    sys.exit(1)

skel_path, out_path = args[0], args[1]
top_down = "--top" in args[2:]

create_walls_from_skeleton(skel_path, wall_height=3.0, wall_thickness=0.2, scale_factor=0.05)

bpy.ops.object.light_add(type="SUN", location=(10, -10, 20))
bpy.context.object.data.energy = 3
if not top_down:
    bpy.context.object.rotation_euler = (0.6, 0.2, 0.8)

mesh_objs = [o for o in bpy.data.objects if o.type == "MESH"]
if mesh_objs:
    xs = [o.matrix_world @ v.co for o in mesh_objs for v in o.data.vertices]
    cx = sum(v.x for v in xs) / len(xs)
    cy = sum(v.y for v in xs) / len(xs)
    span = max(max(v.x for v in xs) - min(v.x for v in xs), max(v.y for v in xs) - min(v.y for v in xs), 1)
else:
    cx, cy, span = 0, 0, 10

if top_down:
    cam_pos = mathutils.Vector((cx, cy, span * 1.5))
    rotation = (0, 0, 0)
else:
    cam_pos = mathutils.Vector((cx - span * 0.9, cy - span * 1.1, span * 0.9))
    target = mathutils.Vector((cx, cy, 1.0))
    rotation = (target - cam_pos).to_track_quat("-Z", "Y").to_euler()

bpy.ops.object.camera_add(location=cam_pos)
camera = bpy.context.object
camera.rotation_euler = rotation
camera.data.type = "ORTHO"
camera.data.ortho_scale = span * (1.3 if top_down else 1.6)
bpy.context.scene.camera = camera

scene = bpy.context.scene
scene.render.resolution_x = 800 if top_down else 900
scene.render.resolution_y = 800 if top_down else 700
scene.render.filepath = out_path
scene.render.image_settings.file_format = "PNG"
bpy.ops.render.render(write_still=True)
print(f"Rendered preview to {out_path}")
