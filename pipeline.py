"""CLI entry point for the blueprint-to-3D-model pipeline.

Takes a raster floor-plan image and runs it through mask extraction,
gap filling, skeletonization, Hough-line merging and re-skeletonization
(stages 1-5, plain Python/OpenCV), then optionally hands the final
wall skeleton to Blender (stage 6) to build the 3D wall mesh.

Usage:
    python pipeline.py examples/simple_blueprint.png
    python pipeline.py examples/simple_blueprint.png --skip-blender
    python pipeline.py examples/simple_blueprint.png --blender /path/to/blender
"""

import argparse
import os
import subprocess
import sys

from step1 import create_mask
from step2 import gap_filler
from step3 import skeleton_maker
from step4 import extract_and_merge_segments
from step5 import skel_maker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a 2D floor-plan image into a Blender wall mesh."
    )
    parser.add_argument("image", help="Path to the input floor-plan image (PNG/JPG).")
    parser.add_argument(
        "-o", "--output-dir", default="output",
        help="Directory to write intermediate masks/skeletons into (default: ./output).",
    )
    parser.add_argument(
        "--blender", default="blender",
        help="Path to the Blender executable (default: 'blender' on PATH).",
    )
    parser.add_argument(
        "--skip-blender", action="store_true",
        help="Run only stages 1-5 (mask through final skeleton) and stop before Blender.",
    )
    return parser.parse_args()


def run_image_pipeline(image_path: str, output_dir: str) -> str:
    """Run stages 1-5 and return the path to the final wall skeleton."""
    os.makedirs(output_dir, exist_ok=True)

    mask_path = create_mask(image_path, os.path.join(output_dir, "mask.png"))
    clean_mask = gap_filler(mask_path, os.path.join(output_dir, "clean_mask.png"))
    skeleton = skeleton_maker(clean_mask, os.path.join(output_dir, "first_skeleton.png"))
    merged = extract_and_merge_segments(skeleton, os.path.join(output_dir, "extended_parallel.png"))
    final_skeleton = skel_maker(merged, os.path.join(output_dir, "final_skeleton.png"))

    return final_skeleton


def run_blender_stage(final_skeleton: str, blender_exe: str) -> None:
    render_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "render_preview.py")
    preview_path = os.path.join(os.path.dirname(final_skeleton), "preview.png")
    cmd = [blender_exe, "--background", "--python", render_script, "--",
           os.path.abspath(final_skeleton), os.path.abspath(preview_path)]
    print(f"Launching Blender: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"Blender finished - wall mesh built, preview rendered to {preview_path}")


def main() -> None:
    args = parse_args()
    image_path = os.path.normpath(args.image)

    if not os.path.isfile(image_path):
        print(f"Input image not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    final_skeleton = run_image_pipeline(image_path, args.output_dir)
    print(f"Final wall skeleton written to {final_skeleton}")

    if args.skip_blender:
        return

    run_blender_stage(final_skeleton, args.blender)


if __name__ == "__main__":
    main()
