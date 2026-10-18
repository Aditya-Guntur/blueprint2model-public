"""Stage 5: re-skeletonize after Hough-line merging.

Stage 4 redraws merged wall segments as thick (2px) lines, so the
skeleton has to be recomputed to get back to clean 1px-wide
centerlines before the Blender mesh stage reads wall paths off it.
"""

import cv2
import numpy as np
from skimage.morphology import skeletonize


def skel_maker(merged_mask_path: str, out_path: str = "final_skeleton.png") -> str:
    image = cv2.imread(merged_mask_path)
    if image is None:
        raise FileNotFoundError(f"Could not read mask: {merged_mask_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    skeleton = skeletonize(gray > 0)
    cv2.imwrite(out_path, (skeleton.astype(np.uint8)) * 255)
    return out_path
