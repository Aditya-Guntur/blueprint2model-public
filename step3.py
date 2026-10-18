"""Stage 3: skeletonize the clean mask into 1px-wide wall centerlines.

Wall thickness varies across a raster floor plan; Hough line detection
in stage 4 works far better on a thin centerline than on a filled
region, so we skeletonize before line extraction.
"""

import cv2
import numpy as np
from skimage.morphology import skeletonize


def skeleton_maker(clean_mask_path: str, out_path: str = "first_skeleton.png") -> str:
    mask = cv2.imread(clean_mask_path)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {clean_mask_path}")

    skeleton = skeletonize(mask > 0)
    cv2.imwrite(out_path, (skeleton.astype(np.uint8)) * 255)
    return out_path
