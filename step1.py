"""Stage 1: build a binary wall mask from the floor-plan raster.

Converts the input image to grayscale and thresholds dark pixels (wall
lines in most floor-plan exports are near-black) into a binary mask.
"""

import cv2
import numpy as np

DARK_THRESHOLD = 40


def create_mask(image_path: str, out_path: str = "mask.png") -> str:
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    mask = np.zeros((height, width), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            if gray[y, x] < DARK_THRESHOLD:
                mask[y, x] = 255

    cv2.imwrite(out_path, mask)
    return out_path
