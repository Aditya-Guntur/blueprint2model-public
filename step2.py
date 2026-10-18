"""Stage 2: close small gaps in the wall mask.

Raster-to-mask thresholding often leaves 1-2px holes where wall lines
cross door/window symbols or get anti-aliased. A morphological close
(a convolution with a small structuring element) bridges those gaps
before skeletonization.
"""

import cv2


def gap_filler(mask_path: str, out_path: str = "clean_mask.png") -> str:
    mask = cv2.imread(mask_path)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {mask_path}")

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    cv2.imwrite(out_path, clean)
    return out_path
