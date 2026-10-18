"""Stage 4: detect and extend wall segments in the skeleton.

Skeletonization leaves a 1px centerline, but it's rarely one clean
straight run per wall - corners and intersections break it into many
short segments. This stage pulls those segments out with a
probabilistic Hough transform, then extends each one slightly at both
ends to close the 1px gaps corners leave behind.
"""

import math
import os

import cv2
import numpy as np

HOUGH_THRESH = 50
MIN_LINE_LEN = 2
MAX_LINE_GAP = 10
EXT_PIXELS = 1  # pixels to extend each line at both ends


def extract_and_merge_segments(skel_path: str, out_path: str = "extended_parallel.png") -> str:
    skel_bgr = cv2.imread(skel_path)
    if skel_bgr is None:
        raise FileNotFoundError(f"Cannot load skeleton image at {skel_path}")
    skel = cv2.cvtColor(skel_bgr, cv2.COLOR_BGR2GRAY)

    raw = cv2.HoughLinesP(
        skel,
        rho=1,
        theta=np.pi / 180,
        threshold=HOUGH_THRESH,
        minLineLength=MIN_LINE_LEN,
        maxLineGap=MAX_LINE_GAP,
    )
    lines = [tuple(l[0]) for l in raw] if raw is not None else []
    extended = [_extend_segment(seg, EXT_PIXELS) for seg in lines]

    out_dir = os.path.dirname(out_path) or "."
    hough_img = np.zeros_like(skel)
    overlay = cv2.cvtColor(skel, cv2.COLOR_GRAY2BGR)

    for x1, y1, x2, y2 in lines:
        cv2.line(hough_img, (x1, y1), (x2, y2), 255, 1)
    for x1, y1, x2, y2 in extended:
        cv2.line(overlay, (x1, y1), (x2, y2), (0, 0, 255), 1)

    cv2.imwrite(os.path.join(out_dir, "step4_hough_lines.png"), hough_img)
    cv2.imwrite(out_path, overlay)
    print(f"Extended {len(extended)} lines by {EXT_PIXELS}px at each end")
    return out_path


def _extend_segment(segment: tuple, ext_pixels: int) -> tuple:
    x1, y1, x2, y2 = segment
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    if length == 0:
        return segment
    ux, uy = dx / length, dy / length
    nx1 = int(round(x1 - ux * ext_pixels))
    ny1 = int(round(y1 - uy * ext_pixels))
    nx2 = int(round(x2 + ux * ext_pixels))
    ny2 = int(round(y2 + uy * ext_pixels))
    return (nx1, ny1, nx2, ny2)
