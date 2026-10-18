"""Stage 4: detect wall segments in the skeleton with a Hough transform.

Skeletonization leaves a 1px centerline, but it's rarely one clean
straight run per wall - corners and intersections break it into many
short segments. This stage pulls those segments out with a
probabilistic Hough transform.
"""

import os

import cv2
import numpy as np

HOUGH_THRESH = 50
MIN_LINE_LEN = 2
MAX_LINE_GAP = 10


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

    out_dir = os.path.dirname(out_path) or "."
    hough_img = np.zeros_like(skel)
    for x1, y1, x2, y2 in lines:
        cv2.line(hough_img, (x1, y1), (x2, y2), 255, 1)
    cv2.imwrite(os.path.join(out_dir, "step4_hough_lines.png"), hough_img)

    cv2.imwrite(out_path, cv2.cvtColor(skel, cv2.COLOR_GRAY2BGR))
    return out_path
