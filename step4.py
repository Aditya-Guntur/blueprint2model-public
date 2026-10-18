"""Stage 4: extract, extend and merge wall segments from the skeleton.

Skeletonization leaves a 1px centerline, but it's rarely one clean
straight run per wall: corners and intersections break it into many
short Hough segments. This stage:

1. Runs a probabilistic Hough transform to pull line segments out of
   the skeleton.
2. Extends each segment by a few pixels at both ends to close the
   1px gaps left at corners.
3. Classifies segments as horizontal/vertical (walls are assumed
   axis-aligned), buckets collinear segments by position, and merges
   overlapping/adjacent intervals in each bucket into a single wall.
"""

import math
import os

import cv2
import numpy as np

HOUGH_THRESH = 50
MIN_LINE_LEN = 2
MAX_LINE_GAP = 10
EXT_PIXELS = 1  # pixels to extend each line at both ends

ANGLE_TOL_DEG = 5  # degrees, for classifying a segment as horizontal/vertical
BUCKET_TOL = 5  # px tolerance to group segments as "the same" wall line
GAP_TOL_PX = 10  # px gap tolerance when merging intervals in a bucket


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
    cv2.imwrite(os.path.join(out_dir, "step4_extended_overlay.png"), overlay)
    print(f"Extended {len(extended)} lines by {EXT_PIXELS}px at each end")

    horiz, vert = _classify_segments(extended)
    h_merged = _bucket_merge(horiz, BUCKET_TOL, GAP_TOL_PX)
    v_merged = _bucket_merge(vert, BUCKET_TOL, GAP_TOL_PX)

    for ymid, x0, x1 in h_merged:
        y = int(round(ymid))
        cv2.line(overlay, (int(round(x0)), y), (int(round(x1)), y), (0, 255, 0), 2)
    for xmid, y0, y1 in v_merged:
        x = int(round(xmid))
        cv2.line(overlay, (x, int(round(y0))), (x, int(round(y1))), (0, 255, 0), 2)

    cv2.imwrite(out_path, overlay)
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


def _classify_segments(segments: list) -> tuple:
    """Split segments into horizontal/vertical buckets; drop diagonals."""
    horiz, vert = [], []
    for x1, y1, x2, y2 in segments:
        angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1))) % 180
        if angle < ANGLE_TOL_DEG or angle > 180 - ANGLE_TOL_DEG:
            ymid = (y1 + y2) / 2
            xs = sorted((x1, x2))
            horiz.append((ymid, xs[0], xs[1]))
        elif abs(angle - 90) < ANGLE_TOL_DEG:
            xmid = (x1 + x2) / 2
            ys = sorted((y1, y2))
            vert.append((xmid, ys[0], ys[1]))
    return horiz, vert


def _bucket_merge(items: list, bucket_tol: int, gap_tol: int) -> list:
    """Group (key, start, end) intervals by nearby key, then merge
    overlapping/adjacent intervals within each group."""
    buckets: dict = {}
    for key, s, e in items:
        bucket_key = next((k for k in buckets if abs(k - key) <= bucket_tol), None)
        if bucket_key is None:
            buckets[key] = [(s, e)]
        else:
            buckets[bucket_key].append((s, e))

    merged = []
    for bucket_key, intervals in buckets.items():
        intervals.sort()
        cur_start, cur_end = intervals[0]
        for next_start, next_end in intervals[1:]:
            if next_start <= cur_end + gap_tol:
                cur_end = max(cur_end, next_end)
            else:
                merged.append((bucket_key, cur_start, cur_end))
                cur_start, cur_end = next_start, next_end
        merged.append((bucket_key, cur_start, cur_end))
    return merged
