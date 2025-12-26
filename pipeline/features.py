from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np


def harris_corners(
    gray: np.ndarray,
    max_features: int,
    quality_level: float,
    min_distance: int,
    block_size: int,
    k: float = 0.04,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Manual Harris detector with optional masking."""
    g = gray.astype(np.float32)
    Ix = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)

    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy

    Sxx = cv2.GaussianBlur(Ixx, (block_size, block_size), 0)
    Syy = cv2.GaussianBlur(Iyy, (block_size, block_size), 0)
    Sxy = cv2.GaussianBlur(Ixy, (block_size, block_size), 0)

    R = (Sxx * Syy - Sxy * Sxy) - k * (Sxx + Syy) ** 2

    if mask is not None:
        R[mask == 0] = 0

    R[R < 0] = 0
    if R.size == 0:
        return np.empty((0, 2), dtype=np.float32)

    R_max = float(R.max())
    if not np.isfinite(R_max) or R_max <= 0:
        return np.empty((0, 2), dtype=np.float32)

    thresh = quality_level * R_max
    mask_r = R >= thresh
    if not np.any(mask_r):
        return np.empty((0, 2), dtype=np.float32)

    dilated = cv2.dilate(R, None)
    peaks = (R == dilated) & mask_r
    ys, xs = np.nonzero(peaks)
    scores = R[ys, xs]
    if len(scores) == 0:
        return np.empty((0, 2), dtype=np.float32)

    order = np.argsort(scores)[::-1]
    pts: List[Tuple[int, int]] = []
    min_dist2 = float(min_distance * min_distance)
    for idx in order:
        y = int(ys[idx])
        x = int(xs[idx])
        if all((x - px) * (x - px) + (y - py) * (y - py) >= min_dist2 for px, py in pts):
            pts.append((x, y))
            if len(pts) >= max_features:
                break

    if not pts:
        return np.empty((0, 2), dtype=np.float32)
    return np.array(pts, dtype=np.float32)


def detect_and_match(
    gray_a: np.ndarray,
    gray_b: np.ndarray,
    max_features: int = 2000,
    keep: int = 400,
    mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect Harris corners and track them with LK optical flow between frames."""
    corners = harris_corners(
        gray_a,
        max_features=max_features,
        quality_level=0.01,
        min_distance=6,
        block_size=9,
        mask=mask,
    )
    if len(corners) < 4:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

    pts_prev = corners.astype(np.float32)
    pts_next, status, err = cv2.calcOpticalFlowPyrLK(
        gray_a,
        gray_b,
        pts_prev,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if pts_next is None or status is None or err is None:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

    status = status.ravel().astype(bool)
    err = err.ravel()
    good = status & np.isfinite(err)
    if good.sum() < 4:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

    pts_prev_good = pts_prev[good]
    pts_next_good = pts_next[good]
    err_good = err[good]

    keep_n = min(keep, len(pts_prev_good))
    order = np.argsort(err_good)[:keep_n]
    pts_prev_good = pts_prev_good[order]
    pts_next_good = pts_next_good[order]

    return pts_prev_good, pts_next_good
