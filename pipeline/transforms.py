from __future__ import annotations

import math
from pathlib import Path
from typing import  List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .features import detect_and_match
from .types import Affine3x3
from .utils import progress_iter


def estimate_affine(src: np.ndarray, dst: np.ndarray, ransac_thresh: float = 3.0) -> Affine3x3:
    if len(src) < 4 or len(dst) < 4:
        return np.eye(3, dtype=np.float32)
    M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh)
    if M is None:
        return np.eye(3, dtype=np.float32)
    out = np.eye(3, dtype=np.float32)
    out[:2, :3] = M
    return out


def decompose_affine(M: Affine3x3) -> Tuple[float, float, float, float]:
    a, b, tx = M[0, 0], M[0, 1], M[0, 2]
    c, d, ty = M[1, 0], M[1, 1], M[1, 2]
    angle = math.atan2(c, a)
    scale_x = math.hypot(a, c)
    scale_y = math.hypot(b, d)
    scale = 0.5 * (scale_x + scale_y)
    return tx, ty, angle, scale


def compose_affine_params(tx: float, ty: float, angle: float, scale: float = 1.0) -> Affine3x3:
    ca = math.cos(angle) * scale
    sa = math.sin(angle) * scale
    M = np.eye(3, dtype=np.float32)
    M[0, 0] = ca
    M[0, 1] = -sa
    M[1, 0] = sa
    M[1, 1] = ca
    M[0, 2] = tx
    M[1, 2] = ty
    return M


def _should_save_debug(idx: int, total: int, max_files: int = 40) -> bool:
    if total <= 1:
        return True
    stride = max(1, math.ceil(total / max_files))
    return idx % stride == 0 or idx == total - 1


def _save_feature_debug(
    gray_prev: np.ndarray,
    gray_curr: np.ndarray,
    pts_prev: np.ndarray,
    pts_curr: np.ndarray,
    frame_idx: int,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    vis_prev = cv2.cvtColor(gray_prev, cv2.COLOR_GRAY2BGR)
    for x, y in pts_prev:
        cv2.circle(vis_prev, (int(round(x)), int(round(y))), 2, (0, 255, 0), -1)
    cv2.imwrite(str(out_dir / f"features_{frame_idx:04d}.png"), vis_prev)

    vis_flow = cv2.cvtColor(gray_curr, cv2.COLOR_GRAY2BGR)
    for (x0, y0), (x1, y1) in zip(pts_prev, pts_curr):
        p0 = (int(round(x0)), int(round(y0)))
        p1 = (int(round(x1)), int(round(y1)))
        cv2.arrowedLine(vis_flow, p0, p1, (0, 200, 255), 1, tipLength=0.25)
        cv2.circle(vis_flow, p1, 2, (40, 220, 40), -1)
    cv2.imwrite(str(out_dir / f"flow_{frame_idx:04d}.png"), vis_flow)


def smooth_signal(values: Sequence[float], radius: int) -> np.ndarray:
    if radius <= 0:
        return np.asarray(values, dtype=np.float32)
    kernel_size = radius * 2 + 1
    kernel = np.ones(kernel_size, dtype=np.float32) / float(kernel_size)
    padded = np.pad(np.asarray(values, dtype=np.float32), (radius, radius), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed.astype(np.float32)


def detrend_video(
    globals_full: Sequence[Affine3x3],
    smooth_radius: int,
    smooth_x: bool = False,
    smooth_y: bool = True,
    smooth_angle: bool = True,
    smooth_scale: bool = False,
    detrend_y: bool = True,
    detrend_angle: bool = True,
) -> Tuple[List[Affine3x3], List[Affine3x3]]:
    """seperate high freqs (jitters) on video from low freqs (the pannning we want) by detrending"""
    txs: List[float] = []
    tys: List[float] = []
    angles: List[float] = []
    scales: List[float] = []
    for M in globals_full:
        tx, ty, ang, scl = decompose_affine(M)
        txs.append(tx)
        tys.append(ty)
        angles.append(ang)
        scales.append(scl)

    angles_unwrapped = np.unwrap(np.asarray(angles, dtype=np.float32))

    n = len(angles_unwrapped)
    txs_arr = np.asarray(txs, dtype=np.float32)
    tys_arr = np.asarray(tys, dtype=np.float32)
    angles_arr = angles_unwrapped.astype(np.float32)
    scales_arr = np.asarray(scales, dtype=np.float32)

    if detrend_y and n > 1:
        y_trend = np.linspace(tys_arr[0], tys_arr[-1], n, dtype=np.float32)
        tys_arr = tys_arr - y_trend + tys_arr[0]

    if detrend_angle and n > 1:
        angle_trend = np.linspace(angles_arr[0], angles_arr[-1], n, dtype=np.float32)
        angles_arr = angles_arr - angle_trend + angles_arr[0]

    sm_txs = smooth_signal(txs_arr, smooth_radius) if smooth_x else txs_arr
    sm_tys = smooth_signal(tys_arr, smooth_radius) if smooth_y else tys_arr
    sm_angles = smooth_signal(angles_arr, smooth_radius) if smooth_angle else angles_arr
    sm_scales = smooth_signal(scales_arr, smooth_radius) if smooth_scale else scales_arr

    globals_detrended_noisy: List[Affine3x3] = []
    globals_stable: List[Affine3x3] = []

    for i in range(n):
        globals_detrended_noisy.append(
            compose_affine_params(float(txs_arr[i]), float(tys_arr[i]), float(angles_arr[i]), float(scales_arr[i]))
        )
        globals_stable.append(
            compose_affine_params(float(sm_txs[i]), float(sm_tys[i]), float(sm_angles[i]), float(sm_scales[i]))
        )

    return globals_detrended_noisy, globals_stable


def local_to_global_transformations(pairwise: Sequence[Affine3x3]) -> List[Affine3x3]:
    globals_: List[Affine3x3] = [np.eye(3, dtype=np.float32)]
    for i in range(1, len(pairwise)):
        globals_.append(globals_[-1] @ pairwise[i])
    return globals_


def lock_convergence_point(transforms: Sequence[Affine3x3], point: Optional[Tuple[float, float]]) -> List[Affine3x3]:
    if point is None:
        return list(transforms)
    px, py = point
    p = np.array([px, py], dtype=np.float32)
    locked: List[Affine3x3] = []
    for T in transforms:
        R = T[:2, :2]
        t = T[:2, 2]
        adjust = (R - np.eye(2, dtype=np.float32)) @ p
        new_t = t - adjust
        new_T = T.copy()
        new_T[:2, 2] = new_t
        locked.append(new_T)
    return locked


def cancel_cumulative_rotation(transforms: Sequence[Affine3x3]) -> List[Affine3x3]:
    if not transforms:
        return []

    pairwise_angles = []
    for T in transforms:
        _, _, ang, _ = decompose_affine(T)
        pairwise_angles.append(ang)

    total_rotation = np.sum(pairwise_angles)
    avg_drift = total_rotation / len(transforms)

    corrected_transforms = []
    for T in transforms:
        tx, ty, ang, scl = decompose_affine(T)
        new_ang = ang - avg_drift
        corrected_transforms.append(compose_affine_params(tx, ty, new_ang, scl))

    return corrected_transforms


def pairwise_transforms(
    frames: Sequence[np.ndarray],
    max_features: int = 2000,
    ransac_thresh: float = 2.0,
    mask: Optional[np.ndarray] = None,
    debug_dir: Optional[Path] = None,
) -> List[Affine3x3]:
    gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    transforms: List[Affine3x3] = [np.eye(3, dtype=np.float32)]

    h, w = gray_frames[0].shape
    #mask to ignore most of the image so we wont get feature points around the subject but the background
    if mask is None:
        mask = np.ones((h, w), dtype=np.uint8) * 255
        mask[int(h * 0.25) : int(h * 0.75), :] = 0

    total_pairs = len(gray_frames) - 1
    for i in progress_iter(range(1, len(gray_frames)), total=total_pairs, desc="Pairwise align"):
        pts_prev, pts_curr = detect_and_match(gray_frames[i - 1], gray_frames[i], max_features=max_features, mask=mask)
        M = estimate_affine(pts_curr, pts_prev, ransac_thresh=ransac_thresh)
        transforms.append(M)

        if debug_dir is not None and len(pts_prev) > 0 and _should_save_debug(i, total_pairs):
            _save_feature_debug(gray_frames[i - 1], gray_frames[i], pts_prev, pts_curr, i, debug_dir)
    return transforms
