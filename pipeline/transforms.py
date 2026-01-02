from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .features import detect_and_match
from .types import Affine3x3
from .utils import progress_iter


def estimate_partial_affine(src: np.ndarray, dst: np.ndarray, ransac_thresh: float = 3.0) -> Tuple[Affine3x3, np.ndarray]:
    if len(src) < 4 or len(dst) < 4:
        return np.eye(3, dtype=np.float32), np.zeros((len(src),), dtype=bool)

    M, inliers = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh
    )
    if M is None:
        return np.eye(3, dtype=np.float32), np.zeros((len(src),), dtype=bool)

    mask = inliers.ravel().astype(bool) if inliers is not None else np.zeros((len(src),), dtype=bool)
    out = np.eye(3, dtype=np.float32)
    out[:2, :3] = M
    return out, mask


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


def _save_ransac_debug(
    gray_prev: np.ndarray,
    gray_curr: np.ndarray,
    pts_prev: np.ndarray,
    pts_curr: np.ndarray,
    inliers: np.ndarray,
    frame_idx: int,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    h, w = gray_prev.shape
    canvas = np.zeros((h, w * 2, 3), dtype=np.uint8)
    canvas[:, :w, :] = cv2.cvtColor(gray_prev, cv2.COLOR_GRAY2BGR)
    canvas[:, w:, :] = cv2.cvtColor(gray_curr, cv2.COLOR_GRAY2BGR)

    inliers_mask = inliers.ravel().astype(bool) if inliers is not None else np.zeros((len(pts_prev),), dtype=bool)
    for idx, (p0, p1) in enumerate(zip(pts_prev, pts_curr)):
        color = (20, 200, 20) if inliers_mask[idx] else (30, 30, 220)
        p0_i = (int(round(p0[0])), int(round(p0[1])))
        p1_i = (int(round(p1[0] + w)), int(round(p1[1])))
        cv2.line(canvas, p0_i, p1_i, color, 1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, p0_i, 2, color, -1)
        cv2.circle(canvas, p1_i, 2, color, -1)

    label = f"inliers {int(inliers_mask.sum())}/{len(inliers_mask)}"
    cv2.putText(canvas, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_dir / f"ransac_{frame_idx:04d}.png"), canvas)


def _save_ransac_plot(
    pts_prev: np.ndarray,
    pts_curr: np.ndarray,
    inliers: np.ndarray,
    frame_idx: int,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    inliers_mask = inliers.ravel().astype(bool) if inliers is not None else np.zeros((len(pts_prev),), dtype=bool)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(pts_prev[:, 0], pts_prev[:, 1], c="gray", s=12, alpha=0.4, label="prev")
    ax.scatter(pts_curr[:, 0], pts_curr[:, 1], c="tab:blue", s=12, alpha=0.4, label="curr")

    if len(pts_prev) == len(pts_curr) and len(pts_prev) == len(inliers_mask):
        ax.scatter(pts_curr[inliers_mask, 0], pts_curr[inliers_mask, 1], c="tab:green", s=16, alpha=0.9, label="inliers")
        ax.scatter(pts_curr[~inliers_mask, 0], pts_curr[~inliers_mask, 1], c="tab:red", s=16, alpha=0.9, label="outliers")

    ax.legend(loc="upper right")
    ax.set_title(f"RANSAC pair {frame_idx:04d}\nInliers {int(inliers_mask.sum())}/{len(inliers_mask)}")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_dir / f"ransac_plot_{frame_idx:04d}.png", dpi=200)
    plt.close(fig)


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
    debug_dir: Optional[Path] = None,
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

    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        axes[0].plot(txs_arr, label="tx raw", alpha=0.5)
        axes[0].plot(sm_txs, label="tx smooth", linewidth=2)
        axes[0].set_ylabel("x (px)")

        axes[1].plot(tys_arr, label="ty raw", alpha=0.5)
        axes[1].plot(sm_tys, label="ty smooth", linewidth=2)
        axes[1].set_ylabel("y (px)")

        axes[2].plot(np.degrees(angles_arr), label="angle raw", alpha=0.5)
        axes[2].plot(np.degrees(sm_angles), label="angle smooth", linewidth=2)
        axes[2].set_ylabel("angle (deg)")
        axes[2].set_xlabel("frame")

        for ax in axes:
            ax.legend(loc="best")
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

        fig.suptitle(f"Detrend radius={smooth_radius}, smooth (x:{smooth_x}, y:{smooth_y}, angle:{smooth_angle})")
        fig.tight_layout()
        fig.savefig(debug_dir / "detrend_debug.png", dpi=200)
        plt.close(fig)

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
    debug_dir: Optional[Path] = None,
) -> List[Affine3x3]:
    gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    transforms: List[Affine3x3] = [np.eye(3, dtype=np.float32)]

    h, w = gray_frames[0].shape

    total_pairs = len(gray_frames) - 1
    inlier_counts: List[int] = []
    match_counts: List[int] = []

    feature_dir = debug_dir / "features" if debug_dir is not None else None
    ransac_dir = debug_dir / "ransac" if debug_dir is not None else None
    ransac_plot_dir = feature_dir if debug_dir is not None else None

    for i in progress_iter(range(1, len(gray_frames)), total=total_pairs, desc="Pairwise align"):
        pts_prev, pts_curr = detect_and_match(gray_frames[i - 1], gray_frames[i], max_features=max_features)
        M, inliers = estimate_partial_affine(pts_curr, pts_prev, ransac_thresh=ransac_thresh)
        transforms.append(M)

        if inliers.size > 0:
            inlier_counts.append(int(inliers.sum()))
            match_counts.append(len(inliers))

        if debug_dir is not None and len(pts_prev) > 0 and _should_save_debug(i, total_pairs):
            if feature_dir is not None:
                _save_feature_debug(gray_frames[i - 1], gray_frames[i], pts_prev, pts_curr, i, feature_dir)
            if ransac_dir is not None:
                _save_ransac_debug(gray_frames[i - 1], gray_frames[i], pts_prev, pts_curr, inliers, i, ransac_dir)
            if ransac_plot_dir is not None and len(pts_prev) == len(pts_curr):
                _save_ransac_plot(pts_prev, pts_curr, inliers, i, ransac_plot_dir)
    if inlier_counts and match_counts:
        ratios = np.asarray(inlier_counts, dtype=np.float32) / np.maximum(1.0, np.asarray(match_counts, dtype=np.float32))
        print(
            f"RANSAC inliers per pair: mean {np.mean(inlier_counts):.1f}/{np.mean(match_counts):.1f} "
            f"(ratio mean {np.mean(ratios):.2f}, min {np.min(ratios):.2f})"
        )

    return transforms
