from __future__ import annotations

import argparse
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image


# ============================================================================
# Utility Functions
# ============================================================================

def resolve_workers(value: Optional[int]) -> int:
    """Resolve worker count, limiting to a sensible CPU bound when value is missing or zero."""
    if value is None or value == 0:
        return max(1, min(8, os.cpu_count() or 1))
    return max(1, value)


def read_video_frames(
    path: Path, max_frames: Optional[int] = None, stride: int = 1
) -> List[np.ndarray]:
    """Read frames from a video, optionally sub-sampling by stride and limiting count."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")

    frames: List[np.ndarray] = []
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index % stride == 0:
            frames.append(frame)
            if max_frames and len(frames) >= max_frames:
                break
        index += 1
    cap.release()
    if not frames:
        raise ValueError("No frames read from video")
    return frames


def write_video(
    frames: Sequence[np.ndarray],
    transforms: Sequence[np.ndarray],
    canvas_size: Tuple[int, int],
    offset: np.ndarray,
    path: Path,
    fps: float,
    frame_postprocess: Callable[[np.ndarray], np.ndarray] | None = None,
) -> None:
    """Write stabilized video (not used in submission to avoid slowing down tests)"""
    def apply_post(img: np.ndarray) -> np.ndarray:
        return frame_postprocess(img) if frame_postprocess else img

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    first_frame = apply_post(warp_frame(frames[0], transforms[0], canvas_size, offset))
    writer = cv2.VideoWriter(str(path), fourcc, fps, (first_frame.shape[1], first_frame.shape[0]))
    writer.write(first_frame)

    if len(frames) > 1:
        for frame, T in zip(frames[1:], transforms[1:]):
            warped = apply_post(warp_frame(frame, T, canvas_size, offset))
            writer.write(warped)
    writer.release()


# ============================================================================
# Feature Detection and Matching
# ============================================================================

def harris_corners(
    gray: np.ndarray,
    max_features: int,
    quality_level: float,
    min_distance: int,
    block_size: int,
    k: float = 0.04,
    mask: np.ndarray | None = None,
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
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect Harris corners and track them with LK optical flow between frames."""
    corners = harris_corners(
        gray_a,
        max_features=max_features,
        quality_level=0.01,
        min_distance=6,
        block_size=9,
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


# ============================================================================
# Transformation Estimation and Processing
# ============================================================================

def estimate_partial_affine(
    src: np.ndarray, dst: np.ndarray, ransac_thresh: float = 3.0
) -> Tuple[np.ndarray, np.ndarray]:
    if len(src) < 4 or len(dst) < 4:
        return np.eye(3, dtype=np.float32), np.zeros((len(src),), dtype=bool)

    M, inliers = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh
    )
    if M is None:
        return np.eye(3, dtype=np.float32), np.zeros((len(src),), dtype=bool)

    mask = (
        inliers.ravel().astype(bool)
        if inliers is not None
        else np.zeros((len(src),), dtype=bool)
    )
    out = np.eye(3, dtype=np.float32)
    out[:2, :3] = M
    return out, mask


def decompose_affine(M: np.ndarray) -> Tuple[float, float, float, float]:
    a, b, tx = M[0, 0], M[0, 1], M[0, 2]
    c, d, ty = M[1, 0], M[1, 1], M[1, 2]
    angle = math.atan2(c, a)
    scale_x = math.hypot(a, c)
    scale_y = math.hypot(b, d)
    scale = 0.5 * (scale_x + scale_y)
    return tx, ty, angle, scale


def compose_affine_params(
    tx: float, ty: float, angle: float, scale: float = 1.0
) -> np.ndarray:
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


def smooth_signal(values: Sequence[float], radius: int) -> np.ndarray:
    if radius <= 0:
        return np.asarray(values, dtype=np.float32)
    kernel_size = radius * 2 + 1
    kernel = np.ones(kernel_size, dtype=np.float32) / float(kernel_size)
    padded = np.pad(
        np.asarray(values, dtype=np.float32), (radius, radius), mode="edge"
    )
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed.astype(np.float32)


def pairwise_transforms(
    frames: Sequence[np.ndarray],
    max_features: int = 2000,
    ransac_thresh: float = 2.0,
) -> List[np.ndarray]:
    gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    transforms: List[np.ndarray] = [np.eye(3, dtype=np.float32)]

    for i in range(1, len(gray_frames)):
        pts_prev, pts_curr = detect_and_match(
            gray_frames[i - 1], gray_frames[i], max_features=max_features
        )
        M, inliers = estimate_partial_affine(
            pts_curr, pts_prev, ransac_thresh=ransac_thresh
        )
        transforms.append(M)

    return transforms


def cancel_cumulative_rotation(transforms: Sequence[np.ndarray]) -> List[np.ndarray]:
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


def local_to_global_transformations(pairwise: Sequence[np.ndarray]) -> List[np.ndarray]:
    globals_: List[np.ndarray] = [np.eye(3, dtype=np.float32)]
    for i in range(1, len(pairwise)):
        globals_.append(globals_[-1] @ pairwise[i])
    return globals_


def detrend_video(
    globals_full: Sequence[np.ndarray],
    smooth_radius: int,
    smooth_x: bool = False,
    smooth_y: bool = True,
    smooth_angle: bool = True,
    smooth_scale: bool = False,
    detrend_y: bool = True,
    detrend_angle: bool = True,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Separate high freqs (jitters) on video from low freqs (the panning we want) by detrending"""
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

    globals_detrended_noisy: List[np.ndarray] = []
    globals_stable: List[np.ndarray] = []

    for i in range(n):
        globals_detrended_noisy.append(
            compose_affine_params(
                float(txs_arr[i]),
                float(tys_arr[i]),
                float(angles_arr[i]),
                float(scales_arr[i]),
            )
        )
        globals_stable.append(
            compose_affine_params(
                float(sm_txs[i]),
                float(sm_tys[i]),
                float(sm_angles[i]),
                float(sm_scales[i]),
            )
        )
    return globals_detrended_noisy, globals_stable


# ============================================================================
# Mosaic Building
# ============================================================================

def compute_slit_scan_canvas_bounds(
    frames: Sequence[np.ndarray],
    transforms: Sequence[np.ndarray],
) -> Tuple[Tuple[int, int], np.ndarray]:
    h, w = frames[0].shape[:2]
    txs = []
    tys = []
    for T in transforms:
        txs.append(T[0, 2])
        tys.append(T[1, 2])
    
    min_tx, max_tx = min(txs), max(txs)
    min_ty, max_ty = min(tys), max(tys)
    
    width = int(math.ceil(max_tx - min_tx)) + 10
    height = int(math.ceil(max_ty - min_ty)) + h
    offset = np.array([-min_tx, -min_ty], dtype=np.float32)
    return (height, width), offset


def warp_frame(
    frame: np.ndarray,
    transform: np.ndarray,
    canvas_size: Tuple[int, int],
    offset: np.ndarray,
    border_value: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    warp = transform[:2, :].copy()
    warp[:, 2] += offset
    return cv2.warpAffine(
        frame,
        warp,
        (canvas_size[1], canvas_size[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


def build_slit_scan_mosaic(
    frames: Sequence[np.ndarray],
    transforms_smooth: Sequence[np.ndarray],
    canvas_size: Tuple[int, int],
    offset: np.ndarray,
    slit_x_offset_ratio: float = 0.0,
    slit_width: int = 1,
    stabilize_vertical: bool = True,
) -> np.ndarray:
    """
    Build a Stabilized Slit-Scan (Time-Panorama) with continuous surface filling.
    
    Fills the entire horizontal range between consecutive frames to eliminate gaps.
    Uses tx for horizontal positioning and ty for vertical alignment.
    
    Args:
        frames: Input video frames
        transforms_smooth: Smooth transformation matrices (3x3) for each frame
        canvas_size: (height, width) of the output canvas
        offset: Translation offset to apply to transformations
        slit_x_offset_ratio: Horizontal offset ratio for slit extraction (0.0 = center)
        slit_width: Width of the slit in pixels (default=1)
        stabilize_vertical: Use vertical displacement for stabilization (default: True)
    
    Returns:
        Slit-scan mosaic image (canvas_size[0] x canvas_size[1] x 3)
    """
    h, w = frames[0].shape[:2]
    canvas_h, canvas_w = canvas_size
    
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    coverage = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    
    slit_x_src = int(w * (0.5 + slit_x_offset_ratio))
    slit_x_src = max(0, min(w - 1, slit_x_src))
    
    x_prev = None
    
    for frame_idx, (frame, T_smooth) in enumerate(zip(frames, transforms_smooth)):
        M = T_smooth[:2, :].copy()
        M[:, 2] += offset
        
        tx = M[0, 2]
        ty = M[1, 2]
        
        x_curr = int(round(tx))
        
        if stabilize_vertical:
            y_offset_curr = int(round(ty))
        else:
            y_offset_curr = 0
        
        if x_prev is None:
            x_prev = x_curr
            continue

        # Determine range to fill
        if x_curr > x_prev:
            x_start = x_prev
            x_end = x_curr if frame_idx == len(frames) - 1 else x_curr - 1
        elif x_curr < x_prev:
            x_start = x_curr if frame_idx == len(frames) - 1 else x_curr + 1
            x_end = x_prev
        else:
            x_start = x_prev
            x_end = x_prev
        
        x_start = max(0, x_start)
        x_end = min(canvas_w - 1, x_end)
        
        if x_start <= x_end:
            fill_width = x_end - x_start + 1
            
            # Extract from source frame
            extract_width = max(1, min(fill_width, w - 1))
            extract_start = max(0, min(w - extract_width, slit_x_src - extract_width // 2))
            slit_strip = frame[:, extract_start:extract_start + extract_width, :]
            
            # Resize to fill width
            if fill_width != extract_width:
                slit_strip = cv2.resize(slit_strip, (fill_width, h), interpolation=cv2.INTER_LINEAR)
            
            y_offset = y_offset_curr
            
            y_start_dst = max(0, y_offset)
            y_end_dst = min(canvas_h, y_offset + h)
            
            y_start_src = max(0, -y_offset)
            y_end_src = min(h, canvas_h - y_offset)
            
            if y_start_dst < y_end_dst and y_start_src < y_end_src:
                strip_to_place = slit_strip[y_start_src:y_end_src, :, :]
                
                # Blend in overlap zones smoothly using addWeighted
                for xi in range(fill_width):
                    x_canvas = x_start + xi
                    if 0 <= x_canvas < canvas_w:
                        if coverage[y_start_dst:y_end_dst, x_canvas].max() > 0:
                            # Blend with existing content
                            alpha = 0.5
                            canvas[y_start_dst:y_end_dst, x_canvas, :] = cv2.addWeighted(
                                canvas[y_start_dst:y_end_dst, x_canvas, :], 1 - alpha,
                                strip_to_place[:, xi, :], alpha, 0
                            ).astype(np.uint8)
                        else:
                            # First coverage at this x position
                            canvas[y_start_dst:y_end_dst, x_canvas, :] = strip_to_place[:, xi, :]
                        coverage[y_start_dst:y_end_dst, x_canvas] = 1
        
        x_prev = x_curr
    
    return canvas


def align_all_panoramas(
    panoramas_list: Sequence[np.ndarray],
    reference_index: int,
    click_coords: Tuple[int, int],
    patch_radius: int = 20,
) -> List[np.ndarray]:
    """
    Align a list of panoramas so the chosen click coordinate is stationary.

    panoramas_list: ordered left-to-right panoramas (HxWxC, BGR)
    reference_index: anchor panorama index
    click_coords: (x, y) coordinate selected in the reference panorama
    patch_radius: half-size of the square template used for matching
    """
    if not panoramas_list:
        return []

    if reference_index < 0 or reference_index >= len(panoramas_list):
        raise ValueError("reference_index is out of bounds for the panoramas list")

    h_ref, w_ref = panoramas_list[reference_index].shape[:2]
    x_click = int(round(click_coords[0]))
    y_click = int(round(click_coords[1]))
    x_click = max(0, min(w_ref - 1, x_click))
    y_click = max(0, min(h_ref - 1, y_click))
    patch_radius = max(1, int(patch_radius))

    x0 = max(0, x_click - patch_radius)
    x1 = min(w_ref, x_click + patch_radius + 1)
    y0 = max(0, y_click - patch_radius)
    y1 = min(h_ref, y_click + patch_radius + 1)

    template = panoramas_list[reference_index][y0:y1, x0:x1]
    if template.size == 0:
        raise ValueError("Template patch is empty; check click coordinates and patch radius")

    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if template.ndim == 3 else template

    aligned: List[np.ndarray] = []
    for idx, pano in enumerate(panoramas_list):
        if pano.shape[:2] != (h_ref, w_ref):
            raise ValueError("All panoramas must share the same spatial resolution")

        if idx == reference_index:
            aligned.append(pano)
            continue

        pano_gray = cv2.cvtColor(pano, cv2.COLOR_BGR2GRAY) if pano.ndim == 3 else pano
        match_map = cv2.matchTemplate(pano_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(match_map)

        matched_center_x = max_loc[0] + template_gray.shape[1] * 0.5
        horizontal_shift = int(round(x_click - matched_center_x))

        # Use translation matrix instead of roll to avoid wrap-around
        M = np.float32([[1, 0, horizontal_shift], [0, 1, 0]])
        shifted = cv2.warpAffine(pano, M, (pano.shape[1], pano.shape[0]), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        aligned.append(shifted)

    return aligned


def align_stereo_pair(
    left: np.ndarray,
    right: np.ndarray,
    convergence_point: Tuple[int, int],
    patch_radius: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align a stereo pair so the convergence point has zero disparity.
    
    Standard stereo alignment: keep left image as reference, shift only the right image.
    Black borders are normal and should be cropped to the common viewing area.
    
    This creates a natural 3D effect where:
    - Objects at the convergence point have zero disparity (appear on screen plane)
    - Objects behind have positive disparity (appear behind screen)
    - Objects in front have negative disparity (appear in front of screen)
    
    Args:
        left: Left panorama image (kept as reference)
        right: Right panorama image (will be shifted)
        convergence_point: (x, y) coordinate in left image to use as convergence point
        patch_radius: Radius of template patch for matching
    
    Returns:
        Tuple of (left, aligned_right) images
    """
    if left.shape != right.shape:
        raise ValueError("Left and right images must have the same dimensions")
    
    h, w = left.shape[:2]
    x_click = int(round(convergence_point[0]))
    y_click = int(round(convergence_point[1]))
    x_click = max(patch_radius, min(w - patch_radius - 1, x_click))
    y_click = max(patch_radius, min(h - patch_radius - 1, y_click))
    
    # Extract template from left image
    x0 = max(0, x_click - patch_radius)
    x1 = min(w, x_click + patch_radius + 1)
    y0 = max(0, y_click - patch_radius)
    y1 = min(h, y_click + patch_radius + 1)
    
    template = left[y0:y1, x0:x1]
    if template.size == 0:
        return left, right
    
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if template.ndim == 3 else template
    right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY) if right.ndim == 3 else right
    
    # Find matching point in right image
    match_map = cv2.matchTemplate(right_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(match_map)
    
    matched_center_x = max_loc[0] + template_gray.shape[1] * 0.5
    
    # Calculate shift: align the matched point in right to the convergence point position
    horizontal_shift = int(round(x_click - matched_center_x))
    
    # Shift only the right image (left stays as reference)
    # Black borders are expected in stereo - crop to common area for viewing
    M = np.float32([[1, 0, horizontal_shift], [0, 1, 0]])
    aligned_right = cv2.warpAffine(right, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    return left, aligned_right


def render_strip_sweep_video(
    frames: Sequence[np.ndarray],
    transforms_smooth: Sequence[np.ndarray],
    canvas_size: Tuple[int, int],
    offset: np.ndarray,
    output_path: Path,
    sweep_extent: float = 0.4,
    steps: int = 100,
    fps: float = 10,
    slit_width: int = 1,
    convergence_point: Optional[Tuple[int, int]] = None,
    patch_radius: int = 20,
) -> None:
    """
    Render a sweep video by varying the strip position with bounce effect.
    
    Args:
        frames: Input video frames
        transforms_smooth: Smooth transformation matrices
        canvas_size: Output canvas size (height, width)
        offset: Translation offset for transformations
        output_path: Path to save output video
        sweep_extent: Maximum sweep offset ratio
        steps: Number of sweep steps (one direction)
        fps: Output video frame rate
        slit_width: Width of slit in pixels
        convergence_point: Optional (x, y) point for stereo convergence alignment.
                          If None, images remain aligned at infinity (default behavior).
                          If specified, stereo pairs are aligned so this point has zero disparity.
        patch_radius: Radius of template patch for convergence point matching
    """
    strip_locations = np.linspace(-abs(sweep_extent), abs(sweep_extent), steps)
    strip_locations = np.concatenate([strip_locations, strip_locations[-2:0:-1]])

    mosaics: List[np.ndarray] = []
    
    # If convergence point is specified, we need to generate stereo pairs and align them
    if convergence_point is not None:
        # Generate pairs of mosaics (left, right) and align them
        for i in range(0, len(strip_locations) - 1, 2):
            left_ratio = strip_locations[i]
            right_ratio = strip_locations[i + 1] if i + 1 < len(strip_locations) else strip_locations[i]
            
            # Generate left and right mosaics
            left = build_slit_scan_mosaic(
                frames,
                transforms_smooth,
                canvas_size,
                offset,
                slit_x_offset_ratio=left_ratio,
                slit_width=slit_width,
                stabilize_vertical=True,
            )
            right = build_slit_scan_mosaic(
                frames,
                transforms_smooth,
                canvas_size,
                offset,
                slit_x_offset_ratio=right_ratio,
                slit_width=slit_width,
                stabilize_vertical=True,
            )
            
            # Align based on convergence point
            left_aligned, right_aligned = align_stereo_pair(
                left, right, convergence_point, patch_radius
            )
            
            # Add both to mosaics
            mosaics.append(left_aligned)
            if i + 1 < len(strip_locations):
                mosaics.append(right_aligned)
    else:
        # Default behavior: no convergence alignment (infinity convergence)
        for center_ratio in strip_locations:
            mosaic = build_slit_scan_mosaic(
                frames,
                transforms_smooth,
                canvas_size,
                offset,
                slit_x_offset_ratio=center_ratio,
                slit_width=slit_width,
                stabilize_vertical=True,
            )
            mosaics.append(mosaic)

    # Write video
    h, w = mosaics[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        return

    for mosaic in mosaics:
        writer.write(mosaic)
    writer.release()


# ============================================================================
# Main Function for Submission
# ============================================================================

def generate_panorama(input_frames_path, n_out_frames):
    """
    Main entry point for ex4
    :param input_frames_path : path to a dir with input video frames.
    We will test your code with a dir that has K frames, each in the format
    "frame_i:05d.jpg" (e.g., frame_00000.jpg, frame_00001.jpg, frame_00002.jpg, ...).
    :param n_out_frames: number of generated panorama frames
    :return: A list of generated panorama frames (of size n_out_frames),
    each list item should be a PIL image of a generated panorama.
    """
    # Read frames from the directory
    input_path = Path(input_frames_path)
    frame_files = sorted(input_path.glob("frame_*.jpg"))

    if not frame_files:
        raise ValueError(f"No frames found in {input_frames_path}")

    frames = [cv2.imread(str(f)) for f in frame_files]
    frames = [f for f in frames if f is not None]

    if not frames:
        raise ValueError("Failed to read any frames")

    # Compute pairwise transformations
    pairwise = pairwise_transforms(
        frames,
        max_features=300,
        ransac_thresh=1.5,
    )

    # Cancel cumulative rotation
    pairwise = cancel_cumulative_rotation(pairwise)

    # Convert to global transformations
    global_full = local_to_global_transformations(pairwise)

    # Detrend video for stabilization
    global_noisy, global_stable = detrend_video(
        global_full,
        smooth_radius=25,
        smooth_x=False,
        smooth_y=True,
        smooth_angle=True,
        smooth_scale=False,
        detrend_y=True,
        detrend_angle=True,
    )

    canvas_size, offset = compute_slit_scan_canvas_bounds(frames, global_stable)

    # Generate n_out_frames panorama frames using slit-scan technique
    # Create sweep positions for n_out_frames to support stereo/multi-view output
    baseline = 0.08  # Default stereo baseline
    sweep_extent = max(0.05, baseline * 2.5)
    
    # Generate exactly n_out_frames with linear sweep from left to right
    if n_out_frames == 1:
        slit_offsets = [0.0]
    else:
        slit_offsets = list(np.linspace(-abs(sweep_extent), abs(sweep_extent), n_out_frames))

    # Generate panorama frames using slit-scan approach
    panorama_frames = []
    for slit_offset in slit_offsets:
        mosaic = build_slit_scan_mosaic(
            frames,
            global_stable,
            canvas_size,
            offset,
            slit_x_offset_ratio=slit_offset,
            slit_width=1,
            stabilize_vertical=True,
        )

        mosaic_rgb = cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(mosaic_rgb)
        panorama_frames.append(pil_image)

    return panorama_frames


# ============================================================================
# Command-line Interface and Main Flow
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stereo mosaicing pipeline")
    parser.add_argument("--input", required=True, type=Path, help="Input video path")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/mosaic.png"),
        help="Output mosaic image path",
    )
    parser.add_argument(
        "--stabilized-video",
        type=Path,
        default=None,
        help="Optional path for stabilized preview video",
    )
    parser.add_argument(
        "--max-frames", type=int, default=None, help="Optional cap on number of frames"
    )
    parser.add_argument(
        "--stride", type=int, default=1, help="Frame stride for subsampling"
    )
    parser.add_argument(
        "--max-features", type=int, default=500, help="feature budget per frame"
    )
    parser.add_argument(
        "--smooth-radius", type=int, default=25, help="Radius for trajectory smoothing"
    )
    parser.add_argument(
        "--stereo-baseline-ratio",
        type=float,
        default=0.08,
        help="Horizontal slit offset ratio (relative to frame width) for stereo pair",
    )
    parser.add_argument(
        "--process-portrait",
        action="store_true",
        help="If the input video is portrait, rotate it 90 degrees CCW for processing and rotate outputs back to portrait",
    )
    parser.add_argument(
        "--save-outputs",
        action="store_true",
        help="Save output files (disabled by default for fast testing)",
    )
    parser.add_argument(
        "--slit-width",
        type=int,
        default=3,
        help="Width of slit in pixels for slit-scan method (default=3)",
    )
    parser.add_argument(
        "--convergence-point",
        type=str,
        default=None,
        help="Convergence point for stereo alignment as 'x,y' (e.g., '500,300'). If not set, convergence is at infinity.",
    )

    return parser.parse_args()


def generate_outputs(
    frames,
    global_stable,
    canvas_size,
    offset,
    args,
    fps,
    run_dir,
    rotate_back_code=None,
    convergence_point=None,
):
    """Generate all outputs (mosaics, stereo pair, sweep video)"""
    stereo_baseline = args.stereo_baseline_ratio

    # Build main slit-scan mosaic
    mosaic = build_slit_scan_mosaic(
        frames,
        global_stable,
        canvas_size,
        offset,
        slit_x_offset_ratio=0.0,
        slit_width=args.slit_width,
        stabilize_vertical=True,
    )

    # Build stereo pair using slit-scan
    left = build_slit_scan_mosaic(
        frames,
        global_stable,
        canvas_size,
        offset,
        slit_x_offset_ratio=-stereo_baseline,
        slit_width=args.slit_width,
        stabilize_vertical=True,
    )
    right = build_slit_scan_mosaic(
        frames,
        global_stable,
        canvas_size,
        offset,
        slit_x_offset_ratio=stereo_baseline,
        slit_width=args.slit_width,
        stabilize_vertical=True,
    )
    
    # Apply convergence point alignment if specified
    if convergence_point is not None:
        left, right = align_stereo_pair(left, right, convergence_point, patch_radius=20)

    def restore_orientation(img):
        if rotate_back_code is None:
            return img
        return cv2.rotate(img, rotate_back_code)

    mosaic = restore_orientation(mosaic)
    left = restore_orientation(left)
    right = restore_orientation(right)

    # Save outputs only if requested
    if args.save_outputs:
        mosaic_path = run_dir / f"{args.input.stem}_mosaic.png"
        cv2.imwrite(str(mosaic_path), mosaic)
        print(f"Mosaic saved to: {mosaic_path}")

        left_path = run_dir / f"{args.input.stem}_mosaic_L.png"
        right_path = run_dir / f"{args.input.stem}_mosaic_R.png"
        cv2.imwrite(str(left_path), left)
        cv2.imwrite(str(right_path), right)
        print(f"Stereo images saved to: {left_path}, {right_path}")

        sweep_path = run_dir / f"{args.input.stem}_sweep.mp4"
        render_strip_sweep_video(
            frames,
            global_stable,
            canvas_size,
            offset,
            output_path=sweep_path,
            sweep_extent=max(0.05, stereo_baseline * 2.5),
            steps=60,
            fps=max(10.0, fps),
            slit_width=args.slit_width,
            convergence_point=convergence_point,
            patch_radius=20,
        )
        print(f"Sweep video saved to: {sweep_path}")
        
        # Create downsampled versions of sweep video
        downsample_targets = [(1280, 720), (1920, 1080)]
        cap = cv2.VideoCapture(str(sweep_path))
        if cap.isOpened():
            orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            for target_w, target_h in downsample_targets:
                if target_w <= 0 or target_h <= 0:
                    continue
                
                scale = min(target_w / orig_w, target_h / orig_h, 1.0)
                new_w = max(2, int(round(orig_w * scale)))
                new_h = max(2, int(round(orig_h * scale)))
                new_w = min(target_w, new_w)
                new_h = min(target_h, new_h)
                
                if new_w % 2 != 0:
                    new_w -= 1
                if new_h % 2 != 0:
                    new_h -= 1
                
                if new_w < 2 or new_h < 2 or (new_w == orig_w and new_h == orig_h):
                    continue
                
                target_path = run_dir / f"{args.input.stem}_sweep_{target_h}p.mp4"
                cap_read = cv2.VideoCapture(str(sweep_path))
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                ds_writer = cv2.VideoWriter(str(target_path), fourcc, max(10.0, fps), (new_w, new_h))
                
                if ds_writer.isOpened():
                    while True:
                        ret, frame = cap_read.read()
                        if not ret:
                            break
                        downsampled = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                        ds_writer.write(downsampled)
                    ds_writer.release()
                    print(f"Downsampled sweep saved to: {target_path} ({new_w}x{new_h})")
                
                cap_read.release()
    else:
        print("Output generation complete (files not saved - use --save-outputs to save)")

    return mosaic, left, right


def main() -> None:
    """Main entry point with complete processing flow"""
    args = parse_args()
    frames = read_video_frames(
        args.input, max_frames=args.max_frames, stride=args.stride
    )
    if not frames:
        raise RuntimeError("No frames loaded; check input path and stride/max-frames settings")

    h, w = frames[0].shape[:2]
    print(f"Loaded {len(frames)} frames from {args.input} at {w}x{h} (stride={args.stride}, max={args.max_frames})")
    fps = cv2.VideoCapture(str(args.input)).get(cv2.CAP_PROP_FPS) or 25.0

    rotate_back_code = None
    if args.process_portrait:
        print("Process-portrait requested; rotating 90 degrees CCW for processing.")
        frames = [cv2.rotate(f, cv2.ROTATE_90_COUNTERCLOCKWISE) for f in frames]
        rotate_back_code = cv2.ROTATE_90_CLOCKWISE

    # Create output directory only if saving
    run_dir = None
    if args.save_outputs:
        run_dir = (
            Path("output") / f"{args.input.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        
        args_out = run_dir / "args.txt"
        with args_out.open("w", encoding="utf-8") as f:
            for k, v in sorted(vars(args).items()):
                f.write(f"{k}: {v}\n")
    else:
        run_dir = Path("output")  # Dummy path, won't be used

    # Process frames
    print("Computing pairwise transformations...")
    pairwise = pairwise_transforms(
        frames,
        max_features=args.max_features,
        ransac_thresh=1.5,
    )

    print("Canceling cumulative rotation...")
    pairwise = cancel_cumulative_rotation(pairwise)

    print("Computing global transformations...")
    global_full = local_to_global_transformations(pairwise)
    
    print("Detrending video for stabilization...")
    global_noisy, global_stable = detrend_video(
        global_full,
        smooth_radius=args.smooth_radius,
        smooth_x=False,
        smooth_y=True,
        smooth_angle=True,
        smooth_scale=False,
        detrend_y=True,
        detrend_angle=True,
    )

    print("Computing canvas bounds...")
    canvas_size, offset = compute_slit_scan_canvas_bounds(frames, global_stable)
    print(f"Canvas size {canvas_size}, offset {offset}")

    # Parse convergence point if provided
    convergence_point = None
    if args.convergence_point:
        try:
            x, y = map(int, args.convergence_point.split(','))
            convergence_point = (x, y)
            print(f"Using convergence point: ({x}, {y})")
        except ValueError:
            print(f"Warning: Invalid convergence point format '{args.convergence_point}'. Expected 'x,y'. Using infinity convergence.")

    print("Generating outputs...")
    generate_outputs(
        frames,
        global_stable,
        canvas_size,
        offset,
        args,
        fps,
        run_dir,
        rotate_back_code,
        convergence_point,
    )

    if args.stabilized_video and args.save_outputs:
        print("Writing stabilized video...")
        stabilized_path = Path(args.stabilized_video)
        frame_postprocess = (
            (lambda img: cv2.rotate(img, rotate_back_code))
            if rotate_back_code is not None
            else None
        )
        write_video(
            frames,
            global_stable,
            canvas_size,
            offset,
            stabilized_path,
            fps=fps,
            frame_postprocess=frame_postprocess,
        )
        print(f"Stabilized video saved to: {stabilized_path}")

    print("Done!")


if __name__ == "__main__":
    main()
