from __future__ import annotations

import math
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image


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

def compute_canvas_bounds(
    frames: Sequence[np.ndarray],
    transforms: Sequence[np.ndarray],
) -> Tuple[Tuple[int, int], np.ndarray]:
    h, w = frames[0].shape[:2]
    corners = np.array(
        [[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], dtype=np.float32
    ).T
    xs = []
    ys = []
    for T in transforms:
        warped = T @ corners
        xs.extend(warped[0])
        ys.extend(warped[1])
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width = int(math.ceil(max_x - min_x))
    height = int(math.ceil(max_y - min_y))
    offset = np.array([-min_x, -min_y], dtype=np.float32)
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


def build_strip_mask(
    height: int, width: int, strip_ratio: float, strip_x_offset: float
) -> np.ndarray:
    strip_width = max(4, int(width * strip_ratio))
    strip_x = width * (0.5 + strip_x_offset)
    left_x_strip = int(strip_x - strip_width * 0.5)
    left_x_strip = max(0, min(width - strip_width, left_x_strip))

    mask = np.zeros((height, width), dtype=np.uint8)
    mask[:, left_x_strip : left_x_strip + strip_width] = 255
    return mask


def build_mosaic(
    frames: Sequence[np.ndarray],
    transforms_smooth: Sequence[np.ndarray],
    transforms_noisy: Sequence[np.ndarray],
    strip_ratio: float,
    canvas_size: Tuple[int, int],
    offset: np.ndarray,
    strip_x_offset: float = 0.0,
    vertical_scale: float = 1.0,
) -> np.ndarray:
    h, orig_width = frames[0].shape[:2]
    mask = build_strip_mask(h, orig_width, strip_ratio, strip_x_offset)

    acc = np.zeros((canvas_size[0], canvas_size[1], 3), dtype=np.float32)
    count_of_strips_per_px = np.zeros((canvas_size[0], canvas_size[1], 1), dtype=np.float32)

    for frame, T_smooth in zip(frames, transforms_smooth):
        warped_frame = warp_frame(
            frame, T_smooth, canvas_size, offset, border_value=(0, 0, 0)
        ).astype(np.float32)
        warped_mask = warp_frame(mask, T_smooth, canvas_size, offset, border_value=0)
        m = (warped_mask > 0).astype(np.float32)[..., None]
        acc += warped_frame * m
        count_of_strips_per_px += m

    count_of_strips_per_px[count_of_strips_per_px == 0] = 1.0
    mosaic = (acc / count_of_strips_per_px).astype(np.uint8)

    return mosaic


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

        aligned.append(np.roll(pano, shift=horizontal_shift, axis=1))

    return aligned


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

    # Compute canvas bounds
    canvas_size, offset = compute_canvas_bounds(frames, global_stable)

    # Generate n_out_frames panorama frames using sweep video technique
    # We'll generate frames by sweeping the strip position
    strip_ratio = 0.06
    vertical_scale = 1.0

    # Create sweep positions for n_out_frames with bounce effect
    baseline = 0.08  # Default stereo baseline
    sweep_extent = max(0.05, baseline * 2.5)
    
    # Calculate steps to create forward sweep
    if n_out_frames == 1:
        strip_locations = [0.0]
    else:
        # For bounce effect: go forward, then backward
        # If n_out_frames is even, split equally
        # If odd, favor forward direction
        forward_steps = (n_out_frames + 1) // 2
        strip_locations = list(np.linspace(-abs(sweep_extent), abs(sweep_extent), forward_steps))
        # Add backward sweep (excluding endpoints to avoid duplication)
        if forward_steps > 2:
            strip_locations.extend(strip_locations[-2:0:-1])
        # Trim to exact n_out_frames
        strip_locations = strip_locations[:n_out_frames]

    # Generate panorama frames
    panorama_frames = []
    for strip_offset in strip_locations:
        mosaic = build_mosaic(
            frames,
            global_stable,
            global_noisy,
            strip_ratio,
            canvas_size,
            offset,
            strip_offset,
            vertical_scale,
        )

        mosaic_rgb = cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(mosaic_rgb)
        panorama_frames.append(pil_image)

    return panorama_frames
