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


def render_strip_sweep_video(
    frames: Sequence[np.ndarray],
    transforms_smooth: Sequence[np.ndarray],
    transforms_noisy: Sequence[np.ndarray],
    canvas_size: Tuple[int, int],
    offset: np.ndarray,
    strip_ratio: float,
    vertical_scale: float,
    output_path: Path,
    sweep_extent: float = 0.4,
    steps: int = 60,
    fps: float = 10,
    downsample_targets: Sequence[Tuple[int, int]] | None = None,
    convergence_click: Tuple[int, int] | None = None,
    convergence_reference_index: int | None = None,
    convergence_patch_radius: int = 20,
) -> List[Tuple[Path, Tuple[int, int]]]:
    """
    Render a sweep video by varying the strip position with bounce effect.
    Optionally applies convergence alignment and downsampling.
    """
    strip_locations = np.linspace(-abs(sweep_extent), abs(sweep_extent), steps)
    # Build a bounce sequence so the sweep plays forward and then backward
    strip_locations = np.concatenate([strip_locations, strip_locations[-2:0:-1]])

    # Build all mosaics
    mosaics: List[np.ndarray] = []
    for center_ratio in strip_locations:
        mosaic = build_mosaic(
            frames,
            transforms_smooth,
            transforms_noisy,
            strip_ratio,
            canvas_size,
            offset,
            center_ratio,
            vertical_scale,
        )
        mosaics.append(mosaic)

    # Apply convergence alignment if requested
    if convergence_click is not None:
        ref_idx = convergence_reference_index if convergence_reference_index is not None else len(mosaics) // 2
        mosaics = align_all_panoramas(
            mosaics,
            reference_index=ref_idx,
            click_coords=convergence_click,
            patch_radius=convergence_patch_radius,
        )

    # Write main video
    h, w = mosaics[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        return []

    for mosaic in mosaics:
        writer.write(mosaic)
    writer.release()

    # Write downsampled versions
    downsample_outputs: List[Tuple[Path, Tuple[int, int]]] = []
    for target_w, target_h in downsample_targets or []:
        if target_w <= 0 or target_h <= 0:
            continue

        scale = min(target_w / w, target_h / h, 1.0)
        new_w = max(2, int(round(w * scale)))
        new_h = max(2, int(round(h * scale)))
        new_w = min(target_w, new_w)
        new_h = min(target_h, new_h)

        if new_w % 2 != 0:
            new_w -= 1
        if new_h % 2 != 0:
            new_h -= 1

        if new_w < 2 or new_h < 2:
            continue

        target_path = output_path.with_name(f"{output_path.stem}_{target_h}p{output_path.suffix}")
        ds_writer = cv2.VideoWriter(str(target_path), fourcc, fps, (new_w, new_h))
        if not ds_writer.isOpened():
            continue

        for mosaic in mosaics:
            downsampled = cv2.resize(mosaic, (new_w, new_h), interpolation=cv2.INTER_AREA)
            ds_writer.write(downsampled)
        ds_writer.release()
        downsample_outputs.append((target_path, (new_w, new_h)))

    return downsample_outputs


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

    # Create sweep positions for n_out_frames
    baseline = 0.08  # Default stereo baseline
    sweep_extent = max(0.05, baseline * 2.5)
    
    # Generate exactly n_out_frames with linear sweep from left to right
    if n_out_frames == 1:
        strip_locations = [0.0]
    else:
        strip_locations = list(np.linspace(-abs(sweep_extent), abs(sweep_extent), n_out_frames))

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
        "--strip-ratio", type=float, default=0.10, help="Strip width ratio"
    )
    parser.add_argument(
        "--vertical-scale", type=float, default=1.0, help="Vertical scale factor"
    )
    parser.add_argument(
        "--max-features", type=int, default=1000, help="feature budget per frame"
    )
    parser.add_argument(
        "--blend-workers", type=int, default=0, help="Worker threads for strip blending"
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
        "--convergence-click",
        type=int,
        nargs=2,
        metavar=("X", "Y"),
        default=None,
        help="Optional pixel (x y) in the reference sweep panorama to keep stationary",
    )
    parser.add_argument(
        "--convergence-ref-index",
        type=int,
        default=None,
        help="Optional reference panorama index for convergence alignment; defaults to the middle panorama",
    )
    parser.add_argument(
        "--convergence-patch-radius",
        type=int,
        default=20,
        help="Half-size of the square template patch used when locking the convergence point",
    )
    parser.add_argument(
        "--save-outputs",
        action="store_true",
        help="Save output files (disabled by default for fast testing)",
    )

    return parser.parse_args()


def generate_outputs(
    frames,
    global_stable,
    global_noisy,
    canvas_size,
    offset,
    args,
    fps,
    run_dir,
    rotate_back_code=None,
):
    """Generate all outputs (mosaics, stereo pair, sweep video)"""
    stereo_baseline = args.stereo_baseline_ratio

    # Build main mosaic
    mosaic = build_mosaic(
        frames,
        global_stable,
        global_noisy,
        args.strip_ratio,
        canvas_size,
        offset,
        0.0,
        args.vertical_scale,
    )

    # Build stereo pair
    left = build_mosaic(
        frames,
        global_stable,
        global_noisy,
        args.strip_ratio,
        canvas_size,
        offset,
        -stereo_baseline,
        args.vertical_scale,
    )
    right = build_mosaic(
        frames,
        global_stable,
        global_noisy,
        args.strip_ratio,
        canvas_size,
        offset,
        stereo_baseline,
        args.vertical_scale,
    )

    def restore_orientation(img):
        if rotate_back_code is None:
            return img
        return cv2.rotate(img, rotate_back_code)

    mosaic = restore_orientation(mosaic)
    left = restore_orientation(left)
    right = restore_orientation(right)

    # Apply zoom for stereo effect when convergence-click is supplied
    if args.convergence_click:
        zoom_factor = 1.3
        
        def zoom_image(img, factor):
            h, w = img.shape[:2]
            new_h, new_w = int(h / factor), int(w / factor)
            y_start = (h - new_h) // 2
            x_start = (w - new_w) // 2
            cropped = img[y_start:y_start + new_h, x_start:x_start + new_w]
            return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        
        mosaic = zoom_image(mosaic, zoom_factor)
        left = zoom_image(left, zoom_factor)
        right = zoom_image(right, zoom_factor)

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
        downsampled_sweeps = render_strip_sweep_video(
            frames,
            global_stable,
            global_noisy,
            canvas_size,
            offset,
            strip_ratio=args.strip_ratio,
            vertical_scale=args.vertical_scale,
            output_path=sweep_path,
            sweep_extent=max(0.05, stereo_baseline * 2.5),
            steps=10,
            fps=max(10.0, fps),
            downsample_targets=[(1280, 720), (1920, 1080)],
            convergence_click=tuple(args.convergence_click) if args.convergence_click else None,
            convergence_reference_index=args.convergence_ref_index,
            convergence_patch_radius=args.convergence_patch_radius,
        )
        print(f"Sweep video saved to: {sweep_path}")
        for ds_path, (ds_w, ds_h) in downsampled_sweeps:
            print(f"Downsampled sweep saved to: {ds_path} ({ds_w}x{ds_h})")
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
    canvas_size, offset = compute_canvas_bounds(frames, global_stable)
    print(f"Canvas size {canvas_size}, offset {offset}")

    print("Generating outputs...")
    generate_outputs(
        frames,
        global_stable,
        global_noisy,
        canvas_size,
        offset,
        args,
        fps,
        run_dir,
        rotate_back_code,
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
            global_noisy,
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
