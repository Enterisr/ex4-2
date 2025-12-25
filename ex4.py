
from __future__ import annotations

import argparse
import math
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


Affine3x3 = np.ndarray


def progress_iter(iterable: Iterable, total: Optional[int] = None, desc: str = ""):
    """Wrap an iterable with tqdm if available to show progress."""
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, leave=False)


def read_video_frames(path: Path, max_frames: Optional[int] = None, stride: int = 1) -> List[np.ndarray]:
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


def harris_corners(
    gray: np.ndarray,
    max_features: int,
    quality_level: float,
    min_distance: int,
    block_size: int,
    k: float = 0.04,
) -> np.ndarray:
    """Manual Harris detector with simple non-max suppression and distance pruning."""
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
    R[R < 0] = 0
    if R.size == 0:
        return np.empty((0, 2), dtype=np.float32)

    R_max = float(R.max())
    if not np.isfinite(R_max) or R_max <= 0:
        return np.empty((0, 2), dtype=np.float32)

    thresh = quality_level * R_max
    mask = R >= thresh
    if not np.any(mask):
        return np.empty((0, 2), dtype=np.float32)

    dilated = cv2.dilate(R, None)
    peaks = (R == dilated) & mask
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


def detect_and_match(gray_a: np.ndarray, gray_b: np.ndarray, max_features: int = 2000, keep: int = 400) -> Tuple[np.ndarray, np.ndarray]:
    """Track manual Harris corners with LK instead of descriptor matching."""
    corners = harris_corners(
        gray_a,
        max_features=max_features,
        quality_level=0.01,
        min_distance=6,
        block_size=7,
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
        flags=0,
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


def estimate_affine(src: np.ndarray, dst: np.ndarray, ransac_thresh: float = 2.0) -> Affine3x3:
    if len(src) < 4 or len(dst) < 4:
        return np.eye(3, dtype=np.float32)
    M, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh)
    if M is None:
        return np.eye(3, dtype=np.float32)
    out = np.eye(3, dtype=np.float32)
    out[:2, :3] = M
    return out


def pairwise_transforms(frames: Sequence[np.ndarray], max_features: int = 2000, ransac_thresh: float = 2.0) -> List[Affine3x3]:
    gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    transforms: List[Affine3x3] = [np.eye(3, dtype=np.float32)]
    for i in progress_iter(range(1, len(gray_frames)), total=len(gray_frames) - 1, desc="Pairwise align"):
        pts_prev, pts_curr = detect_and_match(gray_frames[i - 1], gray_frames[i], max_features=max_features)
        M = estimate_affine(pts_curr, pts_prev, ransac_thresh=ransac_thresh)
        transforms.append(M)
    return transforms


def decompose_affine(M: Affine3x3) -> Tuple[float, float, float, float]:
    a, b, tx = M[0, 0], M[0, 1], M[0, 2]
    c, d, ty = M[1, 0], M[1, 1], M[1, 2]
    angle = math.atan2(c, a)
    scale_x = math.hypot(a, c)
    scale_y = math.hypot(b, d)
    scale = 0.5 * (scale_x + scale_y)
    return tx, ty, angle, scale


def zero_rotation_and_vertical(M: Affine3x3) -> Affine3x3:
    tx, _, _, scale = decompose_affine(M)
    cleaned = np.eye(3, dtype=np.float32)
    cleaned[0, 0] = scale
    cleaned[1, 1] = scale
    cleaned[0, 2] = tx
    return cleaned


def compose_global(pairwise: Sequence[Affine3x3]) -> List[Affine3x3]:
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


def compute_canvas_bounds(frames: Sequence[np.ndarray], transforms: Sequence[Affine3x3]) -> Tuple[Tuple[int, int], np.ndarray]:
    h, w = frames[0].shape[:2]
    corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], dtype=np.float32).T
    xs: List[float] = []
    ys: List[float] = []
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


def warp_frame(frame: np.ndarray, transform: Affine3x3, canvas_size: Tuple[int, int], offset: np.ndarray, border_value: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    warp = transform[:2, :].copy()
    warp[:, 2] += offset
    return cv2.warpAffine(frame, warp, (canvas_size[1], canvas_size[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)


def build_mosaic(
    frames: Sequence[np.ndarray],
    transforms: Sequence[Affine3x3],
    strip_ratio: float,
    canvas_size: Tuple[int, int],
    offset: np.ndarray,
    strip_center_ratio: float = 0.0,
    vertical_scale: float = 1.0,
) -> np.ndarray:
    h, w = frames[0].shape[:2]
    strip_w = max(4, int(w * strip_ratio))
    center_x = w * (0.5 + strip_center_ratio)
    x0 = int(center_x - strip_w * 0.5)
    x0 = max(0, min(w - strip_w, x0))
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[:, x0 : x0 + strip_w] = 255

    acc = np.zeros((canvas_size[0], canvas_size[1], 3), dtype=np.float32)
    weight = np.zeros((canvas_size[0], canvas_size[1], 1), dtype=np.float32)

    for frame, T in progress_iter(zip(frames, transforms), total=len(frames), desc="Blending strips"):
        warped_frame = warp_frame(frame, T, canvas_size, offset, border_value=(0, 0, 0))
        warped_mask = warp_frame(mask, T, canvas_size, offset, border_value=0)
        m = (warped_mask > 0).astype(np.float32)[..., None]
        acc += warped_frame.astype(np.float32) * m
        weight += m

    weight[weight == 0] = 1.0
    mosaic = (acc / weight).astype(np.uint8)

    if not math.isclose(vertical_scale, 1.0):
        new_h = max(1, int(round(mosaic.shape[0] * vertical_scale)))
        mosaic = cv2.resize(mosaic, (mosaic.shape[1], new_h), interpolation=cv2.INTER_CUBIC)

    return mosaic


def render_strip_sweep_video(
    frames: Sequence[np.ndarray],
    transforms: Sequence[Affine3x3],
    canvas_size: Tuple[int, int],
    offset: np.ndarray,
    strip_ratio: float,
    vertical_scale: float,
    output_path: Path,
    sweep_extent: float = 0.4,
    steps: int = 60,
    fps: float = 25.0,
) -> None:
    centers = np.linspace(-abs(sweep_extent), abs(sweep_extent), steps)
    first = build_mosaic(
        frames,
        transforms,
        strip_ratio=strip_ratio,
        canvas_size=canvas_size,
        offset=offset,
        strip_center_ratio=centers[0],
        vertical_scale=vertical_scale,
    )
    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        print(f"Warning: failed to open sweep VideoWriter for {output_path} at size {w}x{h}; skipping sweep video")
        return

    writer.write(first)

    for center_ratio in progress_iter(centers[1:], total=len(centers) - 1, desc="Sweep video"):
        mosaic = build_mosaic(
            frames,
            transforms,
            strip_ratio=strip_ratio,
            canvas_size=canvas_size,
            offset=offset,
            strip_center_ratio=center_ratio,
            vertical_scale=vertical_scale,
        )
        writer.write(mosaic)
    writer.release()


def stabilize_pairwise(pairwise: Sequence[Affine3x3]) -> List[Affine3x3]:
    stabilized = [np.eye(3, dtype=np.float32)]
    for i in range(1, len(pairwise)):
        cleaned = zero_rotation_and_vertical(pairwise[i])
        stabilized.append(cleaned)
    return stabilized


def write_video(frames: Sequence[np.ndarray], transforms: Sequence[Affine3x3], canvas_size: Tuple[int, int], offset: np.ndarray, path: Path, fps: float) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (canvas_size[1], canvas_size[0]))
    for frame, T in progress_iter(zip(frames, transforms), total=len(frames), desc="Rendering video"):
        warped = warp_frame(frame, T, canvas_size, offset)
        writer.write(warped)
    writer.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stereo mosaicing pipeline")
    parser.add_argument("--input", required=True, type=Path, help="Input video path")
    parser.add_argument("--output", type=Path, default=Path("output/mosaic.png"), help="Output mosaic image path")
    parser.add_argument("--stabilized-video", type=Path, default=None, help="Optional path for stabilized preview video")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap on number of frames")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride for subsampling")
    parser.add_argument("--strip-ratio", type=float, default=0.12, help="Strip width ratio used for blending (use smaller for less ghosting)")
    parser.add_argument("--stereo-prefix", type=Path, default=None, help="If set, saves left/right mosaics using this prefix (adds _L.png/_R.png)")
    parser.add_argument("--stereo-baseline-ratio", type=float, default=0.08, help="Horizontal slit offset ratio (relative to frame width) for stereo pair")
    parser.add_argument("--vertical-scale", type=float, default=1.0, help="Vertical scale factor to compensate x-slit aspect distortion")
    parser.add_argument("--lock-point", type=float, nargs=2, default=None, metavar=("X", "Y"), help="Optional convergence point in pixels")
    parser.add_argument("--max-features", type=int, default=2000, help="feature budget per frame")
    parser.add_argument("--ransac-thresh", type=float, default=2.0, help="RANSAC reprojection threshold (pixels) for affine estimation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames = read_video_frames(args.input, max_frames=args.max_frames, stride=args.stride)
    fps = cv2.VideoCapture(str(args.input)).get(cv2.CAP_PROP_FPS) or 25.0

    run_dir = Path("output") / f"{args.input.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    pairwise = pairwise_transforms(frames, max_features=args.max_features, ransac_thresh=args.ransac_thresh)
    stabilized_pairwise = stabilize_pairwise(pairwise)

    global_full = compose_global(pairwise)
    global_stable = compose_global(stabilized_pairwise)

    aligned = lock_convergence_point(global_stable, tuple(args.lock_point) if args.lock_point else None)
    canvas_size, offset = compute_canvas_bounds(frames, aligned)

    mosaic = build_mosaic(
        frames,
        aligned,
        strip_ratio=args.strip_ratio,
        canvas_size=canvas_size,
        offset=offset,
        strip_center_ratio=0.0,
        vertical_scale=args.vertical_scale,
    )

    mosaic_path = run_dir / f"{args.input.stem}_mosaic.png"
    cv2.imwrite(str(mosaic_path), mosaic)

    if args.stereo_prefix is not None:
        baseline = args.stereo_baseline_ratio
        left = build_mosaic(
            frames,
            aligned,
            strip_ratio=args.strip_ratio,
            canvas_size=canvas_size,
            offset=offset,
            strip_center_ratio=-baseline,
            vertical_scale=args.vertical_scale,
        )
        right = build_mosaic(
            frames,
            aligned,
            strip_ratio=args.strip_ratio,
            canvas_size=canvas_size,
            offset=offset,
            strip_center_ratio=baseline,
            vertical_scale=args.vertical_scale,
        )

        args.stereo_prefix.parent.mkdir(parents=True, exist_ok=True)
        left_path = run_dir / f"{args.input.stem}_L.png"
        right_path = run_dir / f"{args.input.stem}_R.png"
        cv2.imwrite(str(left_path), left)
        cv2.imwrite(str(right_path), right)

        sweep_path = run_dir / f"{args.input.stem}_sweep.mp4"
        render_strip_sweep_video(
            frames,
            aligned,
            canvas_size,
            offset,
            strip_ratio=args.strip_ratio,
            vertical_scale=args.vertical_scale,
            output_path=sweep_path,
            sweep_extent=max(0.05, baseline * 2.5),
            steps=80,
            fps=max(10.0, fps),
        )

    video_path = Path(args.stabilized_video) if args.stabilized_video else run_dir / f"{args.input.stem}_mosaic.mp4"
    write_video(frames, aligned, canvas_size, offset, video_path, fps=fps)

    log_path = run_dir / f"{args.input.stem}_transforms.npz"
    np.savez(
        log_path,
        pairwise=pairwise,
        stabilized_pairwise=stabilized_pairwise,
        global_full=global_full,
        global_stable=global_stable,
        aligned=aligned,
    )

    txt_log = run_dir / f"{args.input.stem}_transforms.txt"
    with open(txt_log, "w", encoding="utf-8") as f:
        f.write("parameters\n")
        for k, v in sorted(vars(args).items()):
            f.write(f"{k}: {v}\n")
        f.write("\n")

        def dump(name: str, mats):
            f.write(f"{name}\n")
            for i, M in enumerate(mats):
                f.write(f"#{i}\n{M}\n\n")
        dump("pairwise", pairwise)
        dump("stabilized_pairwise", stabilized_pairwise)
        dump("global_full", global_full)
        dump("global_stable", global_stable)
        dump("aligned", aligned)

    print(f"Run directory: {run_dir}")
    print(f"Mosaic saved to {mosaic_path} with canvas {canvas_size[1]}x{canvas_size[0]}")
    print(f"Mosaic video saved to {video_path}")
    if args.stereo_prefix is not None:
        print(f"Stereo mosaics saved to {left_path} and {right_path}")
        print(f"Stereo sweep video saved to {sweep_path}")
    print(f"Transforms logged to {log_path} and {txt_log}")


if __name__ == "__main__":
    main()
