from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Sequence, Tuple

import cv2
import numpy as np

from .types import Affine3x3
from .utils import progress_iter



def _build_strip_mask(height: int, width: int, strip_ratio: float, strip_x_offset: float) -> np.ndarray:
    strip_width = max(4, int(width * strip_ratio))
    strip_x = width * (0.5 + strip_x_offset)
    left_x_strip = int(strip_x - strip_width * 0.5)
    left_x_strip = max(0, min(width - strip_width, left_x_strip))

    mask = np.zeros((height, width), dtype=np.uint8)
    mask[:, left_x_strip : left_x_strip + strip_width] = 255
    return mask




def compute_canvas_bounds(frames: Sequence[np.ndarray], transforms: Sequence[Affine3x3]) -> Tuple[Tuple[int, int], np.ndarray]:
    h, w = frames[0].shape[:2]
    corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], dtype=np.float32).T
    xs = []
    ys = []
    for T in transforms:
        #find image bounds  per transformation
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
    transforms_smooth: Sequence[Affine3x3],
    transforms_noisy: Sequence[Affine3x3],
    strip_ratio: float,
    canvas_size: Tuple[int, int],
    offset: np.ndarray,
    strip_x_offset: float = 0.0,
    vertical_scale: float = 1.0,
    debug_dir: Path | None = None,
    debug_tag: str | None = None,
    pre_warped_frames: Sequence[np.ndarray] | None = None,
) -> np.ndarray:
    h, orig_width = frames[0].shape[:2]
    mask = _build_strip_mask(h, orig_width, strip_ratio, strip_x_offset)


    acc = np.zeros((canvas_size[0], canvas_size[1], 3), dtype=np.float32)
    count_of_strips_per_px = np.zeros((canvas_size[0], canvas_size[1], 1), dtype=np.float32)

    def warp_task(frame: np.ndarray, warped_frame_cache: np.ndarray | None, T_smooth: Affine3x3):
        # Reuse cached warped frame when available to avoid re-warping per offset.
        warped_frame = warped_frame_cache
        if warped_frame is None:
            warped_frame = warp_frame(frame, T_smooth, canvas_size, offset, border_value=(0, 0, 0)).astype(np.float32)

        # Warp the mask to keep it in the transformed location after stabilization.
        warped_mask = warp_frame(mask, T_smooth, canvas_size, offset, border_value=0)
        m = (warped_mask > 0).astype(np.float32)[..., None]
        return warped_frame * m, m

    for idx, (frame, T_clean, _) in enumerate(
        progress_iter(zip(frames, transforms_smooth, transforms_noisy), total=len(frames), desc="Blending strips")
    ):
        cache_frame = pre_warped_frames[idx] if pre_warped_frames is not None else None
        final_strip, w_map = warp_task(frame, cache_frame, T_clean)
        acc += final_strip
        count_of_strips_per_px += w_map


    count_of_strips_per_px[count_of_strips_per_px == 0] = 1.0
    mosaic = (acc / count_of_strips_per_px).astype(np.uint8)


    return mosaic



def render_strip_sweep_video(
    frames: Sequence[np.ndarray],
    transforms_smooth: Sequence[Affine3x3],
    transforms_noisy: Sequence[Affine3x3],
    canvas_size: Tuple[int, int],
    offset: np.ndarray,
    strip_ratio: float,
    vertical_scale: float,
    output_path: Path,
    num_workers: int,
    sweep_extent: float = 0.4,
    steps: int = 60,
    fps: float =10,
    downsample_targets: Sequence[Tuple[int, int]] | None = None,
    frame_postprocess: Callable[[np.ndarray], np.ndarray] | None = None,
    debug_dir: Path | None = None,
) -> list[tuple[Path, tuple[int, int]]]:
    def apply_post(img: np.ndarray) -> np.ndarray:
        return frame_postprocess(img) if frame_postprocess else img

    strip_locations = np.linspace(-abs(sweep_extent), abs(sweep_extent), steps)
    # Build a bounce sequence so the sweep plays forward and then backward without
    # duplicating the endpoints, e.g., [a, b, c, b].
    strip_locations = np.concatenate([strip_locations, strip_locations[-2:0:-1]])


    # Pre-warp frames once so each sweep offset only needs to warp the mask.
    pre_warped_frames = [
        warp_frame(frame, T_clean, canvas_size, offset, border_value=(0, 0, 0)).astype(np.float32)
        for frame, T_clean in zip(frames, transforms_smooth)
    ]

    first = build_mosaic(
        frames,
        transforms_smooth,
        transforms_noisy,
        strip_ratio=strip_ratio,
        canvas_size=canvas_size,
        offset=offset,
        strip_x_offset=strip_locations[0],
        vertical_scale=vertical_scale,
        debug_dir=debug_dir,
        debug_tag="sweep_first",
        pre_warped_frames=pre_warped_frames,
    )
    first = first
    h, w = first.shape[:2]
    mosaics: list[np.ndarray] = [first]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    with ThreadPoolExecutor(max_workers=max(1, num_workers)) as executor:
        futures = []
        for center_ratio in strip_locations[1:]:
            futures.append(
                executor.submit(
                    build_mosaic,
                    frames,
                    transforms_smooth,
                    transforms_noisy,
                    strip_ratio,
                    canvas_size,
                    offset,
                    center_ratio,
                    vertical_scale,
                    None,
                    None,
                    pre_warped_frames,
                )
            )

        for fut in progress_iter(futures, total=len(futures), desc="Rendering Sweep Video"):
            mosaic = fut.result()
            mosaics.append(apply_post(mosaic))

   
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        print(f"Warning: failed to open sweep VideoWriter for {output_path}")
        return []

    for mosaic in mosaics:
        writer.write(mosaic)
    writer.release()

    downsample_outputs: list[tuple[Path, tuple[int, int]]] = []
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
            print(f"Warning: failed to open downsampled sweep VideoWriter for {target_path}")
            continue

        for mosaic in mosaics:
            downsampled = cv2.resize(mosaic, (new_w, new_h), interpolation=cv2.INTER_AREA)
            ds_writer.write(downsampled)
        ds_writer.release()
        downsample_outputs.append((target_path, (new_w, new_h)))

    return downsample_outputs


def write_video(
    frames: Sequence[np.ndarray],
    transforms: Sequence[Affine3x3],
    canvas_size: Tuple[int, int],
    offset: np.ndarray,
    path: Path,
    fps: float,
    frame_postprocess: Callable[[np.ndarray], np.ndarray] | None = None,
) -> None:
    def apply_post(img: np.ndarray) -> np.ndarray:
        return frame_postprocess(img) if frame_postprocess else img

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    first_frame = apply_post(warp_frame(frames[0], transforms[0], canvas_size, offset))
    writer = cv2.VideoWriter(str(path), fourcc, fps, (first_frame.shape[1], first_frame.shape[0]))
    writer.write(first_frame)

    if len(frames) > 1:
        for frame, T in progress_iter(zip(frames[1:], transforms[1:]), total=len(frames) - 1, desc="Rendering video"):
            warped = apply_post(warp_frame(frame, T, canvas_size, offset))
            writer.write(warped)
    writer.release()
