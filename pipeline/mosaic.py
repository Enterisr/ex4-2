from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Sequence, Tuple

import cv2
import numpy as np

from .types import Affine3x3
from .utils import progress_iter


def compute_canvas_bounds(frames: Sequence[np.ndarray], transforms: Sequence[Affine3x3]) -> Tuple[Tuple[int, int], np.ndarray]:
    h, w = frames[0].shape[:2]
    corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], dtype=np.float32).T
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
) -> np.ndarray:
   
    h, orig_width = frames[0].shape[:2]
    strip_width = max(4, int(orig_width * strip_ratio))
    strip_x = orig_width * (0.5 + strip_x_offset)
    left_x_strip = int(strip_x - strip_width * 0.5)

    #clamp leftx so it iwll not crash on negative\pixels that are too far right
    left_x_strip = max(0, min(orig_width - strip_width, left_x_strip))

    #binary mask for strip
    mask = np.zeros((h, orig_width), dtype=np.uint8)
    mask[:, left_x_strip : left_x_strip + strip_width] = 255


    acc = np.zeros((canvas_size[0], canvas_size[1], 3), dtype=np.float32)
    count_of_strips_per_px = np.zeros((canvas_size[0], canvas_size[1], 1), dtype=np.float32)

    def warp_task(frame: np.ndarray, T_smooth: Affine3x3, T_with_sttuer: Affine3x3):
        warped_frame = warp_frame(frame, T_with_sttuer, canvas_size, offset, border_value=(0, 0, 0)).astype(np.float32)
        #warp the mask to keep it in the transformed location after stablisliztion
        warped_mask = warp_frame(mask, T_smooth, canvas_size, offset, border_value=0)
        m = (warped_mask > 0).astype(np.float32)[..., None]
        #bitwise to save only masked values (strip)
        return warped_frame * m, m

    for frame, T_clean, T_noisy in progress_iter(zip(frames, transforms_smooth, transforms_noisy), total=len(frames), desc="Blending strips"):
        final_strip, w_map = warp_task(frame, T_clean, T_noisy)
        acc += final_strip
        count_of_strips_per_px += w_map

    count_of_strips_per_px[count_of_strips_per_px == 0] = 1.0
    mosaic = (acc / count_of_strips_per_px).astype(np.uint8)

    if not math.isclose(vertical_scale, 1.0):
        new_h = max(1, int(round(mosaic.shape[0] * vertical_scale)))
        mosaic = cv2.resize(mosaic, (mosaic.shape[1], new_h), interpolation=cv2.INTER_CUBIC)

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
    fps: float = 25.0,
) -> None:
    centers = np.linspace(-abs(sweep_extent), abs(sweep_extent), steps)

    first = build_mosaic(
        frames,
        transforms_smooth,
        transforms_noisy,
        strip_ratio=strip_ratio,
        canvas_size=canvas_size,
        offset=offset,
        strip_x_offset=centers[0],
        vertical_scale=vertical_scale,
    )
    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    if not writer.isOpened():
        print(f"Warning: failed to open sweep VideoWriter for {output_path}")
        return

    writer.write(first)

    with ThreadPoolExecutor(max_workers=max(1, num_workers)) as executor:
        futures = []
        for center_ratio in centers[1:]:
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
                    
                )
            )

        for fut in progress_iter(futures, total=len(futures), desc="Rendering Sweep Video"):
            mosaic = fut.result()
            writer.write(mosaic)

    writer.release()


def write_video(
    frames: Sequence[np.ndarray],
    transforms: Sequence[Affine3x3],
    canvas_size: Tuple[int, int],
    offset: np.ndarray,
    path: Path,
    fps: float,
) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (canvas_size[1], canvas_size[0]))
    for frame, T in progress_iter(zip(frames, transforms), total=len(frames), desc="Rendering video"):
        warped = warp_frame(frame, T, canvas_size, offset)
        writer.write(warped)
    writer.release()
