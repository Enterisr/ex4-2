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
    """Create a binary strip mask at the requested horizontal offset."""
    strip_width = max(4, int(width * strip_ratio))
    strip_x = width * (0.5 + strip_x_offset)
    left_x_strip = int(strip_x - strip_width * 0.5)
    left_x_strip = max(0, min(width - strip_width, left_x_strip))

    mask = np.zeros((height, width), dtype=np.uint8)
    mask[:, left_x_strip : left_x_strip + strip_width] = 255
    return mask


def _mask_to_bbox(mask: np.ndarray, pad: int = 1) -> tuple[int, int, int, int] | None:
    """Return (y0, y1, x0, x1) for the non-zero area, padded; None if empty."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(mask.shape[0], int(ys.max()) + pad + 1)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(mask.shape[1], int(xs.max()) + pad + 1)
    return (y0, y1, x0, x1)


def crop_with_mask(img: np.ndarray, mask: np.ndarray, pad: int = 4) -> np.ndarray:
    """Crop an image to the non-zero region of a mask."""
    bbox = _mask_to_bbox(mask, pad=pad)
    if bbox is None:
        return img
    y0, y1, x0, x1 = bbox
    return img[y0:y1, x0:x1]


def compute_union_coverage_mask(
    frames: Sequence[np.ndarray],
    transforms_smooth: Sequence[Affine3x3],
    canvas_size: Tuple[int, int],
    offset: np.ndarray,
    strip_ratio: float,
    strip_offsets: Sequence[float],
    vertical_scale: float = 1.0,
) -> np.ndarray:
    """Coverage mask for one or more strip offsets across all frames."""
    h, w = frames[0].shape[:2]
    acc_mask = np.zeros(canvas_size, dtype=np.uint8)

    for strip_offset in strip_offsets:
        base_mask = _build_strip_mask(h, w, strip_ratio, strip_offset)
        for T_clean in transforms_smooth:
            warped_mask = warp_frame(base_mask, T_clean, canvas_size, offset, border_value=0)
            acc_mask = np.maximum(acc_mask, warped_mask)

    if not math.isclose(vertical_scale, 1.0):
        new_h = max(1, int(round(acc_mask.shape[0] * vertical_scale)))
        acc_mask = cv2.resize(acc_mask, (acc_mask.shape[1], new_h), interpolation=cv2.INTER_NEAREST)

    return acc_mask


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
) -> np.ndarray:
    h, orig_width = frames[0].shape[:2]
    mask = _build_strip_mask(h, orig_width, strip_ratio, strip_x_offset)


    acc = np.zeros((canvas_size[0], canvas_size[1], 3), dtype=np.float32)
    count_of_strips_per_px = np.zeros((canvas_size[0], canvas_size[1], 1), dtype=np.float32)

    def warp_task(frame: np.ndarray, T_smooth: Affine3x3, T_with_sttuer: Affine3x3):
        warped_frame = warp_frame(frame, T_with_sttuer, canvas_size, offset, border_value=(0, 0, 0)).astype(np.float32)
        #warp the mask to keep it in the transformed location after stablisliztion
        warped_mask = warp_frame(mask, T_smooth, canvas_size, offset, border_value=0)
        m = (warped_mask > 0).astype(np.float32)[..., None]
        #bitwise to save only masked values (strip)
        return warped_frame * m, m

    sample_warped_mask: np.ndarray | None = None

    for idx, (frame, T_clean, T_noisy) in enumerate(
        progress_iter(zip(frames, transforms_smooth, transforms_noisy), total=len(frames), desc="Blending strips")
    ):
        final_strip, w_map = warp_task(frame, T_clean, T_noisy)
        acc += final_strip
        count_of_strips_per_px += w_map

        if sample_warped_mask is None:
            sample_warped_mask = warp_frame(mask, T_clean, canvas_size, offset, border_value=0)

    count_of_strips_per_px[count_of_strips_per_px == 0] = 1.0
    mosaic = (acc / count_of_strips_per_px).astype(np.uint8)

    if debug_dir is not None:
        tag = debug_tag or "mosaic"
        dbg_root = debug_dir / "blending"
        dbg_root.mkdir(parents=True, exist_ok=True)

        alpha = count_of_strips_per_px.squeeze(axis=2)
        alpha_max = float(alpha.max()) if alpha.size else 0.0
        if alpha_max > 0:
            alpha_vis = np.clip(alpha / alpha_max * 255.0, 0, 255).astype(np.uint8)
            cv2.imwrite(str(dbg_root / f"{tag}_alpha.png"), alpha_vis)
        np.save(dbg_root / f"{tag}_alpha.npy", alpha)

        if sample_warped_mask is not None:
            cv2.imwrite(str(dbg_root / f"{tag}_warped_mask.png"), sample_warped_mask)

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

    coverage_mask = compute_union_coverage_mask(
        frames,
        transforms_smooth,
        canvas_size,
        offset,
        strip_ratio=strip_ratio,
        strip_offsets=strip_locations,
        vertical_scale=vertical_scale,
    )

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
    )
    first = crop_with_mask(apply_post(first), coverage_mask)
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
                )
            )

        for fut in progress_iter(futures, total=len(futures), desc="Rendering Sweep Video"):
            mosaic = fut.result()
            mosaics.append(crop_with_mask(apply_post(mosaic), coverage_mask))

    if debug_dir is not None:
        sweep_debug_dir = debug_dir / "sweep"
        sweep_debug_dir.mkdir(parents=True, exist_ok=True)

        if frames:
            base = frames[0].copy()
            h0, w0 = base.shape[:2]
            strip_w = max(4, int(w0 * strip_ratio))
            color_cycle = [
                (255, 80, 80),
                (80, 220, 120),
                (80, 180, 255),
                (210, 160, 255),
                (120, 200, 255),
                (255, 200, 120),
            ]
            sample_stride = max(1, len(strip_locations) // 6)
            sample_offsets = list(strip_locations[::sample_stride]) + [strip_locations[-1]]

            for idx, offset_ratio in enumerate(sample_offsets):
                strip_x = w0 * (0.5 + offset_ratio)
                left = int(strip_x - strip_w * 0.5)
                left = max(0, min(w0 - strip_w, left))
                color = color_cycle[idx % len(color_cycle)]
                cv2.rectangle(base, (left, 0), (left + strip_w - 1, h0 - 1), color, 2)
                cv2.putText(base, f"{offset_ratio:+.2f}", (left + 4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

            cv2.imwrite(str(sweep_debug_dir / "strip_positions.png"), base)

        sample_ids = sorted(set([0, len(mosaics) // 2, len(mosaics) - 1]))
        for sid in sample_ids:
            if 0 <= sid < len(mosaics):
                cv2.imwrite(str(sweep_debug_dir / f"sweep_frame_{sid:03d}.png"), mosaics[sid])

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
        if scale >= 1.0:
            # Skip if the source is already smaller than the target box.
            continue

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
