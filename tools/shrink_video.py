from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

# Common 16:9 resolutions ordered from largest to smallest.
STANDARD_RESOLUTIONS: Dict[str, Tuple[int, int]] = {
    "2160p": (3840, 2160),
    "1440p": (2560, 1440),
    "1080p": (1920, 1080),
    "720p": (1280, 720),
    "480p": (854, 480),
    "360p": (640, 360),
}


def pick_auto_resolution(src_w: int, src_h: int) -> Tuple[str, Tuple[int, int]]:
    for name, (w, h) in STANDARD_RESOLUTIONS.items():
        if src_w >= w and src_h >= h:
            return name, (w, h)
    # Fallback to the smallest option if the source is tiny.
    name, (w, h) = next(reversed(STANDARD_RESOLUTIONS.items()))
    return name, (w, h)


def resolve_target(
    target: str,
    override_w: Optional[int],
    override_h: Optional[int],
    src_w: int,
    src_h: int,
) -> Tuple[str, Tuple[int, int]]:
    if override_w and override_h:
        return "custom", (int(override_w), int(override_h))

    if target == "auto":
        return pick_auto_resolution(src_w, src_h)

    if target.lower() in STANDARD_RESOLUTIONS:
        return target.lower(), STANDARD_RESOLUTIONS[target.lower()]

    raise ValueError(f"Unknown target resolution: {target}")


def resize_frame(frame: np.ndarray, dst_size: Tuple[int, int], mode: str) -> np.ndarray:
    dst_w, dst_h = dst_size
    src_h, src_w = frame.shape[:2]

    if mode == "stretch":
        return cv2.resize(frame, (dst_w, dst_h), interpolation=cv2.INTER_AREA)

    scale = min(dst_w / src_w, dst_h / src_h) if mode == "letterbox" else max(dst_w / src_w, dst_h / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    if mode == "letterbox":
        canvas = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
        x0 = (dst_w - new_w) // 2
        y0 = (dst_h - new_h) // 2
        canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
        return canvas

    # mode == "crop"
    x0 = (new_w - dst_w) // 2
    y0 = (new_h - dst_h) // 2
    return resized[y0 : y0 + dst_h, x0 : x0 + dst_w]


def shrink_video(
    input_path: Path,
    output_path: Path,
    target: str = "1080p",
    mode: str = "letterbox",
    codec: str = "mp4v",
    fps_override: Optional[float] = None,
    target_width: Optional[int] = None,
    target_height: Optional[int] = None,
) -> Tuple[int, int, float]:
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {input_path}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = fps_override or cap.get(cv2.CAP_PROP_FPS) or 25.0

    mode = mode.lower()
    if mode not in {"letterbox", "crop", "stretch"}:
        raise ValueError("mode must be one of: letterbox, crop, stretch")

    name, (dst_w, dst_h) = resolve_target(target, target_width, target_height, src_w, src_h)
    if dst_w <= 0 or dst_h <= 0:
        raise ValueError("Target resolution must be positive")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (dst_w, dst_h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter for {output_path}")

    frames_written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        resized = resize_frame(frame, (dst_w, dst_h), mode)
        writer.write(resized)
        frames_written += 1

    cap.release()
    writer.release()

    if frames_written == 0:
        output_path.unlink(missing_ok=True)
        raise RuntimeError("No frames were written; check the input file")

    return dst_w, dst_h, fps


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Shrink a video to a standard resolution")
    p.add_argument("--input", required=True, type=Path, help="Source video path")
    p.add_argument("--output", type=Path, default=None, help="Destination video path")
    p.add_argument("--target", choices=list(STANDARD_RESOLUTIONS.keys()) + ["auto"], default="1080p", help="Standard resolution name or auto to pick the best that fits")
    p.add_argument("--mode", choices=["letterbox", "crop", "stretch"], default="letterbox", help="Resize strategy: pad to fit, crop to fill, or stretch")
    p.add_argument("--codec", default="mp4v", help="FourCC codec for VideoWriter (e.g., mp4v, avc1, XVID)")
    p.add_argument("--fps", type=float, default=None, help="Override output fps; defaults to source fps")
    p.add_argument("--width", type=int, default=None, help="Custom width (overrides target preset when combined with --height)")
    p.add_argument("--height", type=int, default=None, help="Custom height (overrides target preset when combined with --width)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    default_output = args.input.with_name(f"{args.input.stem}_{args.target}.mp4") if args.target != "auto" else args.input.with_name(f"{args.input.stem}_auto.mp4")
    output_path = args.output or default_output
    dst_w, dst_h, fps = shrink_video(
        input_path=args.input,
        output_path=output_path,
        target=args.target,
        mode=args.mode,
        codec=args.codec,
        fps_override=args.fps,
        target_width=args.width,
        target_height=args.height,
    )
    print(f"Saved {output_path} at {dst_w}x{dst_h} ({fps:.2f} fps) with mode={args.mode}")


if __name__ == "__main__":
    main()
