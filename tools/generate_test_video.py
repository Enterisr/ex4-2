"""
Small synthetic video generator to quickly test ex4.py without waiting on real footage.

Default mode simulates a static scene with a panning virtual camera so the mosaic
looks sane (no moving objects causing ghosting). An "objects" mode keeps the old
moving-shape demo if you want to stress-test dynamic content.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def make_background(height: int, width: int) -> np.ndarray:
    # Horizontal/vertical gradients give texture for optical flow.
    x_grad = np.tile(np.linspace(40, 180, width, dtype=np.uint8), (height, 1))
    y_grad = np.tile(np.linspace(40, 160, height, dtype=np.uint8)[:, None], (1, width))
    bg = np.stack([x_grad, y_grad, (x_grad + y_grad) // 2], axis=-1)
    return bg


def draw_static_scene(canvas: np.ndarray) -> None:
    """Paint a few anchored shapes on a large canvas to mimic a static environment."""
    h, w = canvas.shape[:2]
    # Big rectangles at different depths/colors.
    rects = [
        ((int(w * 0.15), int(h * 0.25)), (int(w * 0.4), int(h * 0.6)), (40, 160, 240)),
        ((int(w * 0.55), int(h * 0.35)), (int(w * 0.85), int(h * 0.75)), (200, 120, 60)),
        ((int(w * 0.32), int(h * 0.65)), (int(w * 0.6), int(h * 0.9)), (90, 200, 140)),
    ]
    for tl, br, color in rects:
        cv2.rectangle(canvas, tl, br, color, thickness=-1)
        cv2.rectangle(canvas, tl, br, (0, 0, 0), thickness=2)

    # Circles sprinkled around for corners/texture.
    centers = [
        (int(w * 0.25), int(h * 0.2)),
        (int(w * 0.7), int(h * 0.18)),
        (int(w * 0.52), int(h * 0.55)),
        (int(w * 0.18), int(h * 0.78)),
        (int(w * 0.82), int(h * 0.72)),
    ]
    radii = [max(10, min(w, h) // r) for r in (18, 20, 22, 24, 26)]
    colors = [(80, 80, 220), (220, 80, 120), (80, 200, 220), (200, 200, 80), (120, 80, 220)]
    for (cx, cy), radius, color in zip(centers, radii, colors):
        cv2.circle(canvas, (cx, cy), radius, color, thickness=-1)
        cv2.circle(canvas, (cx, cy), radius, (0, 0, 0), thickness=2)

    # Diagonal lines for parallax cues.
    for frac, color in ((0.18, (40, 60, 200)), (0.32, (40, 120, 120)), (0.46, (60, 60, 60))):
        y = int(h * frac)
        cv2.line(canvas, (0, y), (w, y + int(h * 0.08)), color, thickness=2)

    cv2.putText(canvas, "static scene", (int(w * 0.05), int(h * 0.95)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)


def draw_shapes(frame: np.ndarray, t: float, center: Tuple[int, int]) -> None:
    h, w = frame.shape[:2]
    cx, cy = center
    # Moving rectangle.
    rect_w = max(40, w // 6)
    rect_h = max(40, h // 6)
    dx = int(0.2 * w * np.sin(2 * np.pi * t))
    dy = int(0.1 * h * np.cos(2 * np.pi * t))
    top_left = (cx - rect_w // 2 + dx, cy - rect_h // 2 + dy)
    bottom_right = (top_left[0] + rect_w, top_left[1] + rect_h)
    cv2.rectangle(frame, top_left, bottom_right, (30, 200, 240), thickness=-1)
    cv2.rectangle(frame, top_left, bottom_right, (0, 0, 0), thickness=2)

    # Orbiting circle.
    radius = max(12, min(w, h) // 12)
    orbit_r = max(20, min(w, h) // 8)
    angle = 2 * np.pi * t
    circle_center = (
        cx + int(orbit_r * np.cos(angle)),
        cy + int(orbit_r * np.sin(angle)),
    )
    cv2.circle(frame, circle_center, radius, (240, 80, 80), thickness=-1)
    cv2.circle(frame, circle_center, radius, (0, 0, 0), thickness=2)

    # Guiding line to add corners.
    line_y = int(h * (0.3 + 0.2 * np.sin(4 * np.pi * t)))
    cv2.line(frame, (0, line_y), (w, line_y), (60, 60, 200), thickness=2)


def add_noise(frame: np.ndarray, amount: float = 0.02) -> None:
    if amount <= 0:
        return
    noise = np.random.normal(0, 255 * amount, frame.shape).astype(np.int16)
    noisy = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    frame[...] = noisy


def generate_object_motion(output: Path, width: int, height: int, frames: int, fps: float, noise: float) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    bg = make_background(height, width)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter for {output}")

    center = (width // 2, height // 2)
    for i in range(frames):
        t = i / max(1, frames - 1)
        frame = bg.copy()
        draw_shapes(frame, t, center)
        add_noise(frame, noise)
        cv2.putText(frame, f"t={t:.2f}", (12, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
        writer.write(frame)

    writer.release()


def generate_camera_pan(
    output: Path,
    width: int,
    height: int,
    frames: int,
    fps: float,
    noise: float,
    canvas_scale: float,
    pan_amplitude: float,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas_w = int(round(width * canvas_scale))
    canvas_h = int(round(height * canvas_scale))
    canvas = make_background(canvas_h, canvas_w)
    draw_static_scene(canvas)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter for {output}")

    base_x = (canvas_w - width) // 2
    base_y = (canvas_h - height) // 2
    max_dx = max(0, canvas_w - width)
    max_dy = max(0, canvas_h - height)
    dy_amp = int(min(height * 0.12, max_dy * 0.2))

    for i in range(frames):
        t = i / max(1, frames - 1)
        dx = int(pan_amplitude * 0.5 * max_dx * np.sin(2 * np.pi * t))
        dy = int(dy_amp * np.sin(4 * np.pi * t + 0.6))
        x0 = int(np.clip(base_x + dx, 0, canvas_w - width))
        y0 = int(np.clip(base_y + dy, 0, canvas_h - height))
        frame = canvas[y0 : y0 + height, x0 : x0 + width].copy()
        add_noise(frame, noise)
        cv2.putText(frame, f"t={t:.2f}", (12, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
        writer.write(frame)

    writer.release()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a tiny synthetic video for quick ex4.py checks")
    p.add_argument("--output", type=Path, default=Path("input/test_clip.mp4"), help="Where to save the test video")
    p.add_argument("--width", type=int, default=480, help="Frame width")
    p.add_argument("--height", type=int, default=270, help="Frame height")
    p.add_argument("--frames", type=int, default=90, help="Number of frames to render")
    p.add_argument("--fps", type=float, default=30.0, help="Frames per second")
    p.add_argument("--noise", type=float, default=0.015, help="Gaussian noise level (0 disables noise)")
    p.add_argument("--mode", choices=["pan", "objects"], default="pan", help="Pan a static scene (default) or animate objects")
    p.add_argument("--canvas-scale", type=float, default=1.8, help="Scale of virtual canvas vs output when mode=pan")
    p.add_argument("--pan-amplitude", type=float, default=0.35, help="Horizontal pan amplitude as fraction of available margin when mode=pan")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "pan":
        generate_camera_pan(
            args.output,
            args.width,
            args.height,
            args.frames,
            args.fps,
            args.noise,
            canvas_scale=args.canvas_scale,
            pan_amplitude=args.pan_amplitude,
        )
    else:
        generate_object_motion(args.output, args.width, args.height, args.frames, args.fps, args.noise)
    print(f"Test video written to {args.output} ({args.width}x{args.height}, {args.frames} frames @ {args.fps} fps, mode={args.mode})")


if __name__ == "__main__":
    main()
