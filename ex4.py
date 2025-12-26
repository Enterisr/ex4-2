from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import cv2

from pipeline.io_utils import read_video_frames
from pipeline.mosaic import build_mosaic, compute_canvas_bounds, render_strip_sweep_video, write_video
from pipeline.transforms import (
    cancel_cumulative_rotation,
    local_to_global_transformations,
    pairwise_transforms,
    detrend_video,
)
from pipeline.utils import resolve_workers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stereo mosaicing pipeline")
    parser.add_argument("--input", required=True, type=Path, help="Input video path")
    parser.add_argument("--output", type=Path, default=Path("output/mosaic.png"), help="Output mosaic image path")
    parser.add_argument("--stabilized-video", type=Path, default=None, help="Optional path for stabilized preview video")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap on number of frames")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride for subsampling")
    parser.add_argument("--strip-ratio", type=float, default=0.10, help="Strip width ratio")
    parser.add_argument("--vertical-scale", type=float, default=1.0, help="Vertical scale factor")
    parser.add_argument("--max-features", type=int, default=1000, help="feature budget per frame")
    parser.add_argument("--blend-workers", type=int, default=0, help="Worker threads for strip blending")
    parser.add_argument("--smooth-radius", type=int, default=25, help="Radius for trajectory smoothing")    
    parser.add_argument("--stereo-baseline-ratio", type=float, default=0.08, help="Horizontal slit offset ratio (relative to frame width) for stereo pair")
    parser.add_argument("--lock-point", type=float, nargs=2, default=None, metavar=("X", "Y"), help="Optional convergence point in pixels")
    
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
    mosaic_workers,
):
    stereo_baseline = args.stereo_baseline_ratio

    with ThreadPoolExecutor(max_workers=mosaic_workers) as executor:
        mosaic_future = executor.submit(
            build_mosaic,
            frames,
            global_stable,
            global_noisy,
            args.strip_ratio,
            canvas_size,
            offset,
            0.0,
            args.vertical_scale,
            
        )

        print(f"Generating stereo pair with baseline {stereo_baseline}...")
        left_future = executor.submit(
            build_mosaic,
            frames,
            global_stable,
            global_noisy,
            args.strip_ratio,
            canvas_size,
            offset,
            -stereo_baseline,
            args.vertical_scale,
            
        )
        right_future = executor.submit(
            build_mosaic,
            frames,
            global_stable,
            global_noisy,
            args.strip_ratio,
            canvas_size,
            offset,
            stereo_baseline,
            args.vertical_scale,
            
        )

        mosaic = mosaic_future.result()
        left = left_future.result()
        right = right_future.result()

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
        global_noisy,
        canvas_size,
        offset,
        strip_ratio=args.strip_ratio,
        vertical_scale=args.vertical_scale,
        output_path=sweep_path,
        num_workers=mosaic_workers,
        sweep_extent=max(0.05, stereo_baseline * 2.5),
        steps=60,
        fps=max(10.0, fps),
    )
    print(f"Sweep video saved to: {sweep_path}")

    return mosaic, left, right, mosaic_path, left_path, right_path, sweep_path

def main() -> None:
    args = parse_args()
    frames = read_video_frames(args.input, max_frames=args.max_frames, stride=args.stride)
    fps = cv2.VideoCapture(str(args.input)).get(cv2.CAP_PROP_FPS) or 25.0

    mosaic_workers = resolve_workers(args.blend_workers)

    run_dir = Path("output") / f"{args.input.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    pairwise = pairwise_transforms(frames, max_features=args.max_features, ransac_thresh=3.0)
    pairwise = cancel_cumulative_rotation(pairwise)

    global_full = local_to_global_transformations(pairwise)
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

    canvas_size, offset = compute_canvas_bounds(frames, global_stable)

    generate_outputs(
        frames,
        global_stable,
        global_noisy,
        canvas_size,
        offset,
        args,
        fps,
        run_dir,
        mosaic_workers,
    )

    if args.stabilized_video:
        stabilized_path = Path(args.stabilized_video)
        write_video(frames, global_noisy, canvas_size, offset, stabilized_path, fps=fps)


if __name__ == "__main__":
    main()