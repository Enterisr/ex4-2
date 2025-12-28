from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import cv2

from pipeline.io_utils import read_video_frames
from pipeline.mosaic import (
    build_mosaic,
    compute_canvas_bounds,
    compute_union_coverage_mask,
    crop_with_mask,
    render_strip_sweep_video,
    write_video,
)
from pipeline.transforms import (
    cancel_cumulative_rotation,
    detrend_video,
    local_to_global_transformations,
    lock_convergence_point,
    pairwise_transforms,
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
    parser.add_argument(
        "--process-portrait",
        action="store_true",
        help="If the input video is portrait, rotate it 90 degrees CCW for processing and rotate outputs back to portrait",
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
    mosaic_workers,
    rotate_back_code=None,
    debug_dir=None,
):
    stereo_baseline = args.stereo_baseline_ratio

    coverage_mask = compute_union_coverage_mask(
        frames,
        global_stable,
        canvas_size,
        offset,
        strip_ratio=args.strip_ratio,
        strip_offsets=[0.0, -stereo_baseline, stereo_baseline],
        vertical_scale=args.vertical_scale,
    )

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
            debug_dir,
            "main",
            
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
            debug_dir,
            "stereo_left",
            
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
            debug_dir,
            "stereo_right",
            
        )

        mosaic = mosaic_future.result()
        left = left_future.result()
        right = right_future.result()

    mosaic = crop_with_mask(mosaic, coverage_mask)
    left = crop_with_mask(left, coverage_mask)
    right = crop_with_mask(right, coverage_mask)

    def restore_orientation(img):
        if rotate_back_code is None:
            return img
        return cv2.rotate(img, rotate_back_code)

    mosaic = restore_orientation(mosaic)
    left = restore_orientation(left)
    right = restore_orientation(right)

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
        num_workers=mosaic_workers,
        sweep_extent=max(0.05, stereo_baseline * 2.5),
        steps=60,
        fps=max(10.0, fps),
        downsample_targets=[(1280, 720), (1920, 1080)],
        frame_postprocess=restore_orientation,
        debug_dir=debug_dir,
    )
    print(f"Sweep video saved to: {sweep_path}")
    for ds_path, (ds_w, ds_h) in downsampled_sweeps:
        print(f"Downsampled sweep saved to: {ds_path} ({ds_w}x{ds_h})")

    return mosaic, left, right, mosaic_path, left_path, right_path, sweep_path

def main() -> None:
    args = parse_args()
    frames = read_video_frames(args.input, max_frames=args.max_frames, stride=args.stride)
    fps = cv2.VideoCapture(str(args.input)).get(cv2.CAP_PROP_FPS) or 25.0

    rotate_back_code = None
    if args.process_portrait:
        print("Process-portrait requested; rotating 90 degrees CCW for processing.")
        frames = [cv2.rotate(f, cv2.ROTATE_90_COUNTERCLOCKWISE) for f in frames]
        rotate_back_code = cv2.ROTATE_90_CLOCKWISE


    mosaic_workers = resolve_workers(args.blend_workers)

    run_dir = Path("output") / f"{args.input.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    debug_dir = run_dir

    args_out = run_dir / "args.txt"
    with args_out.open("w", encoding="utf-8") as f:
        for k, v in sorted(vars(args).items()):
            f.write(f"{k}: {v}\n")

    pairwise = pairwise_transforms(
        frames,
        max_features=args.max_features,
        ransac_thresh=3.0,
        debug_dir=debug_dir / "features",
    )
    
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
        rotate_back_code,
        debug_dir,
    )

    if args.stabilized_video:
        stabilized_path = Path(args.stabilized_video)
        frame_postprocess = (lambda img: cv2.rotate(img, rotate_back_code)) if rotate_back_code is not None else None
        write_video(
            frames,
            global_noisy,
            canvas_size,
            offset,
            stabilized_path,
            fps=fps,
            frame_postprocess=frame_postprocess,
        )


if __name__ == "__main__":
    main()