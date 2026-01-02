from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from pipeline.io_utils import read_video_frames
from pipeline.mosaic import (
    build_mosaic,
    compute_canvas_bounds,
    render_strip_sweep_video,
    write_video,
)
from pipeline.transforms import (
    cancel_cumulative_rotation,
    detrend_video,
    local_to_global_transformations,
    pairwise_transforms,
)
from pipeline.utils import resolve_workers


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
        print(f"Applying {zoom_factor}x zoom for stereo effect (convergence mode)")
        
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
        steps=10,
        fps=max(10.0, fps),
        downsample_targets=[(1280, 720), (1920, 1080)],
        frame_postprocess=restore_orientation,
        debug_dir=debug_dir,
        convergence_click=tuple(args.convergence_click) if args.convergence_click else None,
        convergence_reference_index=args.convergence_ref_index,
        convergence_patch_radius=args.convergence_patch_radius,
    )
    print(f"Sweep video saved to: {sweep_path}")
    for ds_path, (ds_w, ds_h) in downsampled_sweeps:
        print(f"Downsampled sweep saved to: {ds_path} ({ds_w}x{ds_h})")

    return mosaic, left, right, mosaic_path, left_path, right_path, sweep_path


def generate_panorama(input_frames_path, n_out_frames):
    """
    Main entry point for ex4
    :param input_frames_path : path to a dir with input video frames.
    We will test your code with a dir that has K frames, each in the format
    "frame_i:05d.jpg" (e.g., frame_00000.jpg, frame_00001.jpg, frame_00002.jpg, ...).
    :param n_out_frames: number of generated panorama frames
    :return: A list of generated panorma frames (of size n_out_frames),
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
        debug_dir=None,
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

    panorama_frames = []

    # Create sweep positions for n_out_frames
    # Sweep from -baseline to +baseline
    baseline = 0.08  # Default stereo baseline
    sweep_extent = max(0.05, baseline * 2.5)

    for i in range(n_out_frames):
        if n_out_frames == 1:
            strip_offset = 0.0
        else:
            t = i / (n_out_frames - 1)
            strip_offset = -sweep_extent + 2 * sweep_extent * t

        mosaic = build_mosaic(
            frames,
            global_stable,
            global_noisy,
            strip_ratio,
            canvas_size,
            offset,
            strip_offset,
            vertical_scale,
            debug_dir=None,
            prefix=f"sweep_{i}",
        )

        mosaic_rgb = cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(mosaic_rgb)
        panorama_frames.append(pil_image)

    return panorama_frames


def main() -> None:
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

    mosaic_workers = resolve_workers(args.blend_workers)

    run_dir = (
        Path("output") / f"{args.input.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    debug_dir = run_dir

    args_out = run_dir / "args.txt"
    with args_out.open("w", encoding="utf-8") as f:
        for k, v in sorted(vars(args).items()):
            f.write(f"{k}: {v}\n")

    pairwise = pairwise_transforms(
        frames,
        max_features=args.max_features,
        ransac_thresh=1.5,
        debug_dir=debug_dir,
    )
    print(f"Pairwise feature/RANSAC debug saved to {debug_dir / 'features'} and {debug_dir / 'ransac'}")

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
        debug_dir=debug_dir / "detrend",
    )
    print(f"Detrend debug plot: {debug_dir / 'detrend' / 'detrend_debug.png'}")

    canvas_size, offset = compute_canvas_bounds(frames, global_stable, debug_dir=debug_dir / "canvas")
    print(f"Canvas size {canvas_size}, offset {offset}")

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


if __name__ == "__main__":
    main()
