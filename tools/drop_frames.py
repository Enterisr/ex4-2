import argparse
from pathlib import Path

import cv2


def drop_every_other_frame(input_path: str, output_path: str, slowdown_factor: float = 1.0) -> None:
    """
    Keep all frames from a video and optionally slow down motion.
    
    :param input_path: Path to input video file
    :param output_path: Path to output video file
    :param slowdown_factor: Factor to slow down video (1.0 = no slowdown, 2.0 = half speed)
    """
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {input_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    new_fps = fps / slowdown_factor
    output_frames = total_frames
    
    print(f"Input: {input_path}")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    print(f"Output: {output_path}")
    print(f"  FPS: {new_fps}")
    print(f"  Expected frames: {output_frames}")
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, new_fps, (width, height))
    
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for: {output_path}")
    
    frame_count = 0
    written_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Write all frames
        writer.write(frame)
        written_count += 1
    
    cap.release()
    writer.release()
    
    print(f"Written {written_count} frames")
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drop every other frame from a video to reduce size")
    parser.add_argument("--input", required=True, type=str, help="Input video path")
    parser.add_argument("--output", required=True, type=str, help="Output video path")
    parser.add_argument("--slowdown", type=float, default=1.0, help="Slowdown factor (1.0 = normal, 2.0 = half speed)")
    
    args = parser.parse_args()
    
    drop_every_other_frame(args.input, args.output, args.slowdown)
