from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np


def read_video_frames(path: Path, max_frames: Optional[int] = None, stride: int = 1) -> List[np.ndarray]:
    """Read frames from a video, optionally sub-sampling by stride and limiting count."""
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
