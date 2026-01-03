# Slit-Scan Mosaic Implementation

## Overview

This implementation adds a **Stabilized Slit-Scan (Time-Panorama)** method to the video mosaicing pipeline. This approach solves the horizontal stretching artifacts that occur with the traditional `warpAffine` method when camera motion is slow or stops.

## The Problem

The original `build_mosaic` function uses `cv2.warpAffine` to geometrically warp entire frames onto a canvas. When the camera moves slowly or stops:

- Multiple frames map to similar horizontal positions
- The interpolation process stretches pixels horizontally
- Results in visible artifacts, especially in slow-motion videos (like the Kessaria examples)

## The Solution: Slit-Scan Approach

The new `build_slit_scan_mosaic` function implements a time-based panorama where:

1. **Each frame contributes exactly ONE vertical column (slit)** to the final image
2. **No geometric warping** - slits are extracted directly from the original frames
3. **Position determined by transformation** - the frame's smooth transformation matrix projects the frame center onto the canvas to determine the column position
4. **Vertical stabilization** - the y-component of the transformation compensates for hand jitter
5. **Last-frame-wins policy** - if multiple frames project to the same column (camera stopped), the latest frame overwrites previous ones, preventing stretching

## Mathematical Approach

For each frame $i$ with transformation matrix $M_i$:

1. **Project frame center to canvas**:

   ```
   center_src = [w/2, h/2, 1]
   center_dst = M_i @ center_src
   x_dst = round(center_dst[0])
   y_offset = round(center_dst[1] - h/2)
   ```

2. **Extract slit from source**:

   ```
   slit_x = w * (0.5 + slit_x_offset_ratio)
   slit = frame[:, slit_x:slit_x+slit_width, :]
   ```

3. **Place on canvas with vertical stabilization**:
   ```
   canvas[y_offset:y_offset+h, x_dst, :] = slit
   ```

## Key Features

### 1. No Stretching

Each frame contributes exactly one column, so slow camera motion doesn't cause interpolation artifacts.

### 2. Stabilization

Vertical jitter is compensated by using the y-component of the transformation to vertically shift each column.

### 3. Stereo Support

The `slit_x_offset_ratio` parameter allows extracting slits from different horizontal positions (e.g., left/right of center) for stereo pairs.

### 4. Efficiency

Uses NumPy slicing for fast column extraction and placement.

## Usage

### In `generate_panorama` (already integrated):

```python
panoramas = generate_panorama("input/frames/", n_out_frames=2)
```

The function now uses `build_slit_scan_mosaic` by default for all panorama generation.

### Command-line interface:

```bash
# Use slit-scan method (recommended for slow motion)
python ex4.py --input video.mp4 --use-slit-scan --save-outputs

# Adjust slit width for smoother results
python ex4.py --input video.mp4 --use-slit-scan --slit-width 3 --save-outputs

# Traditional mosaic (for comparison)
python ex4.py --input video.mp4 --save-outputs
```

### Direct function call:

```python
from ex4 import build_slit_scan_mosaic

mosaic = build_slit_scan_mosaic(
    frames=frames,
    transforms_smooth=global_stable,
    canvas_size=canvas_size,
    offset=offset,
    slit_x_offset_ratio=0.0,  # Center slit (0.0), or use ±0.08 for stereo
    slit_width=1,  # Single-pixel slits
)
```

## Function Signature

```python
def build_slit_scan_mosaic(
    frames: Sequence[np.ndarray],
    transforms_smooth: Sequence[np.ndarray],
    canvas_size: Tuple[int, int],
    offset: np.ndarray,
    slit_x_offset_ratio: float = 0.0,
    slit_width: int = 1,
) -> np.ndarray
```

### Parameters

- **frames**: List of BGR video frames (H×W×3 numpy arrays)
- **transforms_smooth**: List of 3×3 smooth transformation matrices (one per frame)
- **canvas_size**: (height, width) of the output panorama
- **offset**: 2D translation offset to apply to all transformations
- **slit_x_offset_ratio**: Horizontal offset for slit extraction
  - `0.0` = extract from center
  - `-0.08` = extract from left (for left stereo view)
  - `+0.08` = extract from right (for right stereo view)
- **slit_width**: Width of slit in pixels (default=1 for sharpest results)

### Returns

- **canvas**: BGR mosaic image (canvas_size[0] × canvas_size[1] × 3)

## Integration Points

The slit-scan method is integrated at two levels:

### 1. `generate_panorama` function (main entry point)

- Automatically uses slit-scan for all panorama generation
- Supports multi-view output for stereo pairs
- No code changes needed by users of this function

### 2. Command-line interface

- New flag `--use-slit-scan` to enable slit-scan mode
- New flag `--slit-width` to control slit thickness
- Backwards compatible - traditional mosaic still available

### 3. `generate_outputs` function

- Updated to support both methods
- Automatically applies the chosen method to main mosaic and stereo pairs

## Comparison: Traditional vs Slit-Scan

| Aspect                 | Traditional Mosaic                   | Slit-Scan Mosaic                    |
| ---------------------- | ------------------------------------ | ----------------------------------- |
| **Approach**           | Warp entire frames with `warpAffine` | Extract 1-pixel columns, no warping |
| **Slow motion**        | Horizontal stretching artifacts      | Clean, no stretching                |
| **Fast motion**        | Good results                         | Good results                        |
| **Stopped camera**     | Massive stretching                   | Last frame wins, no stretching      |
| **Computational cost** | Higher (warping + blending)          | Lower (direct column copy)          |
| **Output quality**     | Spatial mosaic                       | Time panorama                       |

## When to Use Each Method

### Use Slit-Scan (`--use-slit-scan`) when:

- Camera motion is slow or inconsistent
- Camera stops during recording
- You want a true time-based panorama
- You see horizontal stretching in traditional mosaics

### Use Traditional Mosaic when:

- Camera moves at consistent speed
- You want spatial averaging/blending
- You need the strip mask approach

## Testing

Run the test script to compare both methods:

```bash
python test_slit_scan.py input/1_slow
```

## Technical Notes

1. **Transformation matrices**: The function expects 3×3 affine matrices where the last row is [0, 0, 1]
2. **Coordinate system**: Uses standard image coordinates (x horizontal, y vertical, origin top-left)
3. **Border handling**: Frames that project outside the canvas are safely skipped
4. **Memory efficiency**: Canvas is pre-allocated, columns written in-place
5. **Last-frame-wins**: Overlapping columns are overwritten without blending

## Example Output Differences

**Kessaria slow motion (problematic case)**:

- Traditional: Horizontal stretching visible in areas where camera slowed
- Slit-scan: Clean vertical lines, no stretching artifacts

**Fast motion**:

- Traditional: Good spatial mosaic
- Slit-scan: Good time panorama (slightly different aesthetic)
