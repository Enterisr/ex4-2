"""Test module for ex4.py generate_panorama and render_strip_sweep_video"""
from pathlib import Path
import shutil
import cv2
import numpy as np
from ex4 import (
    generate_panorama,
    pairwise_transforms,
    cancel_cumulative_rotation,
    local_to_global_transformations,
    detrend_video,
    compute_canvas_bounds,
    render_strip_sweep_video,
)


def extract_frames_from_video(video_path: Path, output_dir: Path, max_frames: int = 50, stride: int = 2):
    """Extract frames from video file"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        
        if frame_count % stride == 0:
            frame_path = output_dir / f"frame_{saved_count:05d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            saved_count += 1
            
            if max_frames and saved_count >= max_frames:
                break
        
        frame_count += 1
    
    cap.release()
    print(f"Extracted {saved_count} frames from {video_path} to {output_dir}")
    return saved_count


def test_generate_panorama(frames_dir: Path, n_out_frames: int):
    """Test generate_panorama with specified number of output frames"""
    print(f"\n{'='*60}")
    print(f"Testing generate_panorama with n_out_frames={n_out_frames}")
    print(f"Input directory: {frames_dir}")
    
    try:
        result = generate_panorama(str(frames_dir), n_out_frames)
        
        print(f"✓ Success! Generated {len(result)} panorama frames")
        
        if len(result) != n_out_frames:
            print(f"✗ ERROR: Expected {n_out_frames} frames, got {len(result)}")
            return False
        
        # Check that results are PIL images
        for i, img in enumerate(result):
            if not hasattr(img, 'size'):
                print(f"✗ ERROR: Frame {i} is not a PIL Image")
                return False
        
        # Print info about first frame
        if result:
            first = result[0]
            print(f"  Frame dimensions: {first.size[0]}x{first.size[1]}")
            print(f"  Frame mode: {first.mode}")
        
        return True
        
    except Exception as e:
        print(f"✗ FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sweep_video(frames_dir: Path):
    """Test render_strip_sweep_video function"""
    print(f"\n{'='*60}")
    print("Testing render_strip_sweep_video")
    print(f"{'='*60}")
    
    try:
        # Read frames
        frame_files = sorted(frames_dir.glob("frame_*.jpg"))
        frames = [cv2.imread(str(f)) for f in frame_files]
        frames = [f for f in frames if f is not None]
        
        print(f"Loaded {len(frames)} frames")
        
        # Process frames
        pairwise = pairwise_transforms(frames, max_features=300, ransac_thresh=1.5)
        pairwise = cancel_cumulative_rotation(pairwise)
        global_full = local_to_global_transformations(pairwise)
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
        canvas_size, offset = compute_canvas_bounds(frames, global_stable)
        
        print(f"Canvas size: {canvas_size}, offset: {offset}")
        
        # Render sweep video
        output_path = Path("test_sweep_output.mp4")
        downsampled = render_strip_sweep_video(
            frames,
            global_stable,
            global_noisy,
            canvas_size,
            offset,
            strip_ratio=0.06,
            vertical_scale=1.0,
            output_path=output_path,
            sweep_extent=0.2,
            steps=30,
            fps=15,
            downsample_targets=[(1280, 720)],
        )
        
        if output_path.exists():
            print(f"✓ Main sweep video saved to: {output_path}")
        
        for ds_path, (w, h) in downsampled:
            print(f"✓ Downsampled video saved to: {ds_path} ({w}x{h})")
        
        return True
        
    except Exception as e:
        print(f"✗ FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("="*60)
    print("EX4 Generate Panorama & Sweep Video Test Suite")
    print("="*60)
    
    # Setup test directory
    test_dir = Path("test_frames_kessaria")
    video_path = Path("input/Kessaria.mp4")
    
    if not video_path.exists():
        print(f"✗ ERROR: Video not found at {video_path}")
        print("Please ensure the video exists in the input/ directory")
        return
    
    if test_dir.exists():
        print(f"Cleaning up existing test directory: {test_dir}")
        shutil.rmtree(test_dir)
    
    # Extract frames from video
    try:
        num_frames = extract_frames_from_video(video_path, test_dir, max_frames=50, stride=2)
    except Exception as e:
        print(f"✗ ERROR extracting frames: {e}")
        return
    
    # Run tests with different n_out_frames values
    test_cases = [1, 5, 10]
    results = []
    
    for n in test_cases:
        success = test_generate_panorama(test_dir, n)
        results.append((n, success))
    
    # Print panorama test summary
    print(f"\n{'='*60}")
    print("Panorama Generation Test Summary")
    print(f"{'='*60}")
    for n, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  n_out_frames={n:2d}: {status}")
    
    # Test sweep video
    sweep_success = test_sweep_video(test_dir)
    
    # Final summary
    all_passed = all(success for _, success in results) and sweep_success
    print(f"\n{'='*60}")
    if all_passed:
        print("✓ ALL TESTS PASSED!")
    else:
        print("✗ SOME TESTS FAILED")
    print(f"{'='*60}")
    
    # Save sample output for visual inspection
    print("\nSaving sample panorama for visual inspection...")
    result = generate_panorama(str(test_dir), 1)
    if result:
        output_path = Path("test_panorama_single.jpg")
        result[0].save(output_path)
        print(f"Saved single panorama to: {output_path}")
    
    result_multi = generate_panorama(str(test_dir), 10)
    if result_multi:
        for i, img in enumerate(result_multi):
            output_path = Path(f"test_panorama_{i:02d}.jpg")
            img.save(output_path)
        print(f"Saved 10 panorama frames as test_panorama_00.jpg to test_panorama_09.jpg")


if __name__ == "__main__":
    main()
