"""
Test script to demonstrate the slit-scan method and compare with traditional mosaic.
"""
from pathlib import Path
import sys

# Import the main module
from ex4 import generate_panorama

def test_slit_scan(input_dir: str, n_frames: int = 2):
    """
    Test the slit-scan panorama generation.
    
    Args:
        input_dir: Path to directory containing frame_*.jpg files
        n_frames: Number of output frames (for stereo, use 2)
    """
    print(f"Testing slit-scan panorama generation...")
    print(f"Input directory: {input_dir}")
    print(f"Number of output frames: {n_frames}")
    
    try:
        panoramas = generate_panorama(input_dir, n_frames)
        
        print(f"\n✓ Successfully generated {len(panoramas)} panorama frames")
        print(f"  Frame sizes: {[f'{img.size[0]}x{img.size[1]}' for img in panoramas]}")
        
        # Save the results
        output_dir = Path("output") / "slit_scan_test"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for i, pano in enumerate(panoramas):
            output_path = output_dir / f"panorama_{i:02d}.png"
            pano.save(output_path)
            print(f"  Saved: {output_path}")
        
        return panoramas
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Example usage - adjust the path to your test data
    test_dirs = [
        "input/1",  # Fast motion test
        "input/1_slow",  # Slow motion test (where stretching was visible)
    ]
    
    if len(sys.argv) > 1:
        test_dirs = [sys.argv[1]]
    
    for test_dir in test_dirs:
        test_path = Path(test_dir)
        if test_path.exists():
            print("\n" + "="*70)
            print(f"Testing: {test_dir}")
            print("="*70)
            test_slit_scan(str(test_path), n_frames=2)
        else:
            print(f"Skipping {test_dir} (not found)")
