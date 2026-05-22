import numpy as np
from pathlib import Path
import os 
import cv2
from tqdm import tqdm

def serialize_scan_fragments(base_dir):
    """Serializes the scan fragments into a single Numpy 3D Array (.npy) for easier loading and memory mapping."""
    base_dir = Path(base_dir)
    
    # Find all fragment directories that contain a "surface_volume" subdirectory
    fragment_dirs = [d for d in base_dir.iterdir() if d.is_dir() and (d / "surface_volume").is_dir()]
    
    if not fragment_dirs:
        print(f"No fragment directories found in {base_dir}")
        return

    # Process each fragment directory
    for fragment_dir in fragment_dirs:
        output_file = fragment_dir / "surface_volume_batch.npy"
        
        # skip if already exists to avoid redundant processing
        if output_file.exists():
            print(f"Skipping {fragment_dir.name}, already serialized.")
            continue

        print(f"Processing fragment: {fragment_dir.name}...")
        
        # get all .tif files in the surface_volume directory, ignoring hidden files and SORTING them to maintain correct order
        tif_files = sorted([f for f in os.listdir(fragment_dir / "surface_volume") if f.endswith('.tif') and not f.startswith('.')])
        
        if not tif_files:
            print(f"Warning: No .tif files found in {fragment_dir / 'surface_volume'}")
            continue

        slices = []
        for scan in tqdm(tif_files, desc=f"Loading slices for {fragment_dir.name}"):
            scan_path = fragment_dir / "surface_volume" / scan
            
            scan_data = cv2.imread(str(scan_path), cv2.IMREAD_UNCHANGED)
            if scan_data is not None:
                slices.append(scan_data)
            else:
                raise ValueError(f"Could not read scan: {scan_path}")
            
        # Stack the slices into a 3D volume (Depth, Height, Width)
        volume = np.stack(slices, axis=0)
        
        # Save the consolidated volume as a .npy file (ideal for mmap_mode='r')
        np.save(output_file, volume)
        print(f"Saved consolidated volume for {fragment_dir.name} (Shape: {volume.shape}, Dtype: {volume.dtype})")


def cleanup_files(base_dir, filename):
    """Deletes files with a specific name across all subdirectories of base_dir."""
    base_dir = Path(base_dir)
    print(f"\n--- Cleaning up files named '{filename}' in: {base_dir} ---")
    
    # Find all occurrences of the file
    files_to_delete = list(base_dir.rglob(filename))
    
    if not files_to_delete:
        print(f"No files found named '{filename}'.")
        return

    for f in files_to_delete:
        try:
            os.remove(f)
            print(f"Deleted: {f}")
        except Exception as e:
            print(f"Error deleting {f}: {e}")

if __name__ == "__main__":
    # Define paths to process
    data_paths = [
        "/Volumes/ZX20/eliva-26-ink-detection/train",
        "/Volumes/ZX20/eliva-26-ink-detection/test"
    ]
    # Serialize fragments into .npy
    for path in data_paths:
        if os.path.exists(path):
            print(f"\n--- Processing directory: {path} ---")
            serialize_scan_fragments(path)
        else:
            print(f"\nDirectory not found: {path}")
