import numpy as np
from pathlib import Path
from PIL import Image

class InferenceDataManager:
    def __init__(self, data_dir, patch_size=256, stride=128, slices=(20, 45)):
        self.data_dir = Path(data_dir)
        self.patch_size = patch_size
        self.stride = stride
        self.slices = slices
        
        # State variables for the current active fragment
        self.current_fragment_id = None
        self.volume = None
        self.positions = []
        self.total_patches = 0
        self.image_shape = (0, 0)

    def set_fragment(self, fragment_id):
        """Prepares the manager for a specific fragment. Maps the volume once."""
        self.current_fragment_id = fragment_id
        volume_path = self.data_dir / fragment_id / "surface_volume_batch.npy"
        
        if not volume_path.exists():
            raise FileNotFoundError(f"Serialized volume not found at {volume_path}")
            
        # Map the volume once
        self.volume = np.load(volume_path, mmap_mode='r')
        
        # Original image dimensions from the volume (Depth, Height, Width)
        h, w = self.volume.shape[1], self.volume.shape[2]
        self.image_shape = (h, w)
        
        # Calculate overlapping positions
        self.positions = []
        for y in range(0, h - self.patch_size + 1, self.stride):
            for x in range(0, w - self.patch_size + 1, self.stride):
                self.positions.append((y, x))
        
        # Ensure we cover the right and bottom edges
        if (h - self.patch_size) % self.stride != 0:
            for x in range(0, w - self.patch_size + 1, self.stride):
                self.positions.append((h - self.patch_size, x))
        
        if (w - self.patch_size) % self.stride != 0:
            for y in range(0, h - self.patch_size + 1, self.stride):
                self.positions.append((y, w - self.patch_size))
                
        # Handle the bottom-right corner explicitly
        self.positions.append((h - self.patch_size, w - self.patch_size))
        
        # Remove duplicates and sort for deterministic behavior
        self.positions = sorted(list(set(self.positions)))
        self.total_patches = len(self.positions)
        
        print(f"Fragment {fragment_id} loaded. Image size: {w}x{h}. Total overlapping patches: {self.total_patches}")

    def get_patch(self, patch_idx):
        """Returns a 3D patch and its top-left coordinates given an index."""
        if self.volume is None:
            raise ValueError("No fragment set. Call set_fragment() first.")
            
        y, x = self.positions[patch_idx]
        
        # Extract the 3D patch: [Slices, Patch_H, Patch_W]
        z1, z2 = self.slices
        patch_3d = self.volume[z1:z2, y:y+self.patch_size, x:x+self.patch_size]
        
        # Normalize (uint16 -> float32 [0,1])
        patch_3d = patch_3d.astype(np.float32) / 65535.0
        
        return patch_3d, (y, x)

    def __len__(self):
        return self.total_patches
