import numpy as np
from pathlib import Path
from PIL import Image

class InferenceDataManager:
    def __init__(self, data_dir, patch_size=256, slices=(20, 45)):
        self.data_dir = Path(data_dir)
        self.patch_size = patch_size
        self.slices = slices
        
        # State variables for the current active fragment
        self.current_fragment_id = None
        self.volume = None
        self.num_patches_x = 0
        self.num_patches_y = 0
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
        
        # Calculate grid (we use floor division, or handle padding if needed)
        # To be safe and cover everything, we often pad or handle the remainder
        self.num_patches_x = w // self.patch_size
        self.num_patches_y = h // self.patch_size
        self.total_patches = self.num_patches_x * self.num_patches_y
        
        print(f"Fragment {fragment_id} loaded. Grid: {self.num_patches_x}x{self.num_patches_y} ({self.total_patches} patches)")

    def get_patch(self, patch_idx):
        """Returns a 3D patch and its top-left coordinates given a grid index."""
        if self.volume is None:
            raise ValueError("No fragment set. Call set_fragment() first.")
            
        # 1. Calculate grid coordinates (row, col)
        row = patch_idx // self.num_patches_x
        col = patch_idx % self.num_patches_x
        
        # 2. Convert to pixel coordinates
        y = row * self.patch_size
        x = col * self.patch_size
        
        # 3. Extract the 3D patch: [Slices, Patch_H, Patch_W]
        z1, z2 = self.slices
        patch_3d = self.volume[z1:z2, y:y+self.patch_size, x:x+self.patch_size]
        
        # Normalize (uint16 -> float32 [0,1])
        patch_3d = patch_3d.astype(np.float32) / 65535.0
        
        return patch_3d, (y, x)

    def __len__(self):
        return self.total_patches
