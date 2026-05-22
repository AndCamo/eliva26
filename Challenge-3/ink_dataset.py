import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from pathlib import Path
import os

class InkDetectionDataset(Dataset):
    def __init__(self, fragment_ids, data_dir, patch_size=256, slice_range=(20, 45), 
                 samples_per_epoch=1000, ink_prob=0.5, transform=None):
        """
        Args:
            fragment_ids (list): List of fragment directory names.
            data_dir (str/Path): Path to the 'train' directory.
            patch_size (int): Height and width of the 3D patch to extract.
            slice_range (tuple): Range of Z-slices to load (start, end).
            samples_per_epoch (int): Total number of patches to return in one epoch.
            ink_prob (float): Probability of forcing a patch to contain at least some ink.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.fragment_ids = fragment_ids
        self.data_dir = Path(data_dir)
        self.patch_size = patch_size
        self.slice_range = slice_range
        self.samples_per_epoch = samples_per_epoch
        self.ink_prob = ink_prob
        self.transform = transform

        # Caching masks and ink-pixel coordinates for fast sampling
        self.masks = {}
        self.labels = {}
        self.ink_coords = {} # Stores (y, x) coordinates where ink is present
        self.valid_coords = {} # Stores (y, x) coordinates where papsyrus is present (mask.png == 1)

        print("Initializing Dataset and indexing ink coordinates...")
        valid_fragment_ids = []
        for fid in fragment_ids:
            frag_path = self.data_dir / fid
            
            # Load 2D masks
            try:
                mask = np.array(Image.open(frag_path / "mask.png"), dtype=np.uint8)
                label = np.array(Image.open(frag_path / "inklabels.png"), dtype=np.uint8)
            except FileNotFoundError:
                print(f"Skipping fragment {fid}: mask.png or inklabels.png not found.")
                continue

            # Normalize to 0-1
            mask = (mask > 0).astype(np.uint8)
            label = (label > 0).astype(np.uint8)
            
            self.masks[fid] = mask
            self.labels[fid] = label
            
            # Find coordinates where we have ink
            y_ink, x_ink = np.where(label > 0)
            h, w = label.shape
            valid_ink_idx = (y_ink < h - patch_size) & (x_ink < w - patch_size)
            self.ink_coords[fid] = np.stack([y_ink[valid_ink_idx], x_ink[valid_ink_idx]], axis=1)

            # Find all valid coordinates (where papyrus is present)
            y_valid, x_valid = np.where(mask > 0)
            valid_papyrus_idx = (y_valid < h - patch_size) & (x_valid < w - patch_size)
            self.valid_coords[fid] = np.stack([y_valid[valid_papyrus_idx], x_valid[valid_papyrus_idx]], axis=1)

            if len(self.valid_coords[fid]) == 0:
                print(f"Skipping fragment {fid}: No valid papyrus areas found for patch size {patch_size}.")
                continue
            
            valid_fragment_ids.append(fid)
            print(f"Fragment {fid}: {len(self.ink_coords[fid])} ink pixels, {len(self.valid_coords[fid])} papyrus pixels indexed.")
        
        self.fragment_ids = valid_fragment_ids
        if not self.fragment_ids:
            raise ValueError("No valid fragments found for training/validation!")

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        # 1. Pick a random fragment
        fragment_id = np.random.choice(self.fragment_ids)
        
        # 2. Decide if we want an 'ink' patch or a 'random valid papyrus' patch
        if np.random.random() < self.ink_prob and len(self.ink_coords[fragment_id]) > 0:
            # Pick a coordinate from the ink list
            # We pick a random ink pixel and use it as an anchor (randomly positioned within the patch)
            random_idx = np.random.randint(0, len(self.ink_coords[fragment_id]))
            anchor_y, anchor_x = self.ink_coords[fragment_id][random_idx]
            
            # Randomize offset so the ink pixel isn't always at the top-left
            offset_y = np.random.randint(0, self.patch_size)
            offset_x = np.random.randint(0, self.patch_size)
            
            # Calculate top-left corner of the patch, ensuring it stays within bounds
            # max ensures we don't go negative, min ensures we don't go out of bounds
            y = max(0, min(anchor_y - offset_y, self.masks[fragment_id].shape[0] - self.patch_size))
            x = max(0, min(anchor_x - offset_x, self.masks[fragment_id].shape[1] - self.patch_size))
        else:
            # Pick a completely random coordinate from anywhere within the papyrus mask
            random_idx = np.random.randint(0, len(self.valid_coords[fragment_id]))
            y, x = self.valid_coords[fragment_id][random_idx]

        # 3. Load the 3D volume patch using memory mapping
        # IMPORTANT: Open the .npy file here inside __getitem__ to be compatible with num_workers > 0
        volume_path = self.data_dir / fragment_id / "surface_volume_batch.npy"
        
        # mmap_mode='r' makes this extremely fast and RAM-efficient
        volume = np.load(volume_path, mmap_mode='r')
        
        # Extract the 3D patch: [Slices, Patch_H, Patch_W]
        z1, z2 = self.slice_range
        patch_3d = volume[z1:z2, y:y+self.patch_size, x:x+self.patch_size]
        
        # Convert to float and normalize to [0, 1]
        # Since it's uint16, we divide by 65535
        patch_3d = patch_3d.astype(np.float32) / 65535.0
        
        # Extract the label patch
        label_patch = self.labels[fragment_id][y:y+self.patch_size, x:x+self.patch_size]
        
        # Convert to tensors
        volume_tensor = torch.from_numpy(patch_3d) # Shape: [C, H, W]
        label_tensor = torch.from_numpy(label_patch).float() # Shape: [H, W]
        
        # Apply transforms if any
        if self.transform:
            volume_tensor, label_tensor = self.transform(volume_tensor, label_tensor)
            
        return volume_tensor, label_tensor.unsqueeze(0) # Unsqueeze label to [1, H, W] for easier loss calculation
