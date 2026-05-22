import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Import our custom components
from ink_detection_net import InkDetectionNet
from inference_data_manager import InferenceDataManager

CURRENT_DIR = Path(__file__).parent

def run_all_inferences():
    # --- Configuration ---
    test_dir = Path("/Volumes/ZX20/eliva-26-ink-detection/test")
    checkpoint_path = CURRENT_DIR / "checkpoints" / "best_ink_model.pth"
    patch_size = 1024 
    slice_range = (0, 64)
    input_channels = slice_range[1] - slice_range[0] # 64 channels for the model input
    threshold = 0.5 # Threshold for binary mask generation, can be tuned based on validation results
    
    # Check device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # 1. Load Model
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    model = InkDetectionNet(input_channels=input_channels).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded from {checkpoint_path} (Epoch {checkpoint['epoch']})")

    # 2. Initialize Data Manager
    manager = InferenceDataManager(data_dir=test_dir, patch_size=patch_size, slices=slice_range)

    # 3. Process each fragment in the test directory
    # Skip hidden files
    fragment_ids = [d.name for d in test_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    print(f"Found {len(fragment_ids)} fragments to process: {fragment_ids}")

    for fid in fragment_ids:
        print(f"\n--- Processing Fragment: {fid} ---")
        try:
            manager.set_fragment(fid)
        except Exception as e:
            print(f"Skipping {fid}: {e}")
            continue
            
        # Create an empty map to store probabilities (Height, Width)
        full_prob_map = np.zeros(manager.image_shape, dtype=np.float32)
        # Create a counter map for averaging in case of overlap (though here we use exact grid)
        # count_map = np.zeros(manager.image_shape, dtype=np.float32)

        # Iterate through all patches in the grid
        for i in tqdm(range(len(manager)), desc=f"Inference {fid}"):
            patch_3d, (y, x) = manager.get_patch(i)
            
            # Prepare tensor for model: [1, Slices, H, W]
            patch_tensor = torch.from_numpy(patch_3d).unsqueeze(0).to(device)
            
            # Use the 'predict' utility method we added to the model
            with torch.no_grad():
                # probabilities is [1, 1, H, W]
                probabilities = model.predict(patch_tensor)
                prob_np = probabilities.squeeze().cpu().numpy()
            
            # Stitch the patch into the full map
            # Note: If the fragment dimensions aren't exact multiples of patch_size, 
            # some edges might be missed by the current simple grid logic.
            h_p, w_p = prob_np.shape
            full_prob_map[y:y+h_p, x:x+w_p] = prob_np

        # 4. Save results
        # Apply threshold to create binary mask
        binary_mask = (full_prob_map >= threshold).astype(np.uint8) * 255
        
        # Save as PNG
        result_image = Image.fromarray(binary_mask)
        output_path = test_dir / fid / "predictions.png"
        result_image.save(output_path)
        
        # Also save the raw probability map (optional, useful for fine-tuning threshold)
        # np.save(test_dir / fid / "prob_map.npy", full_prob_map)
        
        print(f"Successfully saved prediction to {output_path}")

    print("\nAll inferences complete! 🚀")

if __name__ == "__main__":
    run_all_inferences()
