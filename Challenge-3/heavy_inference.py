import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Import our custom components
from ink_detection_net_v2 import InkDetectionNetV2
from inference_data_manager import InferenceDataManager

CURRENT_DIR = Path(__file__).parent

def get_hanning_window(size):
    win = np.hanning(size)
    return np.outer(win, win).astype(np.float32)

def predict_tta(model, x, device):
    """
    8-way Test-Time Augmentation (Flips + Rotations).
    Works on CUDA, MPS, and CPU.
    """
    # Initialize total_prob on the same device as x
    total_prob = torch.zeros((x.shape[0], 1, x.shape[2], x.shape[3]), device=device)
    
    with torch.no_grad():
        # 1. Original
        total_prob += model.predict(x)
        
        # 2. Horizontal Flip
        total_prob += torch.flip(model.predict(torch.flip(x, dims=[-1])), dims=[-1])
        
        # 3. Vertical Flip
        total_prob += torch.flip(model.predict(torch.flip(x, dims=[-2])), dims=[-2])
        
        # 4. Horizontal + Vertical Flip
        total_prob += torch.flip(model.predict(torch.flip(x, dims=[-1, -2])), dims=[-1, -2])
        
        # 5. Transpose (Rotate 90 and Flip)
        x_t = x.transpose(-1, -2)
        total_prob += model.predict(x_t).transpose(-1, -2)
        
        # 6. Rotate 90
        total_prob += torch.rot90(model.predict(torch.rot90(x, k=1, dims=[-2, -1])), k=-1, dims=[-2, -1])
        
        # 7. Rotate 270
        total_prob += torch.rot90(model.predict(torch.rot90(x, k=3, dims=[-2, -1])), k=-3, dims=[-2, -1])
        
        # 8. Transpose + Flip
        total_prob += torch.flip(model.predict(torch.flip(x_t, dims=[-1])), dims=[-1]).transpose(-1, -2)
    
    return total_prob / 8.0

def run_all_inferences():
    # --- Configuration ---
    test_dir = Path("/Volumes/ZX20/eliva-26-ink-detection/test")

    checkpoint_path = CURRENT_DIR / "checkpoints" / "best_ink_model.pth"
    patch_size = 1024
    stride = patch_size // 4 
    slice_range = (15, 45)
    input_channels = slice_range[1] - slice_range[0]
    threshold = 0.45 
    
    # --- Device Selection ---
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

    model = InkDetectionNetV2(input_channels=input_channels).to(device)
    # Use weights_only=True for security/efficiency, map to the correct device
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded (Epoch {checkpoint['epoch']})")

    # 2. Setup Manager and Blending Window
    manager = InferenceDataManager(data_dir=test_dir, patch_size=patch_size, stride=stride, slices=slice_range)
    hanning_win = get_hanning_window(patch_size)

    # 3. Processing
    fragment_ids = [d.name for d in test_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    for fid in fragment_ids:
        print(f"\n--- Processing Fragment: {fid} ---")
        try:
            manager.set_fragment(fid)
        except Exception as e:
            print(f"Skipping {fid}: {e}")
            continue
            
        full_prob_map = np.zeros(manager.image_shape, dtype=np.float32)
        weight_map = np.zeros(manager.image_shape, dtype=np.float32)

        for i in tqdm(range(len(manager)), desc=f"Inference {fid}"):
            patch_3d, (y, x) = manager.get_patch(i)
            patch_tensor = torch.from_numpy(patch_3d).unsqueeze(0).to(device)
            
            # 8-way TTA
            probabilities = predict_tta(model, patch_tensor, device)
            
            # Transfer back to CPU for aggregation to save GPU/MPS memory
            prob_np = probabilities.squeeze().cpu().numpy()
            
            full_prob_map[y:y+patch_size, x:x+patch_size] += (prob_np * hanning_win)
            weight_map[y:y+patch_size, x:x+patch_size] += hanning_win

        # 4. Final Aggregation and Post-processing
        print(f"Normalizing and thresholding {fid}...")
        final_prob_map = full_prob_map / (weight_map + 1e-7)
        binary_mask = (final_prob_map >= threshold).astype(np.uint8)
        
        # Noise removal
        from skimage import morphology
        labeled = morphology.label(binary_mask)
        clean = morphology.remove_small_objects(labeled, min_size=512)
        binary_mask = (clean > 0).astype(np.uint8) * 255
        
        # Save result
        result_image = Image.fromarray(binary_mask)
        output_path = test_dir / fid / "predictions_extreme.png"
        result_image.save(output_path)
        print(f"Saved: {output_path}")

    print("\nUniversal extreme inference complete! 🚀")

if __name__ == "__main__":
    run_all_inferences()
