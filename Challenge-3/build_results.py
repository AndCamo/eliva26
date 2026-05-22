import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def rle_encode(mask: np.ndarray) -> str:
    pixels = mask.astype(np.uint8).ravel()
    pixels = np.concatenate([[0], pixels, [0]])
    changes = np.where(pixels[1:] != pixels[:-1])[0] + 1
    changes[1::2] -= changes[::2]
    return " ".join(str(x) for x in changes)

def build_submission():
    # --- Configuration ---
    test_dir = Path("/Volumes/ZX20/eliva-26-ink-detection/test")
    output_csv = "submission.csv"
    
    print(f"Scanning test directory: {test_dir}")
    
    # List all fragment IDs (subdirectories in test/)
    fragment_ids = sorted([d.name for d in test_dir.iterdir() if d.is_dir() and not d.name.startswith('.')])
    
    results = []
    
    for fid in tqdm(fragment_ids, desc="Generating RLE for fragments"):
        pred_path = test_dir / fid / "predictions.png"
        
        if not pred_path.exists():
            print(f"Warning: No prediction found for {fid} at {pred_path}. Skipping.")
            # If a prediction is missing, we must provide an empty string or a zero mask
            # For robustness, let's add an empty result if missing
            results.append({"Id": fid, "Predicted": ""})
            continue
            
        # Load the binary prediction mask
        mask_img = np.array(Image.open(pred_path))
        
        # Ensure it's 0 or 1 (it was saved as 0 or 255)
        binary_mask = (mask_img > 127).astype(np.uint8)
        
        # Encode
        rle_string = rle_encode(binary_mask)
        
        results.append({
            "Id": fid,
            "Predicted": rle_string
        })
        
    # --- Create DataFrame and Save ---
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    
    print(f"\nSubmission file created: {output_csv}")
    print(f"Total fragments encoded: {len(results)}")

if __name__ == "__main__":
    build_submission()
