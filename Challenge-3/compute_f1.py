import numpy as np
from PIL import Image
from pathlib import Path

def calculate_f1(pred_mask, gt_label, papyrus_mask):
    # Robust thresholding: any value > 0 is considered 1
    # This handles 0/1 images, 0/255 images, and anything in between
    pred = (pred_mask > 127).astype(bool)
    gt = (gt_label > 0).astype(bool)
    mask = (papyrus_mask > 0).astype(bool)
    
    # Apply papyrus mask
    pred = pred & mask
    gt = gt & mask
    
    tp = np.sum(pred & gt)
    fp = np.sum(pred & ~gt)
    fn = np.sum(~pred & gt)
    
    pred_count = np.sum(pred)
    gt_count = np.sum(gt)
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    return f1, precision, recall, pred_count, gt_count

def run_evaluation():
    train_dir = Path("/Volumes/ZX20/eliva-26-ink-detection/train")
    fragment_ids = sorted([d.name for d in train_dir.iterdir() if d.is_dir() and not d.name.startswith('.')])
    
    results = []
    print(f"{'Fragment ID':<20} | {'Pred Ink':<10} | {'GT Ink':<10} | {'F1 Score':<10}")
    print("-" * 60)

    for fid in fragment_ids:
        frag_path = train_dir / fid
        pred_path = frag_path / "predictions.png"
        label_path = frag_path / "inklabels.png"
        mask_path = frag_path / "mask.png"
        
        if not pred_path.exists() or not label_path.exists():
            continue
            
        # Use PIL and convert to numpy array
        pred_img = np.array(Image.open(pred_path))
        label_img = np.array(Image.open(label_path))
        mask_img = np.array(Image.open(mask_path))
        
        f1, p, r, p_cnt, g_cnt = calculate_f1(pred_img, label_img, mask_img)
        
        results.append(f1)
        print(f"{fid:<20} | {p_cnt:<10} | {g_cnt:<10} | {f1:.4f}")

    if results:
        print(f"\nMean F1: {np.mean(results):.4f}")
    else:
        print("\nNo valid data pairs found.")

if __name__ == "__main__":
    run_evaluation()
