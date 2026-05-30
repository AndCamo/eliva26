import os
from pathlib import Path

# --- System Settings ---
# Route temporary files to the fast and large storage drive
fast_tmp_dir = str(Path.home() / "Vesuvius-Challenge" / "torch_tmp")
os.environ["TMPDIR"] = fast_tmp_dir
# We removed the CUDA_VISIBLE_DEVICES lock since all 4 GPUs are currently free.
# PyTorch will automatically bind to logical GPU 0.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
import random
import numpy as np
from tqdm import tqdm
import wandb

# Custom Modules
from ink_dataset import InkDetectionDataset
from ink_detection_net_v2 import InkDetectionNetV2


def wandb_enabled():
    return os.getenv("DISABLE_WANDB", "0").lower() not in ("1", "true", "yes")


# ==============================================================================
# LOSS FUNCTION
# ==============================================================================
class FocalDiceLoss(nn.Module):
    """
    Combines Focal Loss and Dice Loss.
    Focal Loss dynamically scales the cross entropy based on prediction confidence,
    forcing the model to focus on hard, misclassified examples (like sparse ink).
    """
    def __init__(self, alpha=0.25, gamma=2.0, focal_weight=0.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.focal_weight = focal_weight

    def forward(self, inputs, targets):
        # 1. Focal Loss (numerically stable)
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss) 
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        focal_loss = focal_loss.mean()
        
        # 2. Dice Loss
        inputs_sigmoid = torch.sigmoid(inputs)
        smooth = 1e-5
        
        intersection = (inputs_sigmoid * targets).sum(dim=(2, 3))
        union = inputs_sigmoid.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        
        dice_loss = 1.0 - (2.0 * intersection + smooth) / (union + smooth)
        dice_loss = dice_loss.mean() 

        return self.focal_weight * focal_loss + (1.0 - self.focal_weight) * dice_loss


# ==============================================================================
# DATA AUGMENTATION
# ==============================================================================
def apply_d4_augmentation(volume, label):
    """
    Applies random combinations of 90-degree rotations and flips (D4 Symmetry).
    This creates 8 possible unique orientations for every patch, 
    drastically reducing overfitting on the papyrus textures.
    """
    # Random horizontal flip
    if random.random() > 0.5:
        volume = torch.flip(volume, dims=[-1])
        label = torch.flip(label, dims=[-1])
        
    # Random vertical flip
    if random.random() > 0.5:
        volume = torch.flip(volume, dims=[-2])
        label = torch.flip(label, dims=[-2])
        
    # Random 90-degree rotation (0, 1, 2, or 3 times)
    k = random.randint(0, 3)
    if k > 0:
        volume = torch.rot90(volume, k, dims=[-2, -1])
        label = torch.rot90(label, k, dims=[-2, -1])
        
    return volume, label


def get_optimizer(model):
    """
    Sets up Differential Learning Rates.
    """
    backbone_params = []
    custom_params = []
    
    for name, param in model.named_parameters():
        if 'encoder' in name and 'conv_stem' not in name:
            backbone_params.append(param)
        else:
            custom_params.append(param)
            
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': 5e-6},
        {'params': custom_params, 'lr': 1e-4}
    ], weight_decay=1e-4)
    
    return optimizer


# ==============================================================================
# MAIN TRAINING PIPELINE
# ==============================================================================
def train_model():
    # --- Configuration ---
    data_dir = "./data/train"
    epochs = 100 
    batch_size = 16
    patch_size = 256
    
    # Pruning the Z-axis: we only load slices 15 to 45 (where ink usually resides)
    # This prevents the model from learning surface noise or empty air.
    slice_range = (15, 45)  
    input_channels = slice_range[1] - slice_range[0]  # 30 channels
    
    num_workers = 0
    patience_limit = 10
    patience_counter = 0
    use_wandb = wandb_enabled()
    
    # 1. Initialize W&B
    if use_wandb:
        wandb.init(
            project="challenge-3-ink-detection",
            config={
                "encoder_lr": 5e-6,
                "decoder_lr": 1e-4,
                "epochs": epochs,
                "batch_size": batch_size,
                "patch_size": patch_size,
                "z_slices": input_channels,
                "architecture": "EfficientNetV2-L-UNet",
                "loss": "FocalDiceLoss"
            }
        )

    # Hardware Selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Splitting ---
    all_fragments = [d.name for d in Path(data_dir).iterdir() if d.is_dir() and not d.name.startswith('.')]
    if len(all_fragments) < 5:
        raise ValueError("Need at least 5 fragments total to support 4 validation fragments.")
        
    val_fragments = ['p_5qlsf5h5b1yy', 'p_hwx6h1ybz19s', 'p_cls0w7rbx6e4', 'p_yjx64pa5da7o']
    train_fragments = [frag for frag in all_fragments if frag not in val_fragments]
    
    print(f"Training on: {len(train_fragments)} fragments")
    print(f"Validating on: {val_fragments}")

    # --- Datasets and DataLoaders ---
    train_dataset = InkDetectionDataset(
        fragment_ids=train_fragments, data_dir=data_dir, patch_size=patch_size, 
        samples_per_epoch=2500, ink_prob=0.3, slice_range=slice_range, augment=True
    )
    val_dataset = InkDetectionDataset(
        fragment_ids=val_fragments, data_dir=data_dir, patch_size=patch_size, 
        samples_per_epoch=800, ink_prob=0, slice_range=slice_range, augment=False
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # --- Model, Loss, Optimizer ---
    model = InkDetectionNetV2(input_channels=input_channels).to(device)
    
    # Initial Backbone Freezing
    print("Freezing encoder for the first 3 epochs...")
    for name, param in model.encoder.named_parameters():
        if "conv_stem" not in name:
            param.requires_grad = False
            
    # Using the advanced Focal+Dice loss to handle severe class imbalance
    criterion = FocalDiceLoss(focal_weight=0.5)
    optimizer = get_optimizer(model)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

    best_val_loss = float('inf')
    save_dir = Path("checkpoints")
    save_dir.mkdir(exist_ok=True)
    
    # Dynamic Scaler initialization (fixes potential CPU fallback crashes)
    scaler = torch.amp.GradScaler(device.type)
    
    # --- Training Loop ---
    for epoch in range(epochs):
        # Gradual Unfreezing
        if epoch == 3:
            print("\n[!] Unfreezing encoder for fine-tuning...")
            for param in model.encoder.parameters():
                param.requires_grad = True
                
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 20)
        
        # -- TRAIN --
        model.train()
        train_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc="Training")
        for batch_idx, (volumes, labels) in enumerate(train_pbar):
            # Apply D4 Spatial Augmentation on the fly
            volumes, labels = apply_d4_augmentation(volumes, labels)
            
            volumes = volumes.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Automatic Mixed Precision
            with torch.amp.autocast(device.type):
                outputs = model(volumes)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * volumes.size(0)
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
                
        epoch_train_loss = train_loss / len(train_dataset)
        
        # -- VALIDATION --
        model.eval()
        val_loss = 0.0
        sample_logged = False
        
        val_pbar = tqdm(val_loader, desc="Validation")
        with torch.no_grad():
            for volumes, labels in val_pbar:
                volumes = volumes.to(device)
                labels = labels.to(device)
                
                with torch.amp.autocast(device.type):
                    outputs = model(volumes)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item() * volumes.size(0)
                val_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
                
                # Visual logging
                if use_wandb and not sample_logged:
                    pred_prob = torch.sigmoid(outputs[0, 0]).cpu().numpy()
                    gt_label = labels[0, 0].cpu().numpy()
                    # Log the middle slice of the current pruned range
                    mid_slice_idx = input_channels // 2
                    wandb.log({
                        "sample_prediction": wandb.Image(pred_prob, caption=f"Epoch {epoch+1}"),
                        "sample_ground_truth": wandb.Image(gt_label),
                        "sample_input_slice": wandb.Image(volumes[0, mid_slice_idx].cpu().numpy())
                    }, commit=False)
                    sample_logged = True
                
        epoch_val_loss = val_loss / len(val_dataset)
        scheduler.step(epoch_val_loss)
        
        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": epoch_train_loss,
                "val_loss": epoch_val_loss,
                "lr_backbone": optimizer.param_groups[0]['lr'],
                "lr_custom": optimizer.param_groups[1]['lr']
            })
        
        print(f"Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")
        
        # -- EARLY STOPPING & CHECKPOINTING --
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            
            save_path = save_dir / "best_ink_model.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
            }, save_path)
            print(f"*** Best model saved to {save_path} ***")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epoch(s).")
            if patience_counter >= patience_limit:
                print(f"\n[!] Early Stopping at epoch {epoch+1}.")
                print(f"Training Interrupted. Best Val Loss: {best_val_loss:.4f}")
                break 
        
    # Save final fallback model
    final_path = save_dir / "final_ink_model.pth"
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining complete. Final weights saved to {final_path}")
    
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    train_model()