import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import os
import time
import numpy as np
from tqdm import tqdm
import wandb

# Import our custom dataset and model
from ink_dataset import InkDetectionDataset
from ink_detection_net import InkDetectionNet

class BCEDiceLoss(nn.Module):
    """
    Combines Binary Cross Entropy (BCE) and Dice Loss.
    BCE helps with pixel-wise classification, while Dice Loss is excellent 
    for highly imbalanced datasets (where ink is rare compared to background).
    """
    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        # We use BCEWithLogitsLoss because our network outputs raw logits (no sigmoid at the end)
        # This is numerically more stable.
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        # 1. Calculate BCE Loss
        bce_loss = self.bce(inputs, targets)
        
        # 2. Calculate Dice Loss
        inputs_sigmoid = torch.sigmoid(inputs)
        smooth = 1e-5 # Prevents division by zero
        
        # Flatten predictions and targets to calculate intersection and union per batch item
        intersection = (inputs_sigmoid * targets).sum(dim=(2, 3))
        union = inputs_sigmoid.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        
        dice_loss = 1.0 - (2.0 * intersection + smooth) / (union + smooth)
        dice_loss = dice_loss.mean() # Average over batch

        # Combine
        return self.bce_weight * bce_loss + (1.0 - self.bce_weight) * dice_loss

def get_optimizer(model):
    """
    Sets up Differential Learning Rates.
    The pre-trained backbone gets a smaller LR so we don't destroy its learned features.
    The randomly initialized stem and decoder get a larger LR.
    """
    backbone_params = []
    custom_params = []
    
    for name, param in model.named_parameters():
        if 'backbone' in name and 'conv1' not in name:
            backbone_params.append(param)
        else:
            # This includes the modified stem (conv1) and the entire decoder
            custom_params.append(param)
            
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': 1e-5},
        {'params': custom_params, 'lr': 1e-3}
    ], weight_decay=1e-4)
    
    return optimizer

def train_model():
    # --- Configuration ---
    data_dir = "/Volumes/ZX20/eliva-26-ink-detection/train"
    epochs = 8 # Increased slightly for more thorough training
    batch_size = 8
    patch_size = 512
    
    # 1. Initialize W&B
    wandb.init(
        project="challenge-3-ink-detection",
        config={
            "backbone_lr": 1e-5,
            "decoder_lr": 1e-4,
            "epochs": epochs,
            "batch_size": batch_size,
            "patch_size": patch_size,
            "architecture": "ResNet50-UNet"
        }
    )

    # Check for Apple Silicon (MPS), CUDA, or fallback to CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # --- Data Splitting ---
    all_fragments = [d.name for d in Path(data_dir).iterdir() if d.is_dir() and not d.name.startswith('.')]
    if len(all_fragments) < 2:
        raise ValueError("Need at least 2 fragments in the train folder to create a validation split.")
        
    val_fragments = [all_fragments[0]]
    train_fragments = all_fragments[1:]
    
    print(f"Training on: {train_fragments}")
    print(f"Validating on: {val_fragments}")

    # --- Datasets and DataLoaders ---
    train_dataset = InkDetectionDataset(
        fragment_ids=train_fragments, data_dir=data_dir, patch_size=patch_size, 
        samples_per_epoch=2000, ink_prob=0.5, slice_range=(0, 64) # Corrected: 0 to 64 yields 64 slices
    )
    val_dataset = InkDetectionDataset(
        fragment_ids=val_fragments, data_dir=data_dir, patch_size=patch_size, 
        samples_per_epoch=500, ink_prob=0.2, slice_range=(0, 64)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # --- Model, Loss, Optimizer ---
    model = InkDetectionNet(input_channels=64).to(device)
    criterion = BCEDiceLoss(bce_weight=0.5)
    optimizer = get_optimizer(model)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

    # --- Training Loop ---
    best_val_loss = float('inf')
    save_dir = Path("checkpoints")
    save_dir.mkdir(exist_ok=True)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 20)
        
        # -- TRAIN --
        model.train()
        train_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc="Training")
        for batch_idx, (volumes, labels) in enumerate(train_pbar):
            volumes = volumes.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(volumes)
            loss = criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * volumes.size(0)
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
                
        epoch_train_loss = train_loss / len(train_dataset)
        
        # -- VALIDATION --
        model.eval()
        val_loss = 0.0
        sample_logged = False # For logging one image per epoch to W&B
        
        val_pbar = tqdm(val_loader, desc="Validation")
        with torch.no_grad():
            for volumes, labels in val_pbar:
                volumes = volumes.to(device)
                labels = labels.to(device)
                
                outputs = model(volumes)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * volumes.size(0)
                val_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
                
                # Visual log to WandB: Only for the first batch of validation
                if not sample_logged:
                    pred_prob = torch.sigmoid(outputs[0, 0]).cpu().numpy()
                    gt_label = labels[0, 0].cpu().numpy()
                    # Log prediction, ground truth, and middle slice
                    wandb.log({
                        "sample_prediction": wandb.Image(pred_prob, caption=f"Epoch {epoch+1}"),
                        "sample_ground_truth": wandb.Image(gt_label),
                        "sample_input_slice": wandb.Image(volumes[0, 12].cpu().numpy())
                    }, commit=False)
                    sample_logged = True
                
        epoch_val_loss = val_loss / len(val_dataset)
        scheduler.step(epoch_val_loss)
        
        # 2. Log Metrics to W&B
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": epoch_train_loss,
            "val_loss": epoch_val_loss,
            "lr_backbone": optimizer.param_groups[0]['lr'],
            "lr_custom": optimizer.param_groups[1]['lr']
        })
        
        print(f"Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")
        
        # -- SAVE CHECKPOINT --
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            save_path = save_dir / "best_ink_model.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
            }, save_path)
            print(f"*** Best model saved to {save_path} ***")

    # Save final model
    final_path = save_dir / "final_ink_model.pth"
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining complete. Final weights saved to {final_path}")
    
    # 3. Finish W&B
    wandb.finish()

if __name__ == "__main__":
    train_model()
