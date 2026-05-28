import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import numpy as np
from tqdm import tqdm
import wandb

# Import dei moduli custom
from ink_dataset import InkDetectionDataset
from ink_detection_net import InkDetectionNet


def wandb_enabled():
    return os.getenv("DISABLE_WANDB", "0").lower() not in ("1", "true", "yes")


class BCEDiceLoss(nn.Module):
    """
    Combina Binary Cross Entropy (BCE) e Dice Loss.
    """
    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        
        inputs_sigmoid = torch.sigmoid(inputs)
        smooth = 1e-5 
        
        intersection = (inputs_sigmoid * targets).sum(dim=(2, 3))
        union = inputs_sigmoid.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        
        dice_loss = 1.0 - (2.0 * intersection + smooth) / (union + smooth)
        dice_loss = dice_loss.mean() 

        return self.bce_weight * bce_loss + (1.0 - self.bce_weight) * dice_loss


def get_optimizer(model):
    """
    Imposta i Learning Rate Differenziali.
    """
    backbone_params = []
    custom_params = []
    
    for name, param in model.named_parameters():
        if 'backbone' in name and 'conv1' not in name:
            backbone_params.append(param)
        else:
            custom_params.append(param)
            
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': 5e-6}, # LR ridotto per la backbone
        {'params': custom_params, 'lr': 1e-4}
    ], weight_decay=1e-4)
    
    return optimizer


def train_model():
    # --- Training Parameters ---
    # Nota: Assicurati che questo percorso sia corretto per il tuo Mac
    data_dir = "/Volumes/ZX20/eliva-26-ink-detection/train"
    epochs = 100 
    batch_size = 4  
    patch_size = 256
    slice_range = (15, 45)  
    input_channels = slice_range[1] - slice_range[0]  
    num_workers = 0   
    patience_limit = 10 # Alzata leggermente vista la fase di freezing
    patience_counter = 0
    
    use_wandb = wandb_enabled()
    
    # Inizializzazione Weights & Biases
    if use_wandb:
        wandb.init(
            project="challenge-3-ink-detection",
            config={
                "backbone_lr": 5e-6,
                "decoder_lr": 1e-4,
                "epochs": epochs,
                "batch_size": batch_size,
                "patch_size": patch_size,
                "architecture": "ResNet50-UNet"
            }
        )

    # Selezione Device (Preferenza per MPS su Mac)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # --- Data Splitting (Multi-Fragment) ---
    all_fragments = [d.name for d in Path(data_dir).iterdir() if d.is_dir() and not d.name.startswith('.')]
    if len(all_fragments) < 5:
        # Fallback se non ci sono abbastanza frammenti per la validazione specifica
        val_fragments = all_fragments[:1]
        train_fragments = all_fragments[1:]
    else:
        # I frammenti indicati nel file originale
        val_fragments = ['p_5qlsf5h5b1yy', 'p_hwx6h1ybz19s', 'p_cls0w7rbx6e4', 'p_yjx64pa5da7o']
        train_fragments = [frag for frag in all_fragments if frag not in val_fragments]
    
    print(f"Training on: {len(train_fragments)} fragments")
    print(f"Validating on: {val_fragments}")

    # --- Datasets e DataLoaders ---
    train_dataset = InkDetectionDataset(
        fragment_ids=train_fragments, data_dir=data_dir, patch_size=patch_size, 
        samples_per_epoch=2500, ink_prob=0.3, slice_range=slice_range, augment=True
    )
    val_dataset = InkDetectionDataset(
        fragment_ids=val_fragments, data_dir=data_dir, patch_size=patch_size, 
        samples_per_epoch=800, ink_prob=0.0, slice_range=slice_range, augment=False
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # --- Model, Loss, Optimizer ---
    model = InkDetectionNet(input_channels=input_channels).to(device)
    
    # --- INITIAL BACKBONE FREEZING ---
    print("Freezing backbone for the first 3 epochs...")
    for name, param in model.backbone.named_parameters():
        if "conv1" not in name:
            param.requires_grad = False
            
    criterion = BCEDiceLoss(bce_weight=0.5)
    optimizer = get_optimizer(model)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

    # --- Training Loop Setup ---
    best_val_loss = float('inf')
    save_dir = Path("checkpoints")
    save_dir.mkdir(exist_ok=True)
    
    for epoch in range(epochs):
        # --- GRADUAL UNFREEZING ---
        if epoch == 3:
            print("\n[!] Unfreezing backbone for fine-tuning...")
            for param in model.backbone.parameters():
                param.requires_grad = True

        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 20)
        
        # -- TRAIN --
        model.train()
        train_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc="Training")
        for volumes, labels in train_pbar:
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
        sample_logged = False
        
        val_pbar = tqdm(val_loader, desc="Validation")
        with torch.no_grad():
            for volumes, labels in val_pbar:
                volumes = volumes.to(device)
                labels = labels.to(device)
                
                outputs = model(volumes)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * volumes.size(0)
                val_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
                
                # Visual log su WandB (solo il primo batch)
                if use_wandb and not sample_logged:
                    pred_prob = torch.sigmoid(outputs[0, 0]).cpu().numpy()
                    gt_label = labels[0, 0].cpu().numpy()
                    wandb.log({
                        "sample_prediction": wandb.Image(pred_prob, caption=f"Epoch {epoch+1}"),
                        "sample_ground_truth": wandb.Image(gt_label),
                        "sample_input_slice": wandb.Image(volumes[0, 12].cpu().numpy())
                    }, commit=False)
                    sample_logged = True
                
        epoch_val_loss = val_loss / len(val_dataset)
        scheduler.step(epoch_val_loss)
        
        # Log Metriche
        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": epoch_train_loss,
                "val_loss": epoch_val_loss,
                "lr_backbone": optimizer.param_groups[0]['lr'],
                "lr_custom": optimizer.param_groups[1]['lr']
            })
        
        print(f"Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")
        
        # -- SAVE CHECKPOINT & EARLY STOPPING --
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            
            save_path = save_dir / "best_ink_model_mac.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
            }, save_path)
            print(f"*** Best model saved to {save_path} ***")
        else:
            patience_counter += 1
            print(f"Nessun miglioramento per {patience_counter} epoca/he.")
            if patience_counter >= patience_limit:
                print(f"\n[!] Early Stopping all'epoca {epoch+1}.")
                break 
        
    # Salvataggio Modello Finale
    final_path = save_dir / "final_ink_model_mac.pth"
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining completo. Pesi salvati in {final_path}")
    
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    train_model()
