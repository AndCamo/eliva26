import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip=None):
        """Forward pass for the decoder block using bilinear upsampling and concatenation with skip connections."""
        
        # 1. Upsample the input 
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        
        # 2. Concatenate with the skip connection (from the encoder) if it exists
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            
        # 3. Apply convolutions
        return self.conv(x)


class InkDetectionNet(nn.Module):
    def __init__(self, input_channels=25):
        super(InkDetectionNet, self).__init__()
        
        # --- Encoder (ResNet50) ---
        # Load pre-trained ResNet50 weights
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        # Stem layer to handle multi-channel input (instead of the original 3 channels)
        self.backbone.conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        
        # --- Decoder Blocks ---
        # ResNet50 channels: layer4=2048, layer3=1024, layer2=512, layer1=256, stem=64
        
        # Block 1: From layer4 (2048) + skip from layer3 (1024) -> 512
        self.dec1 = DecoderBlock(in_channels=2048, skip_channels=1024, out_channels=512)
        
        # Block 2: From dec1 (512) + skip from layer2 (512) -> 256
        self.dec2 = DecoderBlock(in_channels=512, skip_channels=512, out_channels=256)
        
        # Block 3: From dec2 (256) + skip from layer1 (256) -> 128
        self.dec3 = DecoderBlock(in_channels=256, skip_channels=256, out_channels=128)
        
        # Block 4: From dec3 (128) + skip from stem (64) -> 64
        self.dec4 = DecoderBlock(in_channels=128, skip_channels=64, out_channels=64)
        
        # Final upsampling to reach the original image resolution (no skip connection)
        self.dec5 = DecoderBlock(in_channels=64, skip_channels=0, out_channels=32)
        
        # Final 1x1 convolution to output a single channel (ink probability map)
        self.final_conv = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1)
    
    def encoder(self, x):
        features = []
        
        # 1. Stem (Resolution: /2)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        features.append(x) # Skip 1 (highest resolution)
        x = self.backbone.maxpool(x)

        # 2. ResNet Layers
        x = self.backbone.layer1(x) # Resolution: /4
        features.append(x) # Skip 2
        
        x = self.backbone.layer2(x) # Resolution: /8
        features.append(x) # Skip 3
        
        x = self.backbone.layer3(x) # Resolution: /16
        features.append(x) # Skip 4
        
        x = self.backbone.layer4(x) # Resolution: /32 (Bottleneck)
        
        # Return the bottleneck 'x' and the list of features for the decoder
        return x, features
    
    def decoder(self, x, features):
        # Apply decoder blocks symmetrically, matching with correct skip connections from 'features' list
        x = self.dec1(x, features[3])
        x = self.dec2(x, features[2])
        x = self.dec3(x, features[1])
        x = self.dec4(x, features[0])
        
        # Final upsample to original input resolution
        x = self.dec5(x)
        
        # Output layer
        x = self.final_conv(x)
        return x

    def forward(self, x):
        # Forward pass through encoder
        x, features = self.encoder(x)
        
        # Forward pass through decoder
        x = self.decoder(x, features)
        
        return x

    def predict(self, x):
        """
        Utility for inference. 
        Sets model to eval mode, disables gradients, and applies sigmoid.
        Returns probability map [0, 1].
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.sigmoid(logits)
        return probabilities