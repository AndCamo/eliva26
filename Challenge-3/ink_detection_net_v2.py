import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, dropout_prob=0.3):
        super().__init__()
        # Concatenation happens on (in_channels + skip_channels)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_prob)
        )

    def forward(self, x, skip=None):
        # 1. Bilinear upsampling (doubles resolution)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        
        # 2. Concatenation with the skip connection from encoder
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            
        # 3. Apply convolutions
        return self.conv(x)


class InkDetectionNetV2(nn.Module):
    def __init__(self, encoder_name='tf_efficientnetv2_l', pretrained=True, input_channels=30, dropout_prob=0.3):
        super().__init__()
        
        # --- 1. Encoder (Backbone) ---
        # Dynamically load the EfficientNetV2-L encoder from timm.
        # This model is extremely powerful for texture analysis in papyrus scans.
        self.encoder = timm.create_model(
            encoder_name, 
            pretrained=pretrained, 
            in_chans=input_channels, 
            features_only=True,
            out_indices=(0, 1, 2, 3, 4)
        )
        
        # Automatically retrieve the channels produced by the V2-L encoder
        encoder_channels = self.encoder.feature_info.channels()
        
        # --- 2. Enhanced Dynamic Decoder ---
        # We increase the decoder width (starting at 512) to match the Large encoder's capacity.
        
        # Block 1: Input from bottleneck (idx 4), Skip from stage 3 (idx 3)
        self.dec1 = DecoderBlock(encoder_channels[4], encoder_channels[3], 512, dropout_prob)
        # Block 2: Input from dec1, Skip from stage 2 (idx 2)
        self.dec2 = DecoderBlock(512, encoder_channels[2], 256, dropout_prob)
        # Block 3: Input from dec2, Skip from stage 1 (idx 1)
        self.dec3 = DecoderBlock(256, encoder_channels[1], 128, dropout_prob)
        # Block 4: Input from dec3, Skip from stem (idx 0)
        self.dec4 = DecoderBlock(128, encoder_channels[0], 64, dropout_prob)
        
        # --- 3. Final Header ---
        # Final upsampling to reach original resolution
        self.dec5 = DecoderBlock(64, 0, 32, dropout_prob)
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        # 1. Encoder pass: returns a list of 5 feature maps
        features = self.encoder(x)
        
        # 2. Decoder pass with skip connections
        # features[4] is the bottleneck, features[0-3] are skips
        x = self.dec1(features[4], features[3])
        x = self.dec2(x, features[2])
        x = self.dec3(x, features[1])
        x = self.dec4(x, features[0])
        
        # 3. Final upsample and binary prediction
        x = self.dec5(x)
        x = self.final_conv(x)
        
        return x

    def predict(self, x):
        """
        Utility for inference. Returns probability map.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)
