import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, dropout_prob=0.2):
        super().__init__()
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
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = F.pad(x, [0, skip.shape[3] - x.shape[3], 0, skip.shape[2] - x.shape[2]])
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class InkDetectionNetV3(nn.Module):
    def __init__(self, encoder_name='maxvit_large_tf_224.in1k', pretrained=True, input_channels=30, dropout_prob=0.2):
        super().__init__()
        
        # MaxViT Encoder
        self.encoder = timm.create_model(
            encoder_name, 
            pretrained=pretrained, 
            in_chans=input_channels, 
            features_only=True,
            out_indices=(0, 1, 2, 3)
        )
        
        encoder_channels = self.encoder.feature_info.channels()
        
        # Decoder
        self.dec1 = DecoderBlock(encoder_channels[3], encoder_channels[2], 512, dropout_prob)
        self.dec2 = DecoderBlock(512, encoder_channels[1], 256, dropout_prob)
        self.dec3 = DecoderBlock(256, encoder_channels[0], 128, dropout_prob)
        
        # Final Upsampling (1/4 -> 1/1)
        self.final_upsample = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1)
        )

    def forward(self, x):
        features = self.encoder(x)
        x = self.dec1(features[3], features[2])
        x = self.dec2(x, features[1])
        x = self.dec3(x, features[0])
        return self.final_upsample(x)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)

if __name__ == "__main__":
    model = InkDetectionNetV3(input_channels=30, pretrained=False)
    dummy_input = torch.randn(1, 30, 224, 224)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
