import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        
        self.query = nn.Conv2d(in_channels, in_channels//8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels//8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # Query, Key, Value projections
        query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, height * width)
        value = self.value(x).view(batch_size, -1, height * width)
        
        # Attention scores
        attention = self.softmax(torch.bmm(query, key))
        
        # Attention output
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        
        # Add residual connection
        return self.gamma * out + x

class BasicUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        
        # Encoder
        self.encoder1 = DoubleConv(in_channels, 64)
        self.encoder2 = DoubleConv(64, 128)
        self.encoder3 = DoubleConv(128, 256)
        self.encoder4 = DoubleConv(256, 512)
        
        # Attention blocks
        self.attention1 = AttentionBlock(256)
        self.attention2 = AttentionBlock(512)
        self.attention3 = AttentionBlock(1024)
        
        # Middle
        self.middle = DoubleConv(512, 1024)
        
        # Decoder
        self.decoder4 = DoubleConv(1024, 512)
        self.decoder3 = DoubleConv(512, 256)
        self.decoder2 = DoubleConv(256, 128)
        self.decoder1 = DoubleConv(128, 64)
        
        # Max pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Upsampling
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        
        # Final convolution
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        x = self.pool(enc1)
        
        enc2 = self.encoder2(x)
        x = self.pool(enc2)
        
        enc3 = self.encoder3(x)
        x = self.attention1(enc3)  # Add attention
        x = self.pool(x)
        
        enc4 = self.encoder4(x)
        x = self.attention2(enc4)  # Add attention
        x = self.pool(x)
        
        # Middle
        x = self.middle(x)
        x = self.attention3(x)  # Add attention
        
        # Decoder
        x = self.upconv4(x)
        x = torch.cat([enc4, x], dim=1)
        x = self.decoder4(x)
        
        x = self.upconv3(x)
        x = torch.cat([enc3, x], dim=1)
        x = self.decoder3(x)
        
        x = self.upconv2(x)
        x = torch.cat([enc2, x], dim=1)
        x = self.decoder2(x)
        
        x = self.upconv1(x)
        x = torch.cat([enc1, x], dim=1)
        x = self.decoder1(x)
        
        # Final convolution
        x = self.final_conv(x)
        return x

# Example usage
if __name__ == "__main__":
    # Create model instance
    model = BasicUNet(in_channels=3, out_channels=1)
    
    # Test with random input
    x = torch.randn(1, 3, 256, 256)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")