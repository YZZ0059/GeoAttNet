import torch
import torch.nn as nn
import timm
import torch.nn.functional as F

class ChannelAttention(nn.Module):

    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return torch.sigmoid(out)

class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return torch.sigmoid(out)

class CBAM(nn.Module):

    def __init__(self, in_channels, reduction_ratio=16, spatial_kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(spatial_kernel_size)

    def forward(self, x):
      
        x = x * self.channel_attention(x)
 
        x = x * self.spatial_attention(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_rate=0.2, use_attention=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate)
        )
        self.residual = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.attention = CBAM(out_ch) if use_attention else nn.Identity()

    def forward(self, x):
        out = self.conv(x) + self.residual(x)
        out = self.attention(out)
        return out

class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_rate=0.2, use_attention=True):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch, dropout_rate, use_attention)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        return self.pool(x), x

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, dropout_rate=0.2, use_attention=True):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch, dropout_rate, use_attention)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class DeepUNet(nn.Module):
    def __init__(self, in_channels=15, num_classes=1, dropout_rate=0.2, use_attention=True):
        super().__init__()

        base_channels = 64

        self.inc = ConvBlock(in_channels, base_channels, dropout_rate, use_attention)

        self.enc1 = EncoderBlock(base_channels, base_channels*2, dropout_rate, use_attention)
        self.enc2 = EncoderBlock(base_channels*2, base_channels*4, dropout_rate, use_attention)
        self.enc3 = EncoderBlock(base_channels*4, base_channels*8, dropout_rate, use_attention)

        self.bottleneck = ConvBlock(base_channels*8, base_channels*16, dropout_rate, use_attention)

        self.dec3 = DecoderBlock(base_channels*16, base_channels*8, base_channels*8, dropout_rate, use_attention)
        self.dec2 = DecoderBlock(base_channels*8, base_channels*4, base_channels*4, dropout_rate, use_attention)
        self.dec1 = DecoderBlock(base_channels*4, base_channels*2, base_channels*2, dropout_rate, use_attention)

   
        self.outc = nn.Sequential(
            ConvBlock(base_channels*2, base_channels, dropout_rate, use_attention),
            nn.Conv2d(base_channels, num_classes, 1),
            nn.BatchNorm2d(num_classes)
        )

     
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
  
        self._check_input_size(x)
        

        x1 = self.inc(x)
        x2, skip1 = self.enc1(x1)
        x3, skip2 = self.enc2(x2)
        x4, skip3 = self.enc3(x3)
        
     
        x5 = self.bottleneck(x4)
        

        x6 = self.dec3(x5, skip3)
        x7 = self.dec2(x6, skip2)
        x8 = self.dec1(x7, skip1)
        
 
        out = self.outc(x8)
        
        return out
    
    def _check_input_size(self, x):
        h, w = x.shape[2:]
        if h != 32 or w != 32:
            raise ValueError(f"{h}x{w}")

SmallSwinUNet = DeepUNet

def test_model():
  
    model = DeepUNet(in_channels=15, num_classes=1, dropout_rate=0.2)

    batch_sizes = [1, 4, 16, 32]
    for batch_size in batch_sizes:
        x = torch.randn(batch_size, 15, 32, 32)  
        try:
            with torch.no_grad():
                out = model(x)
                print(f" {batch_size:2d}:  {tuple(x.shape)} ->  {tuple(out.shape)}")
        except Exception as e:
            print(f" {batch_size:2d}: fail - {str(e)}")
    

    x = torch.randn(1, 15, 32, 32)  
    

    with torch.no_grad():
        x1 = model.inc(x)
        x2, skip1 = model.enc1(x1)
        x3, skip2 = model.enc2(x2)
        x4, skip3 = model.enc3(x3)
        bottle = model.bottleneck(x4)
 

    if torch.cuda.is_available():
        try:
            model.cuda()
            x = torch.randn(4, 15, 32, 32).cuda()  
            with torch.no_grad():
                out = model(x)

        except Exception as e:
            print(f"GPU fail: {str(e)}")
    else:
        print("no GPU")
    

    model = model.cpu()
    x = torch.randn(1, 15, 32, 32)  
    
    import time
    times = []
    with torch.no_grad():
        for _ in range(100):
            start = time.time()
            _ = model(x)
            times.append(time.time() - start)
    
    avg_time = sum(times) / len(times) * 1000  

    import psutil
    process = psutil.Process()
    
    def get_memory_usage():
        return process.memory_info().rss / 1024 / 1024  
    
    initial_memory = get_memory_usage()
    with torch.no_grad():
        x = torch.randn(32, 15, 32, 32)  
        _ = model(x)
    final_memory = get_memory_usage()
    

if __name__ == '__main__':
    test_model()