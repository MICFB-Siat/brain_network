import torch
import torch.nn as nn

class ResidualBlock(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=3, padding=1), nn.GroupNorm(8, channels), nn.GELU(), nn.Conv2d(channels, channels, kernel_size=3, padding=1), nn.GroupNorm(8, channels))
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))

class FCDiffusionUNet(nn.Module):

    def __init__(self, in_channels: int=4, spatial: int=32):
        super().__init__()
        self.in_channels = in_channels
        self.spatial = spatial
        (c1, c2, c3) = (96, 192, 384)
        self.conv_in = nn.Sequential(nn.Conv2d(in_channels, c1, kernel_size=3, padding=1), nn.GroupNorm(8, c1), nn.GELU())
        self.down1 = nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1)
        self.down2 = nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1)
        self.bottleneck = nn.Sequential(ResidualBlock(c3), ResidualBlock(c3), ResidualBlock(c3), ResidualBlock(c3))
        self.up1_conv = nn.Conv2d(c3 + c2, c2, kernel_size=3, padding=1)
        self.up2_conv = nn.Conv2d(c2 + c1, c1, kernel_size=3, padding=1)
        self.out_conv = nn.Conv2d(c1, in_channels, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x1 = self.conv_in(z)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        h = self.bottleneck(x3)
        h = torch.nn.functional.interpolate(h, scale_factor=2, mode='bilinear', align_corners=False)
        h = torch.cat([h, x2], dim=1)
        h = torch.nn.functional.gelu(self.up1_conv(h))
        h = torch.nn.functional.interpolate(h, scale_factor=2, mode='bilinear', align_corners=False)
        h = torch.cat([h, x1], dim=1)
        h = torch.nn.functional.gelu(self.up2_conv(h))
        out = self.out_conv(h)
        return out + z

    @classmethod
    def from_pretrained(cls, path, in_channels: int=4, **kwargs):
        m = cls(in_channels=in_channels, **kwargs)
        if path:
            state = torch.load(path, map_location='cpu', weights_only=False)
            m.load_state_dict(state, strict=False)
        return m

def build_fc_diffusion_unet(in_channels: int=4, spatial: int=32):
    return FCDiffusionUNet(in_channels=in_channels, spatial=spatial)
