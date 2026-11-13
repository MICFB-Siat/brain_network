import torch
import torch.nn as nn

class D1Decoder(nn.Module):

    def __init__(self, latent_channels: int=4, latent_h: int=32, latent_w: int=32, vertex_count: int=29696, time_steps: int=1, hidden_dim: int=128, down_h: int=8, down_w: int=8):
        super().__init__()
        self.latent_channels = latent_channels
        self.latent_h = latent_h
        self.latent_w = latent_w
        self.vertex_count = vertex_count
        self.time_steps = time_steps
        self.down_h = down_h
        self.down_w = down_w
        self.pool = nn.AdaptiveAvgPool2d((down_h, down_w))
        flat = latent_channels * down_h * down_w
        self.fc1 = nn.Linear(flat, hidden_dim)
        self.act = nn.GELU()
        self.fc_out = nn.Linear(hidden_dim, vertex_count * time_steps)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        z_ds = self.pool(z)
        flat = z_ds.view(B, -1)
        h = self.act(self.fc1(flat))
        out = self.fc_out(h)
        return out.view(B, self.vertex_count, self.time_steps)
