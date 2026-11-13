import torch
import torch.nn as nn

class FCEncoderDecoder(nn.Module):

    def __init__(self, input_size: int=1024, seq_len: int=1024, proj_dim: int=96, latent_channels: int=4, latent_spatial: int=32, nhead: int=8, num_layers: int=2, dim_feedforward: int=512):
        super().__init__()
        self.input_size = input_size
        self.seq_len = seq_len
        self._latent_channels = latent_channels
        self._latent_spatial = latent_spatial
        self.proj = nn.Linear(input_size, proj_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=proj_dim, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        latent_dim = latent_channels * latent_spatial * latent_spatial
        self.mlp = nn.Sequential(nn.Linear(proj_dim, 1024), nn.ReLU(inplace=True), nn.Linear(1024, latent_dim))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4 and x.shape[1] == 1:
            x = x.squeeze(1)
        assert x.shape[1] == self.seq_len and x.shape[2] == self.input_size, (x.shape, self.seq_len, self.input_size)
        h = self.proj(x)
        h = self.transformer(h)
        h_mean = h.mean(dim=1)
        z = self.mlp(h_mean)
        B = z.shape[0]
        return z.view(B, self._latent_channels, self._latent_spatial, self._latent_spatial)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)

def build_default_autoencoder():
    return FCEncoderDecoder(input_size=1024, seq_len=1024, proj_dim=96, latent_channels=4, latent_spatial=32)

class FCDecoder(nn.Module):

    def __init__(self, latent_channels: int=4, latent_spatial: int=32, seq_len: int=1024, hidden_dim: int=640, proj_dim: int=112, input_size: int=1024):
        super().__init__()
        self._latent_channels = latent_channels
        self._latent_spatial = latent_spatial
        self.seq_len = seq_len
        self.proj_dim = proj_dim
        self.input_size = input_size
        flat_in = latent_channels * latent_spatial * latent_spatial
        self.to_seq = nn.Sequential(nn.Flatten(), nn.Linear(flat_in, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, seq_len * proj_dim), nn.ReLU(inplace=True))
        self.per_vertex = nn.Linear(proj_dim, input_size)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        h = self.to_seq(z)
        h = h.view(B, self.seq_len, self.proj_dim)
        x_rec = self.per_vertex(h)
        return x_rec.unsqueeze(1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decode(z)

def build_default_decoder(latent_channels: int=4, latent_spatial: int=32):
    return FCDecoder(latent_channels=latent_channels, latent_spatial=latent_spatial)
