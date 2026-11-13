from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class PersonalizedFCConfig:
    d_k: int = 64
    time_bins: int = 128
    z_channels: int = 4
    z_h: int = 32
    z_w: int = 32

class PersonalizedFCModel(nn.Module):

    def __init__(self, cfg: PersonalizedFCConfig):
        super().__init__()
        self.cfg = cfg
        self.roi_q = nn.Linear(cfg.time_bins, cfg.d_k)
        self.roi_k = nn.Linear(cfg.time_bins, cfg.d_k)
        self.roi_v = nn.Linear(cfg.time_bins, cfg.d_k)
        self.z_k = nn.Linear(cfg.z_channels, cfg.d_k)
        self.z_v = nn.Linear(cfg.z_channels, cfg.d_k)
        self.q_proj = nn.Linear(cfg.d_k, cfg.d_k)

    def _roi_resample(self, H: torch.Tensor) -> torch.Tensor:
        if H.dim() != 2:
            raise ValueError(f'Expected H shape (K,T), got {tuple(H.shape)}')
        (K, T) = H.shape
        if T == self.cfg.time_bins:
            return H
        x = H.unsqueeze(0)
        x = F.interpolate(x, size=self.cfg.time_bins, mode='linear', align_corners=False)
        return x.squeeze(0)

    def encode_rois(self, H: torch.Tensor) -> torch.Tensor:
        H = self._roi_resample(H)
        Q = self.roi_q(H)
        Kp = self.roi_k(H)
        V = self.roi_v(H)
        attn = torch.softmax(Q @ Kp.T * self.cfg.d_k ** (-0.5), dim=-1)
        return attn @ V

    def encode_z_tokens(self, zpattern: torch.Tensor) -> torch.Tensor:
        if zpattern is None:
            raise ValueError('zpattern is required for cross-attention')
        if isinstance(zpattern, np.ndarray):
            zpattern = torch.from_numpy(zpattern).float()
        if zpattern.dim() == 3:
            zpattern = zpattern.unsqueeze(0)
        if zpattern.dim() != 4:
            raise ValueError(f'Expected zpattern dim=4, got {zpattern.dim()}')
        if zpattern.shape[1] != self.cfg.z_channels:
            raise ValueError(f'Expected zpattern channels={self.cfg.z_channels}, got {zpattern.shape[1]}')
        z = zpattern[0]
        tokens = z.permute(1, 2, 0).reshape(-1, self.cfg.z_channels)
        Kz = self.z_k(tokens)
        Vz = self.z_v(tokens)
        return (Kz, Vz)

    @staticmethod
    def corrcoef_rows(X: torch.Tensor, eps: float=1e-06) -> torch.Tensor:
        X = X - X.mean(dim=1, keepdim=True)
        X = X / (X.std(dim=1, keepdim=True) + eps)
        return X @ X.T / max(1, X.shape[1] - 1)

    def forward(self, H: torch.Tensor, zpattern: torch.Tensor) -> torch.Tensor:
        Hintra = self.encode_rois(H)
        Q = self.q_proj(Hintra)
        (Kz, Vz) = self.encode_z_tokens(zpattern)
        attn = torch.softmax(Q @ Kz.T * self.cfg.d_k ** (-0.5), dim=-1)
        Hcross = attn @ Vz
        return self.corrcoef_rows(Hcross)

def load_personalized_fc_model(ckpt_path: str, device: str='cpu', cfg: Optional[PersonalizedFCConfig]=None) -> PersonalizedFCModel:
    cfg = cfg or PersonalizedFCConfig()
    m = PersonalizedFCModel(cfg).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    m.load_state_dict(state, strict=True)
    m.eval()
    return m

def personalized_fc_from_roi_timeseries(H: Union[np.ndarray, torch.Tensor], zpattern: Union[np.ndarray, torch.Tensor, None], d_k: int=64, ckpt_path: Optional[str]=None, device: Optional[str]=None) -> np.ndarray:
    if isinstance(H, np.ndarray):
        H_t = torch.from_numpy(H).float()
    else:
        H_t = H.float()
    dev = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    H_t = H_t.to(dev)
    if zpattern is None:
        return np.corrcoef(H_t.detach().cpu().numpy())
    if ckpt_path and os.path.exists(ckpt_path):
        cfg = PersonalizedFCConfig(d_k=d_k)
        model = load_personalized_fc_model(ckpt_path, device=dev, cfg=cfg)
        with torch.no_grad():
            z_t = zpattern
            if isinstance(z_t, np.ndarray):
                z_t = torch.from_numpy(z_t).float()
            z_t = z_t.to(dev)
            fc = model(H_t, z_t).detach().cpu().numpy()
        return fc
    return np.corrcoef(H_t.detach().cpu().numpy())
