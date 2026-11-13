import os
import torch
import torch.nn as nn
from typing import Optional

class LoRALinear(nn.Module):

    def __init__(self, linear: nn.Linear, r: int=8, scale: float=1.0):
        super().__init__()
        self.linear = linear
        (in_f, out_f) = (linear.in_features, linear.out_features)
        self.lora_A = nn.Parameter(torch.zeros(r, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r))
        self.scale = scale
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        base = self.linear(x)
        lora = x @ self.lora_A.T @ self.lora_B.T * self.scale
        return base + lora

class LoRAConv2d(nn.Module):

    def __init__(self, conv: nn.Conv2d, r: int=8, scale: float=1.0):
        super().__init__()
        self.conv = conv
        (in_c, out_c) = (conv.in_channels, conv.out_channels)
        self.conv1 = nn.Conv2d(in_c, r, kernel_size=1, stride=conv.stride, padding=0, dilation=conv.dilation, groups=1, bias=False)
        self.conv2 = nn.Conv2d(r, out_c, 1)
        self.scale = scale
        nn.init.kaiming_uniform_(self.conv1.weight, a=5 ** 0.5)
        nn.init.zeros_(self.conv2.weight)
        if self.conv2.bias is not None:
            nn.init.zeros_(self.conv2.bias)

    def forward(self, x):
        base = self.conv(x)
        lora = self.conv2(self.conv1(x)) * self.scale
        return base + lora

def _inject_encoder_lora(encoder, r: int=8, scale: float=1.0):
    if hasattr(encoder, 'proj') and isinstance(encoder.proj, nn.Linear):
        encoder.proj = LoRALinear(encoder.proj, r=r, scale=scale)
    if hasattr(encoder, 'mlp'):
        for (name, m) in list(encoder.mlp.named_children()):
            if isinstance(m, nn.Linear):
                setattr(encoder.mlp, name, LoRALinear(m, r=r, scale=scale))
    return encoder

def _inject_decoder_lora(decoder, r: int=8, scale: float=1.0):
    if hasattr(decoder, 'to_seq'):
        for (name, m) in list(decoder.to_seq.named_children()):
            if isinstance(m, nn.Linear):
                setattr(decoder.to_seq, name, LoRALinear(m, r=r, scale=scale))
    if hasattr(decoder, 'per_vertex') and isinstance(decoder.per_vertex, nn.Linear):
        decoder.per_vertex = LoRALinear(decoder.per_vertex, r=r, scale=scale)
    return decoder

def _inject_diffusion_lora(diffusion, r: int=8, scale: float=1.0):
    for (name, m) in list(diffusion.named_modules()):
        pass

    def _replace_conv(module: nn.Module):
        for (child_name, child) in list(module.named_children()):
            if isinstance(child, nn.Conv2d):
                setattr(module, child_name, LoRAConv2d(child, r=r, scale=scale))
            else:
                _replace_conv(child)
    _replace_conv(diffusion)
    return diffusion

def build_encoder_with_lora(r: int=8, lora_scale: float=1.0):
    from encoder_model import build_default_autoencoder
    enc = build_default_autoencoder()
    return _inject_encoder_lora(enc, r=r, scale=lora_scale)

def build_decoder_with_lora(r: int=8, lora_scale: float=1.0):
    from encoder_model import build_default_decoder
    dec = build_default_decoder()
    return _inject_decoder_lora(dec, r=r, scale=lora_scale)

def build_diffusion_with_lora(r: int=8, lora_scale: float=1.0, in_channels: int=4, spatial: int=32):
    from diffusion_model import build_fc_diffusion_unet
    diff = build_fc_diffusion_unet(in_channels=in_channels, spatial=spatial)
    return _inject_diffusion_lora(diff, r=r, scale=lora_scale)

def load_encoder_with_lora(lora_dir: str, device: str='cuda', r: int=8):
    enc = build_encoder_with_lora(r=r).to(device)
    path = os.path.join(lora_dir, 'encoder.pt')
    if os.path.exists(path):
        enc.load_state_dict(torch.load(path, map_location=device, weights_only=False), strict=False)
    enc.eval()
    return enc

def load_decoder_with_lora(lora_dir: str, device: str='cuda', r: int=8):
    dec = build_decoder_with_lora(r=r).to(device)
    path = os.path.join(lora_dir, 'decoder.pt')
    if os.path.exists(path):
        dec.load_state_dict(torch.load(path, map_location=device, weights_only=False), strict=False)
    dec.eval()
    return dec

def load_diffusion_with_lora(lora_dir: str, device: str='cuda', r: int=8, in_channels: int=4):
    diff = build_diffusion_with_lora(r=r, in_channels=in_channels).to(device)
    path = os.path.join(lora_dir, 'diffusion.pt')
    if os.path.exists(path):
        diff.load_state_dict(torch.load(path, map_location=device, weights_only=False), strict=False)
    diff.eval()
    return diff
