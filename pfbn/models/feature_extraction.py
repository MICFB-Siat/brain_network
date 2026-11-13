import os
import numpy as np
import torch
from PIL import Image
try:
    from encoder_model import build_default_autoencoder
    from diffusion_model import build_fc_diffusion_unet
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from encoder_model import build_default_autoencoder
    from diffusion_model import build_fc_diffusion_unet

def load_encoder(ckpt_path, device='cuda'):
    m = build_default_autoencoder().to(device)
    if ckpt_path and os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        m.load_state_dict(state, strict=False)
    m.eval()
    return m

def load_diffusion(ckpt_path, device='cuda'):
    m = build_fc_diffusion_unet(in_channels=4).to(device)
    if ckpt_path and os.path.exists(ckpt_path):
        m.load_state_dict(torch.load(ckpt_path, map_location=device))
    m.eval()
    return m

def load_avg_fc_as_tensor(avgfc_path, device='cuda'):
    if not os.path.exists(avgfc_path):
        raise FileNotFoundError(avgfc_path)
    img = np.array(Image.open(avgfc_path).convert('L'), dtype=np.float32) / 255.0
    if img.shape != (1024, 1024):
        import torch.nn.functional as F
        img_t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
        img_t = F.interpolate(img_t, size=(1024, 1024), mode='bilinear', align_corners=False)
        img = img_t.numpy()[0, 0]
    return torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).to(device)

def extract_zpattern_and_feature(avgfc_path, encoder_ckpt, diffusion_ckpt, device='cuda', save_zpattern_path=None, save_feature_2d_path=None, lora_ckpt=None, lora_r=8):
    x = load_avg_fc_as_tensor(avgfc_path, device)
    if lora_ckpt and os.path.isdir(lora_ckpt):
        from lora_utils import load_encoder_with_lora, load_diffusion_with_lora
        enc = load_encoder_with_lora(lora_ckpt, device=device, r=lora_r)
        diff = load_diffusion_with_lora(lora_ckpt, device=device, r=lora_r, in_channels=4)
    else:
        enc = load_encoder(encoder_ckpt, device)
        diff = load_diffusion(diffusion_ckpt, device)
    with torch.no_grad():
        z = enc.encode(x)
    zpattern = z.detach()
    del enc
    t = torch.zeros(1, dtype=torch.long, device=device)
    with torch.no_grad():
        out = diff(zpattern, t)
    out_2d = torch.nn.functional.interpolate(out, size=(1024, 1024), mode='bilinear', align_corners=False)
    feature_2d = out_2d[0].sum(dim=0).cpu().numpy()
    del diff
    if save_zpattern_path:
        d = os.path.dirname(save_zpattern_path)
        if d:
            os.makedirs(d, exist_ok=True)
        torch.save(zpattern.cpu(), save_zpattern_path)
    if save_feature_2d_path:
        d = os.path.dirname(save_feature_2d_path)
        if d:
            os.makedirs(d, exist_ok=True)
        torch.save(feature_2d, save_feature_2d_path)
    return (zpattern, feature_2d)
