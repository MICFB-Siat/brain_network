import os
import sys
import argparse
import torch
import torch.nn as nn
from PIL import Image
import numpy as np

def _ensure_path():
    root = os.path.dirname(os.path.abspath(__file__))
    if root not in sys.path:
        sys.path.insert(0, root)

def load_avg_fc_as_tensor(avgfc_path, device='cuda'):
    if not os.path.exists(avgfc_path):
        raise FileNotFoundError(avgfc_path)
    img = np.array(Image.open(avgfc_path).convert('L'), dtype=np.float32) / 255.0
    if img.shape != (1024, 1024):
        img_t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
        img_t = torch.nn.functional.interpolate(img_t, size=(1024, 1024), mode='bilinear', align_corners=False)
        img = img_t.numpy()[0, 0]
    return torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).to(device)

def main():
    _ensure_path()
    p = argparse.ArgumentParser(description='LoRA finetune encoder + decoder + diffusion on test_dataset avg FC')
    p.add_argument('--work_dir', type=str, default=None)
    p.add_argument('--encoder_ckpt', type=str, default='models/encoder_decoder/encoder_decoder_latest.pth')
    p.add_argument('--decoder_ckpt', type=str, default=None, help='Default: same as encoder_ckpt')
    p.add_argument('--diffusion_ckpt', type=str, default='models/diffusion/fc_diffusion_latest.pth')
    p.add_argument('--output_dir', type=str, default='models/lora_finetuned')
    p.add_argument('--lora_r', type=int, default=8)
    p.add_argument('--lora_scale', type=float, default=1.0)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=0.0001)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--loss_recon_weight', type=float, default=1.0, help='Weight for reconstruction loss (encoder->decoder->x)')
    p.add_argument('--loss_diff_weight', type=float, default=1.0, help='Weight for diffusion denoising loss')
    p.add_argument('--no_pretrained', action='store_true', help='Do not load pretrained ckpt; train LoRA from scratch')
    p.add_argument('--dataset_base', type=str, default='test_dataset', help='dataset base name; samples are tmp/<base>_lh|_rh/FeatExt_Prepare/avgconnectivity.tiff')
    args = p.parse_args()
    work_dir = os.path.abspath(args.work_dir or os.path.dirname(os.path.abspath(__file__)))
    os.chdir(work_dir)
    if not os.path.isabs(args.encoder_ckpt):
        args.encoder_ckpt = os.path.join(work_dir, args.encoder_ckpt)
    args.decoder_ckpt = args.decoder_ckpt or args.encoder_ckpt
    if not os.path.isabs(args.decoder_ckpt):
        args.decoder_ckpt = os.path.join(work_dir, args.decoder_ckpt)
    if not os.path.isabs(args.diffusion_ckpt):
        args.diffusion_ckpt = os.path.join(work_dir, args.diffusion_ckpt)
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(work_dir, args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    from encoder_model import build_default_autoencoder, build_default_decoder
    from diffusion_model import build_fc_diffusion_unet
    from lora_utils import _inject_encoder_lora, _inject_decoder_lora, _inject_diffusion_lora
    device = args.device if torch.cuda.is_available() else 'cpu'
    tmp = os.path.join(work_dir, 'tmp')
    samples = []
    for name in [args.dataset_base + '_lh', args.dataset_base + '_rh']:
        path = os.path.join(tmp, name, 'FeatExt_Prepare', 'avgconnectivity.tiff')
        if os.path.exists(path):
            samples.append(path)
    if not samples:
        print('No avgconnectivity.tiff in tmp/test_dataset_lh or _rh. Run compute_avgconnectivity first.')
        return 1
    print('Using samples:', samples)
    encoder = build_default_autoencoder()
    if not args.no_pretrained and os.path.exists(args.encoder_ckpt):
        encoder.load_state_dict(torch.load(args.encoder_ckpt, map_location='cpu', weights_only=False), strict=False)
    encoder = _inject_encoder_lora(encoder, r=args.lora_r, scale=args.lora_scale).to(device)
    decoder = build_default_decoder()
    if not args.no_pretrained and os.path.exists(args.decoder_ckpt):
        decoder.load_state_dict(torch.load(args.decoder_ckpt, map_location='cpu', weights_only=False), strict=False)
    decoder = _inject_decoder_lora(decoder, r=args.lora_r, scale=args.lora_scale).to(device)
    diffusion = build_fc_diffusion_unet(in_channels=4)
    if not args.no_pretrained and os.path.exists(args.diffusion_ckpt):
        diffusion.load_state_dict(torch.load(args.diffusion_ckpt, map_location='cpu', weights_only=False), strict=False)
    diffusion = _inject_diffusion_lora(diffusion, r=args.lora_r, scale=args.lora_scale).to(device)
    enc_lora = [p for (n, p) in encoder.named_parameters() if 'lora' in n]
    dec_lora = [p for (n, p) in decoder.named_parameters() if 'lora' in n]
    diff_lora = [p for (n, p) in diffusion.named_parameters() if 'lora' in n or 'conv1' in n or 'conv2' in n]
    opt = torch.optim.Adam(enc_lora + dec_lora + diff_lora, lr=args.lr)
    for ep in range(args.epochs):
        encoder.train()
        decoder.train()
        diffusion.train()
        loss_recon_sum = 0.0
        loss_diff_sum = 0.0
        for path in samples:
            x = load_avg_fc_as_tensor(path, device)
            z0 = encoder.encode(x)
            x_recon = decoder(z0)
            loss_recon = nn.functional.mse_loss(x_recon, x)
            t = torch.zeros(1, dtype=torch.long, device=device)
            z_noisy = z0.detach() + 0.1 * torch.randn_like(z0, device=device)
            pred = diffusion(z_noisy, t)
            loss_diff = nn.functional.mse_loss(pred, z0.detach())
            loss = args.loss_recon_weight * loss_recon + args.loss_diff_weight * loss_diff
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_recon_sum += loss_recon.item()
            loss_diff_sum += loss_diff.item()
        n = len(samples)
        print('Epoch %d/%d loss_recon=%.6f loss_diff=%.6f' % (ep + 1, args.epochs, loss_recon_sum / n, loss_diff_sum / n))
    torch.save(encoder.state_dict(), os.path.join(args.output_dir, 'encoder.pt'))
    torch.save(decoder.state_dict(), os.path.join(args.output_dir, 'decoder.pt'))
    torch.save(diffusion.state_dict(), os.path.join(args.output_dir, 'diffusion.pt'))
    print('Saved encoder.pt, decoder.pt, diffusion.pt to', args.output_dir)
    return 0
if __name__ == '__main__':
    sys.exit(main())
