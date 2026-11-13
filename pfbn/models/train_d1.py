import argparse
import os
import numpy as np
import torch
from feature_extraction import load_encoder, load_avg_fc_as_tensor
from d1_model import D1Decoder

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--encoder', default='models/encoder_decoder/encoder_decoder_latest.pth')
    p.add_argument('--avgfc', required=True, help='path to avgconnectivity.tiff')
    p.add_argument('--fmri_avg', required=True, help='path to (T,V) .npy')
    p.add_argument('--hemisphere', default='lh', choices=['lh', 'rh'])
    p.add_argument('--epochs', type=int, default=1000)
    p.add_argument('--lr', type=float, default=0.0001)
    p.add_argument('--warmup_steps', type=int, default=500)
    p.add_argument('--out', default='models/d1')
    args = p.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    V = 29696 if args.hemisphere == 'lh' else 29716
    B_np = np.load(args.fmri_avg)
    if B_np.shape[1] != V and B_np.shape[0] == V:
        B_np = B_np.T
    assert B_np.shape[1] == V, (B_np.shape, V)
    B_mean = B_np.mean(axis=0, keepdims=True)
    B = torch.from_numpy(B_mean).float().unsqueeze(-1).to(device)
    encoder = load_encoder(args.encoder, device)
    with torch.no_grad():
        x = load_avg_fc_as_tensor(args.avgfc, device)
        zpattern = encoder.encode(x).detach()
    del encoder
    d1 = D1Decoder(vertex_count=V, time_steps=1, hidden_dim=512).to(device)
    opt = torch.optim.AdamW(d1.parameters(), lr=args.lr)
    warmup_steps = args.warmup_steps

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    for ep in range(1, args.epochs + 1):
        d1.train()
        pred = d1(zpattern)
        loss = torch.nn.functional.mse_loss(pred, B)
        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()
        print('Epoch %d loss=%.6f lr=%.2e' % (ep, loss.item(), scheduler.get_last_lr()[0]))
    os.makedirs(args.out, exist_ok=True)
    out_name = 'd1_latest_%s.pth' % args.hemisphere
    out_path = os.path.join(args.out, out_name)
    torch.save(d1.state_dict(), out_path)
    print('Saved to', out_path)
if __name__ == '__main__':
    main()
