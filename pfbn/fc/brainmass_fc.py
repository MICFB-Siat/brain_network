import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn

def pearson_fc(ts):
    ts = np.asarray(ts, dtype=np.float64)
    if ts.ndim != 2 or len(ts) < 3 or (not np.isfinite(ts).all()):
        raise ValueError('Expected finite time x ROI array, at least 3 observations')
    if np.any(ts.std(0) < 1e-08):
        raise ValueError('Constant ROI time series')
    return np.corrcoef(ts.T).astype(np.float32)

class BrainMassFC(nn.Module):

    def __init__(self, n, checkpoint=None, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.n = n
        self.backbone = nn.ModuleList([nn.TransformerEncoderLayer(100, 10, 2048, batch_first=True) for _ in range(16)])
        self.load_report = {'pretrained': False, 'seed': seed}
        if checkpoint is not None:
            raw = torch.load(str(checkpoint), map_location='cpu', weights_only=True)
            state = raw
            for key in ['model', 'state_dict', 'online_network_state_dict', 'online_network', 'online_encoder']:
                if key in raw and isinstance(raw[key], dict):
                    state = raw[key]
                    break
            needed = self.backbone.state_dict()
            prefixes = set()
            for k in state:
                if 'attention_list.' in k:
                    prefixes.add(k.split('attention_list.')[0] + 'attention_list.')
            priority = ['module.online_encoder.encoder.attention_list.', 'online_encoder.encoder.attention_list.', 'module.online_network.encoder.attention_list.', 'online_network.encoder.attention_list.', 'module.encoder.attention_list.', 'encoder.attention_list.', 'attention_list.']
            prefixes = sorted(prefixes, key=lambda x: (priority.index(x) if x in priority else len(priority), x))
            matches = []
            for prefix in prefixes:
                mapped = {k[len(prefix):]: v for (k, v) in state.items() if k.startswith(prefix)}
                if set(needed) == set(mapped) and all((mapped[k].shape == needed[k].shape for k in needed)):
                    matches.append((prefix, mapped))
            if not matches:
                raise RuntimeError('No complete BrainMass encoder found; keys=' + str(list(state)[:12]))
            (prefix, mapped) = matches[0]
            self.backbone.load_state_dict(mapped, strict=True)
            self.load_report = {'pretrained': True, 'prefix': prefix, 'loaded_tensors': len(mapped), 'required_tensors': len(needed), 'coverage': 1.0, 'other_complete_prefixes': [x[0] for x in matches[1:]], 'checkpoint_sha256': hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()}
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        torch.manual_seed(seed + 10000)
        self.pre = nn.Sequential(nn.Linear(n, 128), nn.LeakyReLU(), nn.Linear(128, 100))
        self.post = nn.Sequential(nn.Linear(100, 128), nn.LeakyReLU(), nn.Linear(128, n))

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, fc):
        if fc.ndim != 3 or fc.shape[-2:] != (self.n, self.n):
            raise ValueError('FC must be batch x ROI x ROI')
        x = self.pre(fc)
        for layer in self.backbone:
            x = layer(x)
        x = self.post(x)
        x = x - x.mean(-1, keepdim=True)
        x = x / torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(1e-08)
        e = x @ x.transpose(-1, -2)
        return e

    def refine(self, fc, alpha):
        if not 0 <= alpha <= 1:
            raise ValueError('alpha outside [0,1]')
        return (1 - alpha) * fc + alpha * self(fc)

    def adapter_state(self):
        return {k: v.detach().cpu() for (k, v) in self.state_dict().items() if not k.startswith('backbone.')}

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--timeseries', required=True, help='.npy time x ROI')
    p.add_argument('--adapter', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    d = torch.load(a.adapter, map_location='cpu', weights_only=True)
    net = BrainMassFC(d['n'], a.checkpoint if d['branch'] == 'pretrained' else None, d['seed'])
    (missing, unexpected) = net.load_state_dict(d['adapter'], strict=False)
    assert not unexpected and all((k.startswith('backbone.') for k in missing))
    net = net.to(a.device).eval()
    base = pearson_fc(np.load(a.timeseries))
    with torch.no_grad():
        result = net.refine(torch.tensor(base, device=a.device)[None], d['alpha'])[0].cpu().numpy()
    if Path(a.output).exists():
        raise FileExistsError(a.output)
    np.save(a.output, result)
if __name__ == '__main__':
    main()
