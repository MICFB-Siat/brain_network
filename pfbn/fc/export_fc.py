import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
from pfbn import settings
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse, json, itertools, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from train_fc_adapter import load_data, input_root
from brainmass_fc import BrainMassFC, pearson_fc

def manual_corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    return float(np.dot(x, y) / np.sqrt(np.dot(x, x) * np.dot(y, y)))

def main(root):
    assert (root / 'results/COMPLETE.json').exists()
    cases = load_data(root)
    src = input_root(root)
    labels = np.load(src / 'atlas_labels.npz')
    mask = (labels['Micapipe'] > 0) & (labels['PFBN_v5'] > 0)
    scores = pd.read_csv(root / 'results/individual_metrics.csv')
    summary = pd.read_csv(root / 'results/comparison_summary.csv')
    errs = []
    avg_errors = []
    exports = []
    diversity = []
    branches = {'BrainMass_v7': 'pretrained', 'Random_backbone': 'random'}
    for (i, c) in enumerate(cases):
        s = c['subject']
        a = c['atlas']
        p = np.load(root / 'results/fc' / f'{s}__{a}.npz')
        idx = np.searchsorted(c['ks'], labels[a][mask])
        for (method, branch) in branches.items():
            expected = np.mean([np.load(root / 'results' / f'fc_{branch}_seed{k}.npz')['fc'][i] for k in range(3)], axis=0)
            avg_errors.append(float(np.max(np.abs(expected - p[method]))))
        response = np.load(src / f'{s}__responses.npz')['month1'][mask]
        if np.isfinite(response).all():
            for method in ['Pearson', 'BrainMass_v7', 'Random_backbone']:
                expected = manual_corr(np.abs(p[method][0, 1:201][idx]), np.abs(response))
                row = scores[(scores.subject == s) & (scores.atlas == a) & (scores.target == 'month1') & (scores.predictor == method) & (scores.metric == 'Pearson_strength')]
                assert len(row) == 1
                errs.append(abs(expected - float(row.r.iloc[0])))
        for (method, name) in [('Pearson', 'PFBN_V5' if a == 'PFBN_v5' else 'Micapipe'), ('BrainMass_v7', 'PFBN_V7' if a == 'PFBN_v5' else 'Micapipe_BrainMass')]:
            folder = root / 'exports' / name
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f'{s}_FC.npy'
            if path.exists():
                raise FileExistsError(path)
            np.save(path, p[method][1:201, 1:201])
            exports.append(str(path.relative_to(root)))
        diversity.append(dict(subject=s, atlas=a, Pearson_vs_V7_fc_r=manual_corr(p['Pearson'][np.triu_indices(220, 1)], p['BrainMass_v7'][np.triu_indices(220, 1)])))
    pvalues = []
    for method in ['Pearson', 'BrainMass_v7', 'Random_backbone']:
        part = scores[(scores.target == 'month1') & (scores.predictor == method) & (scores.metric == 'Pearson_strength')]
        w = part.pivot(index='subject', columns='atlas', values='r').dropna()
        d = (w.PFBN_v5 - w.Micapipe).to_numpy()
        d = d[d != 0]
        ranks = rankdata(np.abs(d))
        observed = float(np.sum(ranks[d > 0]))
        total = []
        for signs in itertools.product([0, 1], repeat=len(d)):
            total.append(float(np.dot(signs, ranks)))
        total = np.asarray(total)
        p = min(1.0, 2 * min(np.mean(total <= observed + 1e-10), np.mean(total >= observed - 1e-10)))
        saved = summary[(summary.target == 'month1') & (summary.predictor == method) & (summary.metric == 'Pearson_strength')].P.iloc[0]
        assert abs(p - saved) < 1e-12, (method, p, saved)
        pvalues.append(dict(method=method, enumerated_P=p))
    ts = np.concatenate(cases[0]['runs']).astype(np.float64)
    center = ts - ts.mean(0)
    norm = center / np.linalg.norm(center, axis=0)
    base = pearson_fc(ts)
    pearson_error = float(np.max(np.abs(base - norm.T @ norm)))
    adapter = torch.load(root / 'models/adapter_pretrained_seed0.pt', map_location='cpu', weights_only=True)
    net = BrainMassFC(220, settings.path('brainmass_checkpoint'), 0).cuda().eval()
    (missing, unexpected) = net.load_state_dict(adapter['adapter'], strict=False)
    assert not unexpected and len(missing) == 192 and all((x.startswith('backbone.') for x in missing))
    with torch.no_grad():
        replay = net.refine(torch.tensor(base, device='cuda')[None], adapter['alpha'])[0].cpu().numpy()
    saved = np.load(root / 'results/fc_pretrained_seed0.npz')['fc'][0]
    replay_error = float(np.max(np.abs(replay - saved)))
    assert replay_error < 1e-05, replay_error
    assert max(errs) < 1e-12 and max(avg_errors) < 1e-07 and (pearson_error < 1e-06)
    pd.DataFrame(diversity).to_csv(root / 'results/FC_change_diagnostics.csv', index=False)
    result = dict(max_score_reconstruction_error=max(errs), max_seed_averaging_error=max(avg_errors), max_Pearson_reference_error=pearson_error, max_model_replay_error=replay_error, exact_pvalue_checks=pvalues, n_exported_cortical_matrices=len(exports), checkpoint=net.load_report, exports=exports)
    (root / 'results/validation.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for (k, v) in result.items() if k not in ['exports', 'checkpoint']}, indent=2), flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    a = p.parse_args()
    main(Path(a.root))
