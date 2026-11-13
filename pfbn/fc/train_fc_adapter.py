import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
from pfbn import settings
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse, json, time, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from brainmass_fc import BrainMassFC, pearson_fc
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.mha.set_fastpath_enabled(False)
DEVICE = 'cuda'
EPOCHS = [0, 2, 5, 10, 20]
ALPHAS = [0.0, 0.1, 0.25, 0.5, 1.0]

def input_root(root):
    return Path(settings.path('dbs_inputs'))

def load_data(root):
    source = input_root(root)
    mf = json.loads((source / 'manifest.json').read_text())
    cases = []
    for s in mf['subjects']:
        for a in mf['atlases']:
            with np.load(source / f'{s}__{a}.npz') as d:
                runs = [d['ts'][int(l):int(r)].copy() for (l, r) in d['segments']]
                assert len(runs) == 4 and all((x.shape == (184, 220) for x in runs))
                cases.append(dict(subject=s, atlas=a, runs=runs, ks=d['ks'].copy()))
    return cases

def corr(x, y):
    if not np.isfinite(x).all() or not np.isfinite(y).all() or np.std(x) < 1e-12 or (np.std(y) < 1e-12):
        return float('nan')
    return float(np.corrcoef(x, y)[0, 1])

def fc_score(pred, truth):
    ix = np.triu_indices(200, 1)
    p = pred[1:201, 1:201][ix]
    y = truth[1:201, 1:201][ix]
    return dict(fc_r=corr(p, y), fc_mse=float(np.mean((p - y) ** 2)), offdiag_sd=float(p.std()), offdiag_mean=float(p.mean()))

def tensor(x):
    return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=DEVICE)

def fit(root, branch, seed, cases, train_ids, max_epochs, callback=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    net = BrainMassFC(220, settings.path('brainmass_checkpoint') if branch == 'pretrained' else None, seed).to(DEVICE)
    pairs = []
    for c in cases:
        for i in train_ids:
            target = pearson_fc(np.concatenate([c['runs'][j] for j in train_ids if j != i]))
            pairs.append((c['runs'][i], target))
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=0.0003, weight_decay=5e-05)
    rng = np.random.default_rng(seed)
    mask = ~torch.eye(220, dtype=torch.bool, device=DEVICE)
    history = []
    if callback:
        callback(net, 0)
    for ep in range(1, max_epochs + 1):
        started = time.time()
        net.train()
        order = rng.permutation(len(pairs))
        losses = []
        for j in range(0, len(order), 8):
            ids = order[j:j + 8]
            xs = []
            ys = []
            for k in ids:
                (run, target) = pairs[int(k)]
                if ep % 3:
                    st = int(rng.integers(0, len(run) - 92 + 1))
                    view = run[st:st + 92]
                else:
                    view = run
                xs.append(pearson_fc(view))
                ys.append(target)
            x = tensor(xs)
            y = tensor(ys)
            e = net(x)
            loss = torch.mean((e[:, mask] - y[:, mask]) ** 2)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad], 1.0)
            opt.step()
            losses.append(float(loss.detach()))
        net.eval()
        history.append(dict(epoch=ep, loss=float(np.mean(losses)), seconds=time.time() - started))
        print('EPOCH', branch, seed, ep, history[-1], flush=True)
        if callback and ep in EPOCHS:
            callback(net, ep)
    return (net, history)

def predict(net, bases):
    net.eval()
    out = []
    with torch.no_grad():
        for j in range(0, len(bases), 8):
            out.extend(net(tensor(bases[j:j + 8])).cpu().numpy())
    return np.asarray(out)

def select(root, branch):
    cases = load_data(root)
    bases = [pearson_fc(np.concatenate(c['runs'][:2])) for c in cases]
    truths = [pearson_fc(c['runs'][2]) for c in cases]
    rows = []

    def callback(net, ep):
        encoded = predict(net, bases)
        for alpha in ALPHAS:
            for (c, b, e, y) in zip(cases, bases, encoded, truths):
                rows.append(dict(subject=c['subject'], atlas=c['atlas'], branch=branch, epoch=ep, alpha=alpha, **fc_score((1 - alpha) * b + alpha * e, y)))
        pd.DataFrame(rows).to_csv(root / 'selection' / f'{branch}.csv', index=False)
    (net, history) = fit(root, branch, 0, cases, [0, 1], max(EPOCHS), callback)
    (root / 'selection' / f'{branch}_load.json').write_text(json.dumps(net.load_report, indent=2))
    pd.DataFrame(history).to_csv(root / 'selection' / f'{branch}_loss.csv', index=False)

def choose(root):
    configs = {}
    tables = []
    for branch in ['pretrained', 'random']:
        tab = pd.read_csv(root / 'selection' / f'{branch}.csv')
        assert len(tab) == 28 * len(EPOCHS) * len(ALPHAS) and tab.fc_r.notna().all()
        summary = tab.groupby(['branch', 'epoch', 'alpha'], as_index=False).agg(fc_r=('fc_r', 'mean'), fc_mse=('fc_mse', 'mean'), n=('fc_r', 'size'))
        summary = summary.sort_values(['fc_r', 'alpha', 'epoch'], ascending=[False, True, True])
        win = summary.iloc[0]
        configs[branch] = dict(epoch=int(win.epoch), alpha=float(win.alpha), validation_fc_r=float(win.fc_r), selection='rest runs1/2 fit, run3 validation; pooled equally over both atlases and all patients')
        tables.append(summary)
    pd.concat(tables).to_csv(root / 'results/selection_summary.csv', index=False)
    (root / 'results/selected.json').write_text(json.dumps(configs, indent=2))
    print('SELECTED', configs, flush=True)

def final(root, branch, seed):
    cases = load_data(root)
    cfg = json.loads((root / 'results/selected.json').read_text())[branch]
    ep = cfg['epoch']
    alpha = cfg['alpha']
    bases = [pearson_fc(np.concatenate(c['runs'][:3])) for c in cases]
    (net, h) = fit(root, branch, seed, cases, [0, 1, 2], ep)
    encoded = predict(net, bases)
    diag = []
    test_matrices = []
    for (c, b, e) in zip(cases, bases, encoded):
        pred = (1 - alpha) * b + alpha * e
        truth = pearson_fc(c['runs'][3])
        test_matrices.append(pred)
        diag.append(dict(subject=c['subject'], atlas=c['atlas'], branch=branch, seed=seed, **fc_score(pred, truth)))
    pd.DataFrame(diag).to_csv(root / 'results' / f'test_{branch}_seed{seed}.csv', index=False)
    np.savez_compressed(root / 'results' / f'test_{branch}_seed{seed}.npz', fc=np.asarray(test_matrices))
    del net
    torch.cuda.empty_cache()
    bases = [pearson_fc(np.concatenate(c['runs'])) for c in cases]
    (net, h) = fit(root, branch, seed, cases, [0, 1, 2, 3], ep)
    encoded = predict(net, bases)
    matrices = np.asarray([(1 - alpha) * b + alpha * e for (b, e) in zip(bases, encoded)])
    np.savez_compressed(root / 'results' / f'fc_{branch}_seed{seed}.npz', fc=matrices, encoded=encoded)
    torch.save(dict(adapter=net.adapter_state(), n=220, alpha=alpha, seed=seed, branch=branch), root / 'models' / f'adapter_{branch}_seed{seed}.pt')
    (root / 'results' / f'done_{branch}_seed{seed}.json').write_text(json.dumps(dict(branch=branch, seed=seed, config=cfg, load_report=net.load_report, trainable_parameters=sum((p.numel() for p in net.parameters() if p.requires_grad)), epochs=ep), indent=2))
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--mode', choices=['select', 'choose', 'final'], required=True)
    p.add_argument('--branch', choices=['pretrained', 'random'])
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    root = Path(a.root)
    for d in ['selection', 'results', 'models']:
        (root / d).mkdir(exist_ok=True)
    if a.mode == 'choose':
        choose(root)
    elif a.mode == 'select':
        select(root, a.branch)
    else:
        final(root, a.branch, a.seed)
