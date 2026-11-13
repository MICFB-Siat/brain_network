import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
from pfbn import settings
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_common as C
from eval_common import log
import numpy as np
from sklearn.covariance import LedoitWolf
import torch
dev = torch.device('cuda')
RUNS = {'REST1': ('rfMRI_REST1_LR', 'rfMRI_REST1_RL'), 'REST2': ('rfMRI_REST2_LR', 'rfMRI_REST2_RL')}
CONS = settings.path('gradient_cohort', 'adaptive/consensus/young_adults_adaptive_consensus.npz')
CATO = os.path.join(C.SIX, 'assets', 'lausanne250_labels.npz')
FC_ATLASES = ('Micapipe', 'PFBN v0', 'PFBN v5|REST1', 'PFBN v5|REST2', 'v4_filt|REST1', 'v4_filt|REST2', 'CATO (Lausanne-219)')

def hcp_atlases():
    atl = {'Micapipe': C.load_labels(C.MICAPIPE)}
    z = np.load(CONS, allow_pickle=True)
    (l, r) = (z['labels_lh'].astype(np.int64), z['labels_rh'].astype(np.int64))
    if r.max() <= 100:
        r = np.where(r > 0, r + 100, 0)
    atl['PFBN v0'] = np.concatenate([l, r])
    for s in RUNS:
        atl[f'PFBN v5|{s}'] = C.load_labels(os.path.join(C.HREP, f'labels_v5_stim_t03_r30_{s}_clean.npy'))
        atl[f'v4_filt|{s}'] = C.load_labels(os.path.join(C.HREP, f'labels_v4_filt_t03_c1_{s}_clean.npy'))
    atl['CATO (Lausanne-219)'] = C.load_labels(CATO)
    for (k, v) in atl.items():
        assert v.shape == (C.NCORT,), k
        assert not np.intersect1d(np.unique(v[:C.NL][v[:C.NL] > 0]), np.unique(v[C.NL:][v[C.NL:] > 0])).size, f'{k}: lh/rh label overlap'
    return atl

def relabel(lab):
    u = np.unique(lab[lab > 0])
    m = np.zeros(lab.max() + 1, np.int64)
    m[u] = np.arange(1, len(u) + 1)
    return (np.where(lab > 0, m[lab], 0), len(u))

def load_run(sid, run):
    cache = os.path.join(C.SWEEP, 'input_cache', sid, run)
    meta = json.load(open(os.path.join(cache, 'complete.json')))
    assert meta['frames'] == 1195 and meta['cortex_vertices'] == C.NCORT
    x = np.load(os.path.join(cache, 'cortex_l2_drop5.npy'))
    assert x.shape == (1195, C.NCORT)
    return (x, meta['sha256'])

def homogeneity(Z, valid, lab, n):
    T = Z.shape[0]
    lab = np.where(valid, lab, 0)
    M = torch.zeros(len(lab), n + 1, device=dev)
    M[torch.arange(len(lab)), torch.from_numpy(lab).to(dev)] = 1.0
    S = Z @ M
    size = M.sum(0)
    ss = (S.double() ** 2).sum(0)
    size_d = size.double()
    hom = ((ss / (T - 1) - size_d) / (size_d * (size_d - 1)).clamp(min=1)).cpu().numpy()[1:]
    size = size.cpu().numpy().astype(np.int64)[1:]
    hom[size < 2] = np.nan
    return (hom, size)

class Pairs:

    def __init__(self, h):
        z = np.load(os.path.join(C.DCBC_SCR, f'geodesic_{h}_35mm.npz'))
        D = z['dist']
        keep = (D > 0) & (D <= C.MAX_DIST)
        self.I_np = z['i'][keep].astype(np.int64)
        J = z['j'][keep].astype(np.int64)
        assert (np.diff(self.I_np) >= 0).all()
        b = np.clip(np.ceil(D[keep]).astype(np.int64) - 1, 0, C.N_BINS - 1)
        self.n = C.HN[h]
        self.h = h
        self.I = torch.from_numpy(self.I_np).to(dev)
        self.J = torch.from_numpy(J).to(dev)
        self.bin = torch.from_numpy(b).to(dev)

    def set_data(self, Zh, block=2000):
        Zh = Zh - Zh.mean(0, keepdim=True)
        nrm = Zh.norm(dim=0)
        valid = nrm > 0
        Zh = Zh / torch.where(valid, nrm, torch.ones_like(nrm))
        r = torch.empty(len(self.I_np), dtype=torch.float32, device=dev)
        for a in range(0, self.n, block):
            b = min(a + block, self.n)
            (lo, hi) = np.searchsorted(self.I_np, [a, b])
            if hi > lo:
                Cb = Zh[:, a:b].T @ Zh
                r[lo:hi] = Cb[self.I[lo:hi] - a, self.J[lo:hi]]
                del Cb
        self.r = r
        self.ok = valid[self.I] & valid[self.J]

    def evaluate(self, lab_h):
        lab = torch.from_numpy(np.asarray(lab_h).astype(np.int64)).to(dev)
        (li, lj) = (lab[self.I], lab[self.J])
        ok = (li > 0) & (lj > 0) & self.ok
        out = {}
        for (name, m) in (('within', ok & (li == lj)), ('between', ok & (li != lj))):
            bb = self.bin[m]
            rr = self.r[m].double()
            n = torch.bincount(bb, minlength=C.N_BINS).double()
            s_ = torch.bincount(bb, weights=rr, minlength=C.N_BINS)
            s2 = torch.bincount(bb, weights=rr ** 2, minlength=C.N_BINS)
            mean = s_ / n.clamp(min=1)
            var = (s2 / n.clamp(min=1) - mean ** 2).clamp(min=0)
            (out[name], out['n_' + name], out['var_' + name]) = (mean.cpu().numpy(), n.cpu().numpy(), var.cpu().numpy())
        vb = (out['n_within'] > 1) & (out['n_between'] > 1)
        d = out['within'] - out['between']
        w = np.zeros(C.N_BINS)
        w[vb] = 1 / (out['var_within'][vb] / out['n_within'][vb] + out['var_between'][vb] / out['n_between'][vb])
        return (float(np.sum(w * d) / np.sum(w)), out['within'], out['between'], w)

def parcel_fc(x, lab):
    (lab, n) = relabel(lab)
    idx = np.where(lab > 0)[0]
    cnt = np.bincount(lab[idx], minlength=n + 1)[1:]
    W = np.zeros((len(lab), n), np.float32)
    W[idx, lab[idx] - 1] = 1.0 / cnt[lab[idx] - 1]
    return (x @ W).astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshard', type=int, default=1)
    ap.add_argument('--max-subjects', type=int, default=None)
    a = ap.parse_args()
    subs = json.load(open(os.path.join(C.PRIOR, 'selected_subjects.json')))
    assert len(subs) == 50
    subs = subs[:a.max_subjects] if a.max_subjects else subs
    subs = subs[a.shard::a.nshard]
    atl = hcp_atlases()
    names = sorted(atl)
    pairs = {h: Pairs(h) for h in C.HEMIS}
    out_dir = os.path.join(C.SCR, 'subjects')
    os.makedirs(out_dir, exist_ok=True)
    for sid in subs:
        op = os.path.join(out_dir, f'{sid}.npz')
        if os.path.exists(op):
            log('skip', sid)
            continue
        t0 = time.time()
        out = {}
        shas = {}
        for (s, runs) in RUNS.items():
            X = []
            for r in runs:
                (x, sh) = load_run(sid, r)
                shas[r] = sh
                for k in FC_ATLASES:
                    ts = parcel_fc(x, atl[k])
                    fc = np.corrcoef(ts.T)
                    iu = np.triu_indices(fc.shape[0], 1)
                    out[f'fc|{k}|{r}'] = fc[iu].astype(np.float32)
                    if k.startswith('CATO'):
                        tz = ts - ts.mean(0)
                        tz /= tz.std(0)
                        P = LedoitWolf().fit(tz).precision_
                        d = np.sqrt(np.diag(P))
                        pc = -P / d[:, None] / d[None]
                        out[f'fcpartial|{k}|{r}'] = pc[iu].astype(np.float32)
                X.append(x - x.mean(0, keepdims=True))
                del x
            log(f'{sid} {s}: loaded + FC {time.time() - t0:.0f}s')
            Xt = np.concatenate(X, 0)
            del X
            T = Xt.shape[0]
            Xg = torch.from_numpy(Xt).to(dev)
            del Xt
            sd = Xg.std(0, keepdim=True)
            valid_t = sd[0] > 0
            sd[sd == 0] = 1.0
            Z = (Xg - Xg.mean(0, keepdim=True)) / sd
            Z[:, ~valid_t] = 0.0
            valid = valid_t.cpu().numpy()
            del Xg
            for k in names:
                (lab, n) = relabel(atl[k])
                (hom, size) = homogeneity(Z, valid, lab, n)
                out[f'hom|{k}|{s}'] = hom
                out[f'size|{k}|{s}'] = size
            log(f'{sid} {s}: homogeneity {time.time() - t0:.0f}s')
            for h in C.HEMIS:
                P = pairs[h]
                P.set_data(Z[:, :C.NL] if h == 'lh' else Z[:, C.NL:])
                for k in names:
                    (dc, wi, be, w) = P.evaluate(C.hemi_lab(atl[k], h))
                    out[f'dcbc|{k}|{s}|{h}'] = np.float64(dc)
                    out[f'within|{k}|{s}|{h}'] = wi
                    out[f'between|{k}|{s}|{h}'] = be
                    out[f'w|{k}|{s}|{h}'] = w
                log(f'{sid} {s} {h}: DCBC {time.time() - t0:.0f}s')
            del Z
            torch.cuda.empty_cache()
        out['input_sha256'] = np.array(json.dumps(shas))
        out['seconds'] = np.float64(time.time() - t0)
        np.savez(op + '.tmp.npz', **out)
        os.replace(op + '.tmp.npz', op)
        log(f'{sid}: {time.time() - t0:.0f}s, DCBC REST2 ' + ', '.join((f"{k} {np.mean([out[f'dcbc|{k}|REST2|{h}'] for h in C.HEMIS]):.4f}" for k in names)))
    log('PASS_END')
if __name__ == '__main__':
    main()
