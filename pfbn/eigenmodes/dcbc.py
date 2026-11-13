import argparse
import os
import time
import numpy as np
import common as C
import geodesic as G

def hcp_run_paths(sid, runs=C.HCP_RUNS):
    return [f'{C.HCP_ROOT}/{sid}_3T_rfMRI_REST_fix/{sid}/MNINonLinear/Results/{r}/{r}_Atlas_MSMAll_hp2000_clean.dtseries.nii' for r in runs]
DENOISE_MODES = ('none', 'gsr')
TMP_CACHE_MODES = ('all', 'pairr', 'none')
DROP_FRAMES = 5

def _paths_tag(paths, denoise='none'):
    import hashlib
    tag = f'{len(paths)}f_' + hashlib.sha1('\n'.join(paths).encode()).hexdigest()[:8]
    return tag if denoise == 'none' else f'{tag}_{denoise}'

def ts_cache_path(sid, paths, denoise='none'):
    return os.path.join(C.TMP_DIR, f'ts_{sid}_{_paths_tag(paths, denoise)}.npz')

def denoise_run(x, valid, mode='gsr'):
    if mode == 'none':
        return x
    if mode != 'gsr':
        raise ValueError(mode)
    x = np.ascontiguousarray(x[DROP_FRAMES:], dtype=np.float32)
    T = x.shape[0]
    t = np.arange(T, dtype=np.float64)
    t = (t - t.mean()) / t.std()
    try:
        import torch
        dev = C.torch_device()
        xt = torch.from_numpy(x).to(dev)

        def proj(xt, cols):
            X = torch.tensor(np.column_stack(cols), dtype=torch.float32, device=dev)
            beta = torch.linalg.solve(X.T @ X, X.T @ xt)
            return xt - X @ beta
        xt = proj(xt, [np.ones(T), t])
        gs = xt[:, torch.from_numpy(np.asarray(valid, bool)).to(dev)].mean(1).cpu().numpy().astype(np.float64)
        dgs = np.gradient(gs)
        xt = proj(xt, [np.ones(T), (gs - gs.mean()) / (gs.std() + 1e-12), (dgs - dgs.mean()) / (dgs.std() + 1e-12)])
        out = xt.cpu().numpy()
        del xt
        if dev.type == 'cuda':
            torch.cuda.empty_cache()
        return out
    except ImportError:

        def proj(x, cols):
            X = np.column_stack(cols).astype(np.float32)
            beta = np.linalg.solve(X.T @ X, X.T @ x)
            return x - X @ beta
        x = proj(x, [np.ones(T), t])
        gs = x[:, valid].mean(1).astype(np.float64)
        dgs = np.gradient(gs)
        return proj(x, [np.ones(T), (gs - gs.mean()) / (gs.std() + 1e-12), (dgs - dgs.mean()) / (dgs.std() + 1e-12)])

def load_subject_timeseries(sid, runs=C.HCP_RUNS, use_cache=True, verbose=True, paths=None, denoise='none', write_cache=True):
    import nibabel as nib
    paths = list(paths) if paths else hcp_run_paths(sid, runs)
    p = ts_cache_path(sid, paths, denoise)
    if use_cache and os.path.exists(p):
        z = np.load(p)
        return (z['Z'], z['valid'])
    t0 = time.time()
    n_c = C.cortex_axis()['n_cortex']
    (Zs, valid) = ([], np.ones(n_c, bool))
    for f in paths:
        r = os.path.basename(f)
        img = nib.load(f)
        ai = C.cortex_axis(f)
        assert ai['n_cortex'] == n_c and np.array_equal(ai['cortex2full'], C.cortex_axis()['cortex2full']), 'cortex axis mismatch'
        x = np.array(img.dataobj[:, :n_c], dtype=np.float32, copy=True)
        pre_ok = np.isfinite(x).all(0) & (x.std(0) > 1e-06)
        x = denoise_run(x, pre_ok, denoise)
        x -= x.mean(0, keepdims=True)
        sd = x.std(0, keepdims=True)
        ok = np.isfinite(sd[0]) & (sd[0] > 1e-06) & np.isfinite(x).all(0)
        valid &= ok
        sd[~np.isfinite(sd) | (sd <= 1e-06)] = 1.0
        x /= sd
        Zs.append(x)
        if verbose:
            C.log(f'{sid} {r}: {x.shape}, invalid vertices so far {(~valid).sum()}, {time.time() - t0:.0f}s')
    Z = np.concatenate(Zs, 0)
    Z[:, ~valid] = 0.0
    Z = np.nan_to_num(Z)
    if use_cache and write_cache:
        np.savez(p, Z=Z, valid=valid, paths=np.array(paths), denoise=denoise)
    return (Z, valid)

def pair_correlations(Z, valid, I, J, H, device=None, block=6144, verbose=True):
    import torch
    device = device or C.torch_device()
    ai = C.cortex_axis()
    T = Z.shape[0]
    r = np.full(len(I), np.nan, np.float32)
    t0 = time.time()
    for (k, h) in enumerate(C.HEMIS):
        sel = np.flatnonzero(H == k)
        (lo_all, hi_all) = (sel[0], sel[-1] + 1)
        off = 0 if h == 'L' else ai['n_L']
        n_h = ai[f'n_{h}']
        i_loc = I[lo_all:hi_all].astype(np.int64) - off
        j_loc = J[lo_all:hi_all].astype(np.int64) - off
        assert (np.diff(i_loc) >= 0).all(), 'pairs must be sorted by I within hemisphere'
        with torch.no_grad():
            Zh = torch.from_numpy(np.ascontiguousarray(Z[:, off:off + n_h])).to(device)
            for a in range(0, n_h, block):
                b = min(a + block, n_h)
                Cb = Zh[:, a:b].T @ Zh / T
                (lo, hi) = np.searchsorted(i_loc, [a, b])
                if hi > lo:
                    ii = torch.from_numpy(i_loc[lo:hi] - a).to(device)
                    jj = torch.from_numpy(j_loc[lo:hi]).to(device)
                    r[lo_all + lo:lo_all + hi] = Cb[ii, jj].float().cpu().numpy()
                del Cb
            del Zh
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    bad = ~(valid[I] & valid[J])
    r[bad] = np.nan
    if verbose:
        C.log(f'pair correlations: {len(r):,} pairs, {bad.sum():,} invalid, mean r {np.nanmean(r):.4f}, {time.time() - t0:.1f}s')
    return r

class DCBCEvaluator:

    def __init__(self, r, max_dist=35.0, bin_width=1.0, device=None, limit=35.0, pairs=None):
        import torch
        self.device = device or C.torch_device()
        (I, J, D, H) = pairs if pairs is not None else G.load_geodesic_cortex(limit)
        self.n_pairs = len(I)
        keep = np.isfinite(r) & (D > 0) & (D <= max_dist)
        self.bin_edges = np.arange(0, max_dist + bin_width / 2, bin_width)
        self.n_bins = len(self.bin_edges) - 1
        b = np.ceil(D[keep] / bin_width).astype(np.int64) - 1
        b = np.clip(b, 0, self.n_bins - 1)
        self.I = torch.from_numpy(I[keep].astype(np.int64)).to(self.device)
        self.J = torch.from_numpy(J[keep].astype(np.int64)).to(self.device)
        self.bin = torch.from_numpy(b).to(self.device)
        self.r = torch.from_numpy(r[keep].astype(np.float32)).to(self.device)
        self.n_used = int(keep.sum())
        self.ai = C.cortex_axis()

    @classmethod
    def for_subject(cls, sid, runs=C.HCP_RUNS, use_cache=True, device=None, verbose=True, paths=None, denoise='none', tmp_cache='all', **kw):
        if tmp_cache not in TMP_CACHE_MODES:
            raise ValueError(tmp_cache)
        paths = list(paths) if paths else hcp_run_paths(sid, runs)
        p = os.path.join(C.TMP_DIR, f'pairr_{sid}_{_paths_tag(paths, denoise)}_35mm.npy')
        pairs = kw.pop('pairs', None) or G.load_geodesic_cortex()
        if use_cache and os.path.exists(p):
            r = np.load(p)
        else:
            (Z, valid) = load_subject_timeseries(sid, runs, use_cache, verbose, paths=paths, denoise=denoise, write_cache=tmp_cache == 'all')
            r = pair_correlations(Z, valid, pairs[0], pairs[1], pairs[3], device, verbose=verbose)
            del Z
            if use_cache and tmp_cache in ('all', 'pairr'):
                np.save(p, r)
        return cls(r, device=device, pairs=pairs, **kw)

    def evaluate(self, labels_full, weighting=True):
        import torch
        lab = torch.from_numpy(np.asarray(labels_full)[self.ai['cortex2full']].astype(np.int64)).to(self.device)
        (li, lj) = (lab[self.I], lab[self.J])
        ok = (li > 0) & (lj > 0)
        same = ok & (li == lj)
        diff = ok & (li != lj)
        out = {}
        for (name, m) in (('within', same), ('between', diff)):
            bb = self.bin[m]
            rr = self.r[m]
            n = torch.bincount(bb, minlength=self.n_bins).double()
            s = torch.bincount(bb, weights=rr.double(), minlength=self.n_bins)
            s2 = torch.bincount(bb, weights=rr.double() ** 2, minlength=self.n_bins)
            mean = s / n.clamp(min=1)
            var = (s2 / n.clamp(min=1) - mean ** 2).clamp(min=0)
            out[name] = mean.cpu().numpy()
            out['n_' + name] = n.cpu().numpy()
            out['var_' + name] = var.cpu().numpy()
        valid_bins = (out['n_within'] > 1) & (out['n_between'] > 1)
        d = out['within'] - out['between']
        w = np.zeros(self.n_bins)
        w[valid_bins] = 1.0 / (out['var_within'][valid_bins] / out['n_within'][valid_bins] + out['var_between'][valid_bins] / out['n_between'][valid_bins])
        res = dict(DCBC=float(np.sum(w * d) / np.sum(w)) if w.sum() > 0 else np.nan, DCBC_unweighted=float(np.nanmean(d[valid_bins])) if valid_bins.any() else np.nan, within=out['within'], between=out['between'], diff=d, weights=w, n_within=out['n_within'], n_between=out['n_between'], bin_edges=self.bin_edges, n_pairs_used=int(ok.sum().item()))
        if not weighting:
            res['DCBC'] = res['DCBC_unweighted']
        return res

class SpinRotations:

    def __init__(self, n_rot=1000, seed=0, use_cache=True, device=None, verbose=True):
        (self.n_rot, self.seed) = (n_rot, seed)
        p = os.path.join(C.CACHE_DIR, f'spin_rotations_n{n_rot}_seed{seed}.npz')
        if use_cache and os.path.exists(p):
            z = np.load(p)
            self.idx = {'L': z['idx_L'], 'R': z['idx_R']}
            self.R = z['R']
            return
        from scipy.spatial.transform import Rotation
        t0 = time.time()
        rots = Rotation.random(n_rot, random_state=np.random.RandomState(seed)).as_matrix()
        M = np.diag([-1.0, 1.0, 1.0])
        X = {h: C.load_surf(h, 'sphere')[0] for h in C.HEMIS}
        X = {h: X[h] / np.linalg.norm(X[h], axis=1, keepdims=True) for h in C.HEMIS}
        self.idx = {h: np.zeros((n_rot, C.N_VERT), np.int32) for h in C.HEMIS}
        device = device or C.torch_device()
        import torch
        Xt = {h: torch.from_numpy(X[h]).float().to(device) for h in C.HEMIS}
        with torch.no_grad():
            for k in range(n_rot):
                for h in C.HEMIS:
                    Rk = rots[k] if h == 'L' else M @ rots[k] @ M
                    Xr = Xt[h] @ torch.from_numpy(Rk.T).float().to(device)
                    out = torch.empty(C.N_VERT, dtype=torch.int64, device=device)
                    for a in range(0, C.N_VERT, 8192):
                        out[a:a + 8192] = torch.argmax(Xt[h][a:a + 8192] @ Xr.T, dim=1)
                    self.idx[h][k] = out.cpu().numpy().astype(np.int32)
                if verbose and (k + 1) % 200 == 0:
                    C.log(f'spin: {k + 1}/{n_rot} rotations, {time.time() - t0:.0f}s')
        self.R = rots
        if use_cache:
            np.savez(p, idx_L=self.idx['L'], idx_R=self.idx['R'], R=rots, seed=seed)
        if verbose:
            C.log(f'spin rotations built in {time.time() - t0:.0f}s -> {p}')

    def rotate(self, values_full, k, fill_medial=0):
        v = np.asarray(values_full)
        out = np.empty_like(v)
        for h in C.HEMIS:
            o = C.HEMI_OFFSET[h]
            out[o:o + C.N_VERT] = v[o:o + C.N_VERT][self.idx[h][k]]
        if fill_medial is not None:
            out[~C.cortex_axis()['cortex_mask_full']] = fill_medial
        return out
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--subjects', nargs='+', required=True)
    ap.add_argument('--parcellation', nargs='+', default=['schaefer200'])
    ap.add_argument('--n-spin', type=int, default=10)
    ap.add_argument('--out', default=None)
    ap.add_argument('--denoise', choices=DENOISE_MODES, default='none')
    a = ap.parse_args()
    spins = SpinRotations(1000, 0) if a.n_spin > 0 else None
    parcs = {n: C.load_parcellation(n) for n in a.parcellation}
    results = {}
    for sid in a.subjects:
        t0 = time.time()
        ev = DCBCEvaluator.for_subject(sid, denoise=a.denoise)
        results[sid] = {}
        for (n, lab) in parcs.items():
            res = ev.evaluate(lab)
            nulls = [ev.evaluate(spins.rotate(lab, k))['DCBC'] for k in range(a.n_spin)]
            results[sid][n] = dict(DCBC=res['DCBC'], DCBC_unweighted=res['DCBC_unweighted'], within=res['within'], between=res['between'], null=nulls, null_mean=float(np.mean(nulls)) if nulls else None, null_sd=float(np.std(nulls)) if nulls else None)
            print(f"{sid} {n}: DCBC={res['DCBC']:.4f} (unweighted {res['DCBC_unweighted']:.4f}); rotated null {(np.mean(nulls) if nulls else float('nan')):.4f} +- {(np.std(nulls) if nulls else float('nan')):.4f} (n={a.n_spin}); {time.time() - t0:.0f}s", flush=True)
        del ev
    if a.out:
        C.save_json(results, a.out)
