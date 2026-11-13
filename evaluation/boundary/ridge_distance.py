import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
from pfbn import settings
import os, sys, json, time, argparse
os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('GY_GPU', '1')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dcbc as D
import ds005498_common as C
from ds005498_common import log
import numpy as np, pandas as pd
from scipy.sparse.csgraph import dijkstra
SPINS = settings.path('spin_rotations')
GMAP = os.path.join(D.MV2, 'results', 'gmap_rest_{h}.npy')
THRESHOLDS = (85, 90)

def load_gmap(hemi):
    g = np.load(GMAP.format(h=hemi)).astype(np.float64)
    g = g[C.cortex_idx(hemi)] if g.shape[0] == C.NV else g
    assert g.shape == (C.HEMI_N[hemi],), g.shape
    nz = np.isfinite(g) & (g != 0)
    log(f'gmap_rest {hemi}: {int(nz.sum())} usable cortex vertices, {int((~np.isfinite(g)).sum())} NaN, {int((g == 0).sum())} zero')
    return (g, nz)

def boundary_mask(lab_h, e):
    (la, lb) = (lab_h[e[:, 0]], lab_h[e[:, 1]])
    d = (la != lb) & (la > 0) & (lb > 0)
    m = np.zeros(len(lab_h), bool)
    m[e[d, 0]] = True
    m[e[d, 1]] = True
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-spin', type=int, default=1000)
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    t_all = time.time()
    TAG = a.tag
    atl = D.load_atlases()
    sp_ = np.load(SPINS)
    n_spin = min(a.n_spin, sp_['idx_L'].shape[0])
    rows = []
    nulls = {}
    for h in C.HEMIS:
        t0 = time.time()
        G = D.mesh_graph(h)
        e = C.hemi_edges_cortex(h)
        cidx = C.cortex_idx(h)
        n = C.HEMI_N[h]
        (g, nz) = load_gmap(h)
        idx = sp_['idx_' + C.HEMI_LETTER[h]][:n_spin]
        (d_ridge, ridge) = ({}, {})
        for thr in THRESHOLDS:
            cut = np.percentile(g[nz], thr)
            ridge[thr] = nz & (g >= cut)
            d_ridge[thr] = dijkstra(G, directed=False, indices=np.flatnonzero(ridge[thr]), min_only=True)
            assert np.isfinite(d_ridge[thr]).all()
            np.save(os.path.join(D.SCR, f'dist_to_ridge_{h}_p{thr}.npy'), d_ridge[thr].astype(np.float32))
            log(f'{h} p{thr}: cut={cut:.5f}, {int(ridge[thr].sum())} ridge vertices, distance map in {time.time() - t0:.1f}s')
        for (k, lab) in atl.items():
            lab_h = D.hemi_labels(lab, h)
            full = np.zeros(C.NV, np.int64)
            full[cidx] = lab_h
            bm = boundary_mask(lab_h, e)
            d_bnd = dijkstra(G, directed=False, indices=np.flatnonzero(bm), min_only=True)
            t1 = time.time()
            null = np.empty((len(THRESHOLDS), n_spin))
            for s in range(n_spin):
                rot = full[idx[s]][cidx]
                bs = boundary_mask(rot, e)
                for (ti, thr) in enumerate(THRESHOLDS):
                    null[ti, s] = d_ridge[thr][bs].mean()
            for (ti, thr) in enumerate(THRESHOLDS):
                obs = float(d_ridge[thr][bm].mean())
                rev = float(d_bnd[ridge[thr]].mean())
                nl = null[ti]
                nulls[f'{k}|{thr}|{h}'] = nl
                rows.append(dict(atlas=k, threshold_pct=thr, hemi=h, boundary_to_ridge_mm=obs, ridge_to_boundary_mm=rev, symmetric_mm=(obs + rev) / 2, null_mean=float(nl.mean()), null_sd=float(nl.std(ddof=1)), z=float((obs - nl.mean()) / nl.std(ddof=1)), null_percentile=float(np.mean(nl <= obs)), p_one_sided_closer=float((1 + np.sum(nl <= obs)) / (n_spin + 1)), n_boundary=int(bm.sum()), n_ridge=int(ridge[thr].sum()), n_spin=n_spin))
            log(f'{h} {k}: ' + ', '.join((f"p{r['threshold_pct']} b->r {r['boundary_to_ridge_mm']:.2f} (null {r['null_mean']:.2f}+-{r['null_sd']:.2f}, z {r['z']:.2f}) r->b {r['ridge_to_boundary_mm']:.2f}" for r in rows[-len(THRESHOLDS):])) + f', spins {time.time() - t1:.1f}s')
        D.timings_put('boundary_distance', h + TAG, time.time() - t0)
    df = pd.DataFrame(rows)
    both = []
    for k in atl:
        for thr in THRESHOLDS:
            r = df[(df.atlas == k) & (df.threshold_pct == thr)].set_index('hemi')
            nl = (nulls[f'{k}|{thr}|lh'] + nulls[f'{k}|{thr}|rh']) / 2
            obs = float(r.boundary_to_ridge_mm.mean())
            nulls[f'{k}|{thr}|both'] = nl
            both.append(dict(atlas=k, threshold_pct=thr, hemi='both', boundary_to_ridge_mm=obs, ridge_to_boundary_mm=float(r.ridge_to_boundary_mm.mean()), symmetric_mm=float(r.symmetric_mm.mean()), null_mean=float(nl.mean()), null_sd=float(nl.std(ddof=1)), z=float((obs - nl.mean()) / nl.std(ddof=1)), null_percentile=float(np.mean(nl <= obs)), p_one_sided_closer=float((1 + np.sum(nl <= obs)) / (n_spin + 1)), n_boundary=int(r.n_boundary.sum()), n_ridge=int(r.n_ridge.sum()), n_spin=n_spin))
    df = pd.concat([df, pd.DataFrame(both)], ignore_index=True)
    df.to_csv(os.path.join(D.RESULTS, f'boundary_distance{TAG}.csv'), index=False)
    np.savez(os.path.join(D.RESULTS, f'boundary_distance_null{TAG}.npz'), **{k.replace('|', '__'): v for (k, v) in nulls.items()})
    D.timings_put('boundary_distance', 'all' + TAG, time.time() - t_all, {'n_spin': n_spin})
    print(df[df.hemi == 'both'][['atlas', 'threshold_pct', 'boundary_to_ridge_mm', 'ridge_to_boundary_mm', 'symmetric_mm', 'null_mean', 'null_sd', 'z', 'null_percentile']].to_string(index=False))
    log('BD_END')
if __name__ == '__main__':
    main()
