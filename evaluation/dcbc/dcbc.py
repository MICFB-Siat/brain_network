import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
from pfbn import settings
import os, sys, json, time, argparse, hashlib
os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('GY_GPU', '1')
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_v, '8')
EXP = settings.out('dcbc_ds005498')
SCR = settings.scratch('dcbc_ds005498')
MV2 = settings.out('ds005498')
sys.path.insert(0, settings.code('evaluation', 'dcbc'))
import ds005498_common as C
from ds005498_common import log
import numpy as np, nibabel as nib, torch, pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
from scipy.stats import wilcoxon
RESULTS = os.path.join(EXP, 'results')
for d in (RESULTS, SCR):
    os.makedirs(d, exist_ok=True)
CONTE69 = settings.path('conte69', 'Conte69.{H}.midthickness.32k_fs_LR.surf.gii')
OLD_GEO = settings.path('gradient_cohort', 'analysis', 'cache', 'geodesic_{H}_35mm.npz')
ATLASES = {'v0': os.path.join(C.EXP1, 'results', 'SdTemplate.dlabel.nii'), 'v4_stim_t03_c1': os.path.join(MV2, 'v4', 'results', 'labels_v4_stim_t03_c1.npy'), 'v4_filt_t03_c1_clean': os.path.join(MV2, 'v4', 'results', 'labels_v4_filt_t03_c1_clean.npy'), 'v5_stim_t03_r30_clean': os.path.join(MV2, 'v5', 'results', 'labels_v5_stim_t03_r30_clean.npy'), 'Schaefer200': C.MICAPIPE}
REFS = ('Schaefer200', 'v0')
(MAX_DIST, BIN_W) = (35.0, 1.0)
N_BINS = int(round(MAX_DIST / BIN_W))
dev = torch.device('cuda')

def timings_put(stage, key, seconds, extra=None):
    p = os.path.join(RESULTS, 'timings.json')
    t = json.load(open(p)) if os.path.exists(p) else {}
    t.setdefault(stage, {})
    t[stage][key] = {'seconds': round(float(seconds), 2), 'finished': time.strftime('%Y-%m-%d %H:%M:%S')}
    if extra:
        t[stage][key].update(extra)
    json.dump(t, open(p, 'w'), indent=1)

def load_atlases():
    atl = {}
    for (k, p) in ATLASES.items():
        if p.endswith('.dlabel.nii'):
            lab = C.read_dlabel_cortex(p)
        elif p.endswith('.npz'):
            lab = np.load(p)['labels'].astype(np.int64)
        else:
            lab = np.load(p).astype(np.int64)
        assert lab.shape == (C.NCORT,), (k, lab.shape)
        atl[k] = lab
        log(f'atlas {k}: {len(np.unique(lab[lab > 0]))} parcels ({len(np.unique(lab[:C.NL][lab[:C.NL] > 0]))} lh / {len(np.unique(lab[C.NL:][lab[C.NL:] > 0]))} rh), {int((lab == 0).sum())} unassigned  <- {p}')
    return atl

def hemi_labels(lab, hemi):
    return lab[:C.NL] if hemi == 'lh' else lab[C.NL:]

def load_midthickness(hemi):
    g = nib.load(CONTE69.format(H=C.HEMI_LETTER[hemi]))
    xyz = g.darrays[0].data.astype(np.float64)
    assert xyz.shape == (C.NV, 3)
    return xyz

def mesh_graph(hemi):
    xyz = load_midthickness(hemi)
    cidx = C.cortex_idx(hemi)
    e = C.hemi_edges_cortex(hemi)
    n = C.HEMI_N[hemi]
    w = np.linalg.norm(xyz[cidx[e[:, 0]]] - xyz[cidx[e[:, 1]]], axis=1)
    G = sp.coo_matrix((w, (e[:, 0], e[:, 1])), shape=(n, n))
    G = (G + G.T).tocsr()
    return G

def geodesic_path(hemi, limit=MAX_DIST):
    return os.path.join(SCR, f'geodesic_{hemi}_{limit:g}mm.npz')

def build_geodesic(hemi, limit=MAX_DIST, chunk=2000, force=False):
    out = geodesic_path(hemi, limit)
    if os.path.exists(out) and (not force):
        return out
    t0 = time.time()
    G = mesh_graph(hemi)
    n = G.shape[0]
    (ii, jj, dd) = ([], [], [])
    for s in range(0, n, chunk):
        src = np.arange(s, min(s + chunk, n))
        D = dijkstra(G, directed=False, indices=src, limit=limit)
        (r, c) = np.nonzero(np.isfinite(D) & (D > 0))
        i = src[r]
        keep = i < c
        ii.append(i[keep].astype(np.int32))
        jj.append(c[keep].astype(np.int32))
        dd.append(D[r[keep], c[keep]].astype(np.float32))
        log(f'geodesic {hemi}: {src[-1] + 1}/{n} sources, {sum((len(x) for x in ii)):,} pairs, {time.time() - t0:.0f}s')
    i = np.concatenate(ii)
    j = np.concatenate(jj)
    d = np.concatenate(dd)
    o = np.lexsort((j, i))
    (i, j, d) = (i[o], j[o], d[o])
    el = time.time() - t0
    np.savez(out, i=i, j=j, dist=d, hemi=hemi, limit=limit, n_cortex=n, seconds=el, surface=CONTE69.format(H=C.HEMI_LETTER[hemi]), note='graph Dijkstra on cortex mesh edges (common2.hemi_edges_cortex), Conte69 midthickness edge lengths, medial wall excluded, i<j, hemisphere-cortex indexing')
    log(f'geodesic {hemi}: {len(i):,} pairs <= {limit} mm in {el:.0f}s -> {out} ({os.path.getsize(out) / 1000000.0:.0f} MB)')
    timings_put('geodesic', hemi, el, {'n_pairs': int(len(i)), 'chunk': chunk})
    return out

def crosscheck_geodesic(hemi):
    p = OLD_GEO.format(H=C.HEMI_LETTER[hemi])
    if not os.path.exists(p):
        return {'old_cache': None}
    z = np.load(geodesic_path(hemi))
    o = np.load(p)
    cidx = C.cortex_idx(hemi)
    f2c = -np.ones(C.NV, np.int64)
    f2c[cidx] = np.arange(len(cidx))
    (oi, oj) = (f2c[o['i'].astype(np.int64)], f2c[o['j'].astype(np.int64)])
    (lo, hi) = (np.minimum(oi, oj), np.maximum(oi, oj))
    k = np.lexsort((hi, lo))
    same_pairs = len(lo) == len(z['i']) and np.array_equal(lo[k], z['i']) and np.array_equal(hi[k], z['j'])
    res = {'old_cache': p, 'n_pairs_new': int(len(z['i'])), 'n_pairs_old': int(len(lo)), 'same_pairs': bool(same_pairs), 'max_abs_dist_diff': float(np.abs(o['dist'][k] - z['dist']).max()) if same_pairs else None}
    log(f'geodesic cross-check {hemi}: {res}')
    return res

class HemiPairs:

    def __init__(self, hemi):
        z = np.load(build_geodesic(hemi))
        self.hemi = hemi
        self.n = C.HEMI_N[hemi]
        D = z['dist']
        keep = (D > 0) & (D <= MAX_DIST)
        self.I_np = z['i'][keep].astype(np.int64)
        self.J_np = z['j'][keep].astype(np.int64)
        assert (np.diff(self.I_np) >= 0).all()
        b = np.clip(np.ceil(D[keep] / BIN_W).astype(np.int64) - 1, 0, N_BINS - 1)
        self.I = torch.from_numpy(self.I_np).to(dev)
        self.J = torch.from_numpy(self.J_np).to(dev)
        self.bin = torch.from_numpy(b).to(dev)
        self.r = None
        self.pair_ok = None
        log(f'{hemi}: {len(self.I_np):,} pairs <= {MAX_DIST} mm, {np.bincount(b, minlength=N_BINS).min():,}-{np.bincount(b, minlength=N_BINS).max():,} pairs per bin')

    def set_subject(self, Zh, block=2000):
        Zh = Zh - Zh.mean(0, keepdim=True)
        nrm = Zh.norm(dim=0)
        valid = nrm > 0
        nrm = torch.where(valid, nrm, torch.ones_like(nrm))
        Zh = Zh / nrm
        r = torch.empty(len(self.I_np), dtype=torch.float32, device=dev)
        for a in range(0, self.n, block):
            b = min(a + block, self.n)
            Cb = Zh[:, a:b].T @ Zh
            (lo, hi) = np.searchsorted(self.I_np, [a, b])
            if hi > lo:
                r[lo:hi] = Cb[self.I[lo:hi] - a, self.J[lo:hi]]
            del Cb
        self.r = r
        self.pair_ok = valid[self.I] & valid[self.J]
        self.n_invalid_vertices = int((~valid).sum())
        return self

    def evaluate(self, lab_h):
        lab = torch.from_numpy(np.asarray(lab_h).astype(np.int64)).to(dev)
        (li, lj) = (lab[self.I], lab[self.J])
        ok = (li > 0) & (lj > 0) & self.pair_ok
        same = ok & (li == lj)
        diff = ok & (li != lj)
        out = {}
        for (name, m) in (('within', same), ('between', diff)):
            bb = self.bin[m]
            rr = self.r[m].double()
            n = torch.bincount(bb, minlength=N_BINS).double()
            s = torch.bincount(bb, weights=rr, minlength=N_BINS)
            s2 = torch.bincount(bb, weights=rr ** 2, minlength=N_BINS)
            mean = s / n.clamp(min=1)
            var = (s2 / n.clamp(min=1) - mean ** 2).clamp(min=0)
            out[name] = mean.cpu().numpy()
            out['n_' + name] = n.cpu().numpy()
            out['var_' + name] = var.cpu().numpy()
        vb = (out['n_within'] > 1) & (out['n_between'] > 1)
        d = out['within'] - out['between']
        w = np.zeros(N_BINS)
        w[vb] = 1.0 / (out['var_within'][vb] / out['n_within'][vb] + out['var_between'][vb] / out['n_between'][vb])
        return dict(DCBC_weighted=float(np.sum(w * d) / np.sum(w)) if w.sum() > 0 else np.nan, DCBC_unweighted=float(np.mean(d[vb])) if vb.any() else np.nan, within=out['within'], between=out['between'], n_within=out['n_within'], n_between=out['n_between'], weights=w, n_bins_used=int(vb.sum()), n_pairs_within=int(out['n_within'].sum()), n_pairs_between=int(out['n_between'].sum()))

def paired(A, B):
    try:
        w = wilcoxon(A, B, alternative='two-sided')
        p = float(w.pvalue)
        st = float(w.statistic)
    except ValueError as e:
        (p, st) = (None, str(e))
    q = lambda x: {'median': float(np.median(x)), 'q1': float(np.percentile(x, 25)), 'q3': float(np.percentile(x, 75)), 'mean': float(np.mean(x)), 'sd': float(np.std(x, ddof=1))}
    return {'n': int(len(A)), 'a': q(A), 'b': q(B), 'diff_a_minus_b': q(A - B), 'wilcoxon_statistic': st, 'wilcoxon_p_two_sided': p, 'fraction_a_greater': float(np.mean(A > B))}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-subjects', type=int, default=None)
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    t_all = time.time()
    TAG = a.tag
    atl = load_atlases()
    (subs, runs) = C.fit_subjects()
    subs = subs[:a.max_subjects] if a.max_subjects else subs
    t0 = time.time()
    pairs = {h: HemiPairs(h) for h in C.HEMIS}
    geo_check = {h: crosscheck_geodesic(h) for h in C.HEMIS}
    t_geo = time.time() - t0
    rows = []
    curves = {}
    t_sub = time.time()
    for (si, s) in enumerate(subs):
        t1 = time.time()
        X = torch.from_numpy(C.load_cortex_bold(runs[s][C.REST_KEY])).to(dev)
        T = X.shape[0]
        for h in C.HEMIS:
            P = pairs[h]
            P.set_subject(X[:, :C.NL] if h == 'lh' else X[:, C.NL:])
            for (k, lab) in atl.items():
                r = P.evaluate(hemi_labels(lab, h))
                rows.append(dict(subject=s, hemi=h, atlas=k, DCBC_weighted=r['DCBC_weighted'], DCBC_unweighted=r['DCBC_unweighted'], n_bins_used=r['n_bins_used'], n_pairs_within=r['n_pairs_within'], n_pairs_between=r['n_pairs_between'], n_invalid_vertices=P.n_invalid_vertices, T=T, src=runs[s][C.REST_KEY]))
                for key in ('within', 'between', 'n_within', 'n_between', 'weights'):
                    curves.setdefault(f'{k}|{h}|{key}', []).append(r[key])
        del X
        torch.cuda.empty_cache()
        log(f'subject {si + 1}/{len(subs)} {s}: T={T}, ' + ', '.join((f"{k} {np.mean([x['DCBC_weighted'] for x in rows[-2 * len(atl):] if x['atlas'] == k]):.4f}" for k in atl)) + f', {time.time() - t1:.1f}s')
    per = pd.DataFrame(rows)
    per.to_csv(os.path.join(RESULTS, f'dcbc_per_subject{TAG}.csv'), index=False)
    np.savez(os.path.join(RESULTS, f'dcbc_bins{TAG}.npz'), subjects=np.array(subs), bin_edges=np.arange(0, MAX_DIST + BIN_W / 2, BIN_W), **{k.replace('|', '__'): np.stack(v) for (k, v) in curves.items()})
    sub = per.groupby(['subject', 'atlas'])[['DCBC_weighted', 'DCBC_unweighted']].mean().reset_index()
    stats = {'n_subjects': int(sub.subject.nunique()), 'reported_metric': 'DCBC_weighted (inverse-variance weighted mean over 1-mm bins, DCBC toolbox weighting=True convention); DCBC_unweighted = simple mean over bins', 'distance': 'graph Dijkstra on Conte69 fs_LR 32k midthickness mesh (cortex edges only, medial wall removed), limit 35 mm, 1-mm bins (0,1]..(34,35]', 'data': 'ses-1 resting run of each fit subject (common2.REST_KEY), vertices z-scored over time; label-0 and zero-variance vertices excluded from pairs', 'geodesic_crosscheck_vs_gradient_cohort_cache': geo_check, 'comparisons': {}}
    srows = []
    for k in atl:
        row = {'atlas': k}
        for metric in ('DCBC_weighted', 'DCBC_unweighted'):
            x = sub[sub.atlas == k].set_index('subject')[metric]
            row[f'{metric}_median'] = float(x.median())
            row[f'{metric}_mean'] = float(x.mean())
            row[f'{metric}_sd'] = float(x.std(ddof=1))
            for hh in C.HEMIS:
                row[f'{metric}_{hh}_mean'] = float(per[(per.atlas == k) & (per.hemi == hh)][metric].mean())
            for ref in REFS:
                if ref == k:
                    continue
                y = sub[sub.atlas == ref].set_index('subject')[metric]
                com = x.index.intersection(y.index)
                st = paired(x[com].values, y[com].values)
                stats['comparisons'][f'{metric}|{k}_vs_{ref}'] = st
                row[f'{metric}_p_vs_{ref}'] = st['wilcoxon_p_two_sided']
                row[f'{metric}_frac_gt_{ref}'] = st['fraction_a_greater']
                row[f'{metric}_diff_vs_{ref}_median'] = st['diff_a_minus_b']['median']
        srows.append(row)
    S = pd.DataFrame(srows)
    S.to_csv(os.path.join(RESULTS, f'dcbc_summary{TAG}.csv'), index=False)
    stats['seconds_total'] = time.time() - t_all
    stats['seconds_pairs_setup'] = t_geo
    stats['seconds_subjects'] = time.time() - t_sub
    json.dump(stats, open(os.path.join(RESULTS, f'dcbc_stats{TAG}.json'), 'w'), indent=1)
    timings_put('dcbc', 'all' + TAG, time.time() - t_all, {'n_subjects': len(subs), 'seconds_pairs_setup': round(t_geo, 1), 'seconds_subjects': round(time.time() - t_sub, 1)})
    cols = ['atlas', 'DCBC_weighted_median', 'DCBC_weighted_mean', 'DCBC_weighted_p_vs_Schaefer200', 'DCBC_weighted_p_vs_v0', 'DCBC_weighted_frac_gt_Schaefer200', 'DCBC_unweighted_median', 'DCBC_unweighted_p_vs_Schaefer200']
    print(S[[c for c in cols if c in S.columns]].to_string(index=False))
    log('DCBC_END')
if __name__ == '__main__':
    main()
