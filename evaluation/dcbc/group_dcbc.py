import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
from pfbn import settings
import os, sys, json, argparse, time, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, settings.code('pfbn', 'pipeline'))
sys.path.insert(0, settings.code('evaluation', 'dcbc'))
import numpy as np, pandas as pd, torch
import cohort_lib as R
import common as PC
import dcbc as D
from scipy.sparse.csgraph import dijkstra
from scipy.stats import wilcoxon, spearmanr
ap = argparse.ArgumentParser()
ap.add_argument('--exp_dir', required=True)
ap.add_argument('--groups', required=True)
ap.add_argument('--ref', required=True)
ap.add_argument('--tol', type=float, default=4.0)
ap.add_argument('--max_subjects', type=int, default=0)
a = ap.parse_args()
RES = os.path.join(a.exp_dir, 'results')
os.makedirs(RES, exist_ok=True)
G = [g.split(':') for g in a.groups.split(',')]
labels = [g[0] for g in G]
ATL = {}
for (lab, ca, cb) in G:
    ATL[lab, 'A'] = np.load(os.path.join(R.ATL, ca, R.FILES['PFBN v5'])).astype(np.int64)
    ATL[lab, 'B'] = np.load(os.path.join(R.ATL, cb, R.FILES['PFBN v5'])).astype(np.int64)
MICA = R.micapipe()
t0 = time.time()
pairs = {h: D.HemiPairs(h) for h in ('lh', 'rh')}
rows = []
for (lab, ca, cb) in G:
    cj = json.load(open(os.path.join(R.RUNNER, 'cohorts', f'{cb}.json')))
    os.environ['PFBN_COHORT'] = os.path.join(R.RUNNER, 'cohorts', f'{cb}.json')
    PC.DENOISE = cj.get('denoise', 'none')
    PC.DROP = int(cj.get('drop', 5))
    subs = cj['subjects'][:a.max_subjects] if a.max_subjects else cj['subjects']
    for (si, s) in enumerate(subs):
        runs = [r if isinstance(r, dict) else {'path': r} for r in s['runs']]
        X = np.concatenate([(lambda x: x - x.mean(0))(PC.load_run(r)[0]) for r in runs], 0)
        Xt = torch.from_numpy(X).to(D.dev)
        res = {}
        for (h, sl) in (('lh', slice(0, R.NL)), ('rh', slice(R.NL, R.NCORT))):
            P = pairs[h]
            P.set_subject(Xt[:, sl])
            for other in labels:
                res.setdefault(other, []).append(P.evaluate(ATL[other, 'A'][sl])['DCBC_weighted'])
            res.setdefault('Micapipe', []).append(P.evaluate(MICA[sl])['DCBC_weighted'])
        for (k, v) in res.items():
            rows.append(dict(subject=s['sid'], group=lab, atlas=k, own=k == lab, DCBC=float(np.mean(v)), frames=int(X.shape[0])))
        del Xt
        torch.cuda.empty_cache()
    R.log(f'DCBC {lab}: {len(subs)} subjects, {time.time() - t0:.0f}s')
dc = pd.DataFrame(rows)
dc.to_csv(os.path.join(RES, 'dcbc_per_subject.csv'), index=False)
graphs = {h: D.mesh_graph(h) for h in ('lh', 'rh')}
edges = {h: PC.hemi_edges_cortex(h) for h in ('lh', 'rh')}
SL = {'lh': slice(0, R.NL), 'rh': slice(R.NL, R.NCORT)}

def border(lab, h):
    e = edges[h]
    x = lab[SL[h]]
    (a1, b1) = (x[e[:, 0]], x[e[:, 1]])
    d = (a1 != b1) & (a1 > 0) & (b1 > 0)
    m = np.zeros(len(x), bool)
    m[e[d, 0]] = True
    m[e[d, 1]] = True
    return m
_DT = {}

def dist_to_border(key, lab):
    if key not in _DT:
        out = np.zeros(R.NCORT)
        for h in ('lh', 'rh'):
            src = np.where(border(lab, h))[0]
            out[SL[h]] = dijkstra(graphs[h], directed=False, indices=src, min_only=True)
        _DT[key] = out
    return _DT[key]
mica_units = MICA.copy()

def displacement(kx, ky):
    (lx, ly) = (ATL[kx], ATL[ky])
    (dx, dy) = (dist_to_border(kx, lx), dist_to_border(ky, ly))
    bx = np.concatenate([border(lx, h) for h in ('lh', 'rh')])
    by = np.concatenate([border(ly, h) for h in ('lh', 'rh')])
    v = np.full(R.NCORT, np.nan)
    v[by] = dx[by]
    v[bx] = np.where(by[bx], 0.0, dy[bx])
    keys = np.unique(mica_units[mica_units > 0])
    per = np.array([np.nanmean(v[mica_units == k]) if np.isfinite(v[mica_units == k]).any() else np.nan for k in keys])
    prec = float(np.mean(dx[by] <= a.tol))
    rec = float(np.mean(dy[bx] <= a.tol))
    f1 = 2 * prec * rec / (prec + rec)
    return (v, per, keys, dict(precision=prec, recall=rec, F1=f1, median_mm=float(np.nanmedian(v)), mean_mm=float(np.nanmean(v))))
disp_rows = []
f1_rows = []
maps = {}
comps = []
for lab in labels:
    comps.append((f'{lab} test-retest', (lab, 'A'), (lab, 'B'), lab, 'test-retest'))
    if lab != a.ref:
        comps.append((f'{lab} vs {a.ref} (A)', (lab, 'A'), (a.ref, 'A'), lab, 'vs ref A'))
        comps.append((f'{lab} vs {a.ref} (B)', (lab, 'B'), (a.ref, 'B'), lab, 'vs ref B'))
for (name, kx, ky, lab, kind) in comps:
    (v, per, keys, f) = displacement(kx, ky)
    maps[name] = v.astype(np.float32)
    maps[name + '|parcel'] = per.astype(np.float32)
    f1_rows.append(dict(comparison=name, group=lab, kind=kind, **f))
    for (k, p) in zip(keys, per):
        disp_rows.append(dict(comparison=name, group=lab, kind=kind, mica_parcel=int(k), displacement_mm=float(p)))
    R.log(f"displacement {name}: F1 {f['F1']:.3f}, median {f['median_mm']:.2f} mm")
dp = pd.DataFrame(disp_rows)
dp.to_csv(os.path.join(RES, 'displacement_parcels.csv'), index=False)
pd.DataFrame(f1_rows).to_csv(os.path.join(RES, 'boundary_f1.csv'), index=False)
np.savez(os.path.join(RES, 'maps.npz'), mica_keys=np.unique(MICA[MICA > 0]), **{k.replace(' ', '_'): v for (k, v) in maps.items()}, **{f'atlas_{l}_{h}': ATL[l, h] for l in labels for h in ('A', 'B')})
st = {'groups': {l: {'A': ca, 'B': cb} for (l, ca, cb) in G}, 'ref': a.ref, 'tol_mm': a.tol, 'dcbc': {}, 'displacement': {}}
for lab in labels:
    d = dc[dc.group == lab]
    own = d[d.atlas == lab].set_index('subject').DCBC
    oth = d[~d.own & (d.atlas != 'Micapipe')].groupby('subject').DCBC.mean().loc[own.index]
    mic = d[d.atlas == 'Micapipe'].set_index('subject').DCBC.loc[own.index]
    w1 = wilcoxon(own, oth)
    w2 = wilcoxon(own, mic)
    st['dcbc'][lab] = dict(n=int(len(own)), own_median=float(own.median()), others_median=float(oth.median()), micapipe_median=float(mic.median()), own_vs_others_W=float(w1.statistic), own_vs_others_p=float(w1.pvalue), frac_own_gt_others=float(np.mean(own > oth)), own_vs_micapipe_W=float(w2.statistic), own_vs_micapipe_p=float(w2.pvalue), frac_own_gt_micapipe=float(np.mean(own > mic)))
    if lab != a.ref:
        tr = dp[dp.comparison == f'{lab} test-retest'].set_index('mica_parcel').displacement_mm
        ra = dp[dp.comparison == f'{lab} vs {a.ref} (A)'].set_index('mica_parcel').displacement_mm
        rb = dp[dp.comparison == f'{lab} vs {a.ref} (B)'].set_index('mica_parcel').displacement_mm
        ok = tr.notna() & ra.notna() & rb.notna()
        w = wilcoxon(ra[ok], tr[ok])
        rr = spearmanr(ra[ok], rb[ok])
        st['displacement'][lab] = dict(n_parcels=int(ok.sum()), test_retest_median_mm=float(tr[ok].median()), vs_ref_A_median_mm=float(ra[ok].median()), vs_ref_B_median_mm=float(rb[ok].median()), vsrefA_vs_testretest_W=float(w.statistic), vsrefA_vs_testretest_p=float(w.pvalue), pattern_reliability_spearman_r=float(rr.correlation), pattern_reliability_p=float(rr.pvalue))
st['seconds'] = time.time() - t0
json.dump(st, open(os.path.join(RES, 'stats.json'), 'w'), indent=1)
R.log('BOUNDARY_GROUPS_END')
