import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
from pfbn import settings
import os, sys, json, time, argparse, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gradient_tools import *
import pandas as pd
from scipy.sparse.csgraph import dijkstra
from scipy.stats import rankdata
from concurrent.futures import ProcessPoolExecutor
EXP = os.path.join(ROOT, 'gradient_zoom')
RES = os.path.join(EXP, 'results')
os.makedirs(RES, exist_ok=True)
GM_HCP = os.path.join(C.HREP, 'gmap_REST2_{h}.npy')
NSPIN = int(os.environ.get('NSPIN', 1000))

def hcp_atlases():
    from hcp_retest import hcp_atlases as ha
    a = ha()
    return {'Micapipe': a['Micapipe'], 'PFBN v0': a['PFBN v0'], 'PFBN v5': a['PFBN v5|REST1']}

def roi_job(args):
    (name, lab, gpath, h, rois, kind) = args
    os.environ['OMP_NUM_THREADS'] = '2'
    G = C.mesh_graph(h)
    A = C.adjacency(h)
    e = C.edges(h)
    cidx = C.cortex_idx(h)
    (g, nz) = load_gmap(gpath, h) if kind == 'npy' else load_func_cortex(gpath.format(H=C.HL[h]), h)
    ridge = nz & (g >= np.percentile(g[nz], THR))
    d_r = dijkstra(G, directed=False, indices=np.flatnonzero(ridge), min_only=True)
    lab_h = C.hemi_lab(lab, h)
    full = full_hemi(lab, h)
    idx = spins(h, NSPIN)
    obs = {}
    vd = {}
    for (r, m) in rois.items():
        (o, B) = ridge_stats(lab_h, e, A, ridge, d_r, m)
        obs[r] = o
        vd[r] = d_r[B].astype(np.float32)
    null = {r: [] for r in rois}
    for s in range(len(idx)):
        rl = full[idx[s]][cidx]
        Bs = C.boundary_mask(rl, e)
        for (r, m) in rois.items():
            B = Bs & m
            R = ridge & m
            null[r].append(dict(nB=int(B.sum()), tpB=int((B & C.dilate(R, A, K_TOL)).sum()), nR=int(R.sum()), tpR=int((R & C.dilate(B, A, K_TOL)).sum()), dsum=float(d_r[B].sum())))
    return (name, h, obs, null, vd)

def summarise(res, tag):
    by = {}
    for (nm, h, obs, null, vd) in res:
        by.setdefault(nm, {})[h] = (obs, null, vd)
    (rows, nulls, verts) = ([], {}, [])
    for (nm, d) in by.items():
        for r in d['lh'][0]:
            o = pool(d['lh'][0][r], d['rh'][0][r])
            (P_, R_, F) = f1(o)
            dist = o['dsum'] / max(o['nB'], 1)
            nl = [pool(a, b) for (a, b) in zip(d['lh'][1][r], d['rh'][1][r])]
            nf = np.array([f1(x)[2] for x in nl])
            nd = np.array([x['dsum'] / max(x['nB'], 1) for x in nl])
            rows.append(dict(map=tag, atlas=nm, roi=r, precision=P_, recall=R_, F1=F, F1_null_mean=nf.mean(), F1_null_sd=nf.std(ddof=1), F1_z=(F - nf.mean()) / nf.std(ddof=1), F1_p_one_sided=(1 + np.sum(nf >= F)) / (len(nf) + 1), dist_mm=dist, dist_null_mean=nd.mean(), dist_z=(dist - nd.mean()) / nd.std(ddof=1), dist_p_one_sided=(1 + np.sum(nd <= dist)) / (len(nd) + 1), n_boundary=o['nB'], n_ridge=o['nR'], n_spin=len(nf)))
            key = f'{tag}|{nm}|{r}'
            nulls[key + '|F1'] = nf
            nulls[key + '|dist'] = nd
            for h in C.HEMIS:
                verts.append(pd.DataFrame(dict(map=tag, atlas=nm, roi=r, hemi=h, dist_mm=d[h][2][r])))
    return (pd.DataFrame(rows), nulls, pd.concat(verts))

def pair_border(names_h, e, a_set, b_set):
    (a, b) = (names_h[e[:, 0]], names_h[e[:, 1]])
    hit = np.isin(a, a_set) & np.isin(b, b_set) | np.isin(a, b_set) & np.isin(b, a_set)
    m = np.zeros(len(names_h), bool)
    m[e[hit, 0]] = True
    m[e[hit, 1]] = True
    return m

def honey_job(args):
    (tag, path, kind, h, names_h) = args
    A = C.adjacency(h)
    e = C.edges(h)
    cidx = C.cortex_idx(h)
    idx = spins(h, NSPIN)
    (g, nz) = load_gmap(path, h) if kind == 'npy' else load_func_cortex(path.format(H=C.HL[h]), h)
    codes = {n: i + 1 for (i, n) in enumerate(sorted(set(names_h) - {''}))}
    ci = np.array([codes.get(n, 0) for n in names_h])
    borders = {'V1|V2': pair_border(names_h, e, ['V1'], ['V2']), '4|3a+3b': pair_border(names_h, e, ['4'], ['3a', '3b']), 'All HCP-MMP borders': C.boundary_mask(ci, e)}
    borders = {k: C.dilate(v, A, 1) for (k, v) in borders.items()}
    rk = np.full(len(g), np.nan)
    rk[nz] = rankdata(g[nz])
    n = nz.sum()
    out = {}
    for (k, m) in borders.items():
        mm = m & nz
        rest = nz & ~m
        auc = (np.nanmean(rk[mm]) - (mm.sum() + 1) / 2) / rest.sum() if mm.sum() else np.nan
        full = np.zeros(C.NV, bool)
        full[cidx] = m
        null = []
        for s in range(len(idx)):
            ms = full[idx[s]][cidx] & nz
            ns = ms.sum()
            null.append((np.nanmean(rk[ms]) - (ns + 1) / 2) / (n - ns) if ns else np.nan)
        out[k] = dict(auc=float(auc), ratio=float(np.median(g[mm]) / np.median(g[rest])), pct=float(np.mean(g[nz] < np.mean(g[mm])) * 100), n=int(mm.sum()), null=np.array(null))
    top = g[nz] >= np.percentile(g[nz], THR)
    ridge_cv = float(np.std(g[nz][top]) / np.mean(g[nz][top]))
    return (tag, h, out, ridge_cv, g, nz)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=24)
    ap.add_argument('--part', default='ABC')
    a = ap.parse_args()
    t0 = time.time()
    names = mmp_names()
    A = {h: C.adjacency(h) for h in C.HEMIS}
    rois = {h: {r: roi_mask(names[h], ar, A[h], 3) for (r, ar) in ROIS.items()} for h in C.HEMIS}
    for h in C.HEMIS:
        rois[h]['Whole cortex'] = np.ones(C.HN[h], bool)
    log('ROI vertices', {h: {r: int(m.sum()) for (r, m) in rois[h].items()} for h in C.HEMIS})
    np.savez(os.path.join(RES, 'roi_masks.npz'), **{f'{h}|{r}': m for h in C.HEMIS for (r, m) in rois[h].items()})
    inputs = [GM_HCP.format(h=h) for h in C.HEMIS] + [MMP, C.MICAPIPE, C.SPINS]
    if 'A' in a.part:
        atl = hcp_atlases()
        jobs = [(nm, lab, GM_HCP, h, rois[h], 'npy') for (nm, lab) in atl.items() for h in C.HEMIS]
        with ProcessPoolExecutor(a.workers) as ex:
            res = list(ex.map(roi_job, jobs))
        (df, nulls, verts) = summarise(res, 'HCP gmap_REST2')
        df.to_csv(os.path.join(RES, 'roi_f1_distance_HCP.csv'), index=False)
        np.savez(os.path.join(RES, 'roi_nulls_HCP.npz'), **nulls)
        verts.to_csv(os.path.join(RES, 'roi_vertex_dist_HCP.csv.gz'), index=False)
        log('A done\n' + df[['atlas', 'roi', 'F1', 'F1_z', 'dist_mm', 'dist_z']].to_string(index=False))
    if 'B' in a.part:
        maps = [(f'tyf {g} iter{it}', os.path.join(TYF, f'{g}_iter{it}_' + '{H}.func.gii'), 'gii') for g in ('children', 'adolescents', 'young_adults', 'older_adults', 'age_independent') for it in (0, 8)]
        maps += [('HCP gmap_REST2', GM_HCP, 'npy'), ('Lifespan cohort gmap', os.path.join(ATL, 'lifespan_all/results/gmap_{h}.npy'), 'npy')]
        jobs = [(t, p, k, h, names[h]) for (t, p, k) in maps for h in C.HEMIS]
        with ProcessPoolExecutor(a.workers) as ex:
            res = list(ex.map(honey_job, jobs))
        by = {}
        for (t, h, o, cv, g, nz) in res:
            by.setdefault(t, {})[h] = (o, cv, g, nz)
        ref = by['HCP gmap_REST2']
        rows = []
        nulls = {}
        for (t, d) in by.items():
            for k in d['lh'][0]:
                auc = (d['lh'][0][k]['auc'] + d['rh'][0][k]['auc']) / 2
                nl = (d['lh'][0][k]['null'] + d['rh'][0][k]['null']) / 2
                rows.append(dict(map=t, border=k, AUC=auc, AUC_null_mean=np.nanmean(nl), AUC_z=(auc - np.nanmean(nl)) / np.nanstd(nl, ddof=1), median_ratio=(d['lh'][0][k]['ratio'] + d['rh'][0][k]['ratio']) / 2, percentile_of_border_mean=(d['lh'][0][k]['pct'] + d['rh'][0][k]['pct']) / 2, ridge_cv=(d['lh'][1] + d['rh'][1]) / 2, r_with_HCP_gmap_REST2=np.mean([np.corrcoef(d[h][2][d[h][3] & ref[h][3]], ref[h][2][d[h][3] & ref[h][3]])[0, 1] for h in C.HEMIS])))
                nulls[f'{t}|{k}'] = nl
        hc = pd.DataFrame(rows)
        hc.to_csv(os.path.join(RES, 'honeycomb_canonical_borders.csv'), index=False)
        np.savez(os.path.join(RES, 'honeycomb_nulls.npz'), **nulls)
        log('B done\n' + hc.to_string(index=False))
        inputs += sorted(glob.glob(os.path.join(TYF, '*_iter[08]_*.func.gii'))) + [os.path.join(ATL, f'lifespan_all/results/gmap_{h}.npy') for h in C.HEMIS]
    if 'C' in a.part:
        mica = C.load_labels(C.MICAPIPE)
        jobs = []
        whole = {h: {'Whole cortex': np.ones(C.HN[h], bool)} for h in C.HEMIS}
        for g in ('children', 'adolescents', 'young_adults', 'older_adults'):
            gp = os.path.join(TYF, f'{g}_iter8_' + '{H}.func.gii')
            sets = {'Micapipe': mica}
            for half in ('A', 'B'):
                p = os.path.join(ATL, f'life_{g}_{half}', 'results', 'labels_v5_stim_t03_r30_clean.npy')
                sets[f'PFBN v5 ({half})'] = C.load_labels(p)
                inputs.append(p)
            for (nm, lab) in sets.items():
                for h in C.HEMIS:
                    jobs.append((f'{g}|{nm}', lab, gp, h, whole[h], 'gii'))
        gp = os.path.join(TYF, 'age_independent_iter8_{H}.func.gii')
        sets = {'Micapipe': mica, 'PFBN v5': C.load_labels(os.path.join(ATL, 'lifespan_all/results/labels_v5_stim_t03_r30_clean.npy')), 'PFBN v5 w/o dynamics': C.load_labels(os.path.join(ATL, 'lifespan_all/results/labels_v4_stim_t03_c1_clean.npy'))}
        v0 = sorted(glob.glob(os.path.join(ATL, 'lifespan_all/v0/results/*.npz')) + glob.glob(os.path.join(ATL, 'lifespan_all/v0/results/*labels*.npy')))
        log('lifespan v0 candidates', v0)
        for (nm, lab) in sets.items():
            for h in C.HEMIS:
                jobs.append((f'age_independent|{nm}', lab, gp, h, whole[h], 'gii'))
        with ProcessPoolExecutor(a.workers) as ex:
            res = list(ex.map(roi_job, jobs))
        (df, nulls, _) = summarise(res, 'tyf iter8')
        df[['group', 'atlas']] = df['atlas'].str.split('|', expand=True)
        df.to_csv(os.path.join(RES, 'lifespan_vs_tyf_maps.csv'), index=False)
        np.savez(os.path.join(RES, 'lifespan_vs_tyf_nulls.npz'), **nulls)
        log('C done\n' + df[['group', 'atlas', 'F1', 'F1_z', 'dist_mm', 'dist_z']].to_string(index=False))
    write_sha(inputs + [os.path.abspath(__file__), os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gradient_tools.py'), os.path.join(EVAL_CODE, 'eval_common.py'), settings.code('evaluation', 'dcbc', 'hcp_retest.py')], os.path.join(RES, f'inputs_sha256_{a.part}.txt'))
    log(f'GRADIENT_ZOOM_END {time.time() - t0:.0f}s')
if __name__ == '__main__':
    main()
