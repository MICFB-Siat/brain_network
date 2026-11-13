import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_common as C
from eval_common import log
import numpy as np, pandas as pd
from scipy.sparse.csgraph import dijkstra
from concurrent.futures import ProcessPoolExecutor
(K_TOL, THR) = (2, 85)
DS = {'Micapipe': C.MICAPIPE, 'PFBN v0': os.path.join(C.V0, 'results', 'SdTemplate.dlabel.nii'), 'PFBN v5 w/o dynamics': os.path.join(C.MV2, 'v4/results/labels_v4_stim_t03_c1.npy'), 'v4_filt': os.path.join(C.MV2, 'v4/results/labels_v4_filt_t03_c1_clean.npy'), 'PFBN v5': os.path.join(C.MV2, 'v5/results/labels_v5_stim_t03_r30_clean.npy')}

def hcp_sets():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from hcp_retest import hcp_atlases
    atl = hcp_atlases()
    sets = []
    for k in ('Micapipe', 'PFBN v0', 'CATO (Lausanne-219)'):
        sets.append((k, atl[k], 'REST2'))
    for k in ('PFBN v5', 'v4_filt'):
        sets.append((k + '|REST1', atl[f'{k}|REST1'], 'REST2'))
        sets.append((k + '|REST2', atl[f'{k}|REST2'], 'REST1'))
    return sets

def load_gmap(path, h):
    g = np.load(path.format(h=h)).astype(np.float64)
    g = g[C.cortex_idx(h)] if g.shape[0] == C.NV else g
    nz = np.isfinite(g) & (g != 0)
    return (g, nz)

def one_hemi(args):
    (name, lab, gpath, h, n_spin) = args
    os.environ['OMP_NUM_THREADS'] = '2'
    G = C.mesh_graph(h)
    A = C.adjacency(h)
    e = C.edges(h)
    cidx = C.cortex_idx(h)
    (g, nz) = load_gmap(gpath, h)
    ridge = nz & (g >= np.percentile(g[nz], THR))
    d_r = dijkstra(G, directed=False, indices=np.flatnonzero(ridge), min_only=True)
    R_dil = C.dilate(ridge, A, K_TOL)
    idx = np.load(C.SPINS)['idx_' + C.HL[h]][:n_spin]
    lab_h = C.hemi_lab(lab, h)
    full = np.zeros(C.NV, np.int64)
    full[cidx] = lab_h

    def stats(lh_):
        B = C.boundary_mask(lh_, e)
        B_dil = C.dilate(B, A, K_TOL)
        return (dict(nB=int(B.sum()), tpB=int((B & R_dil).sum()), nR=int(ridge.sum()), tpR=int((ridge & B_dil).sum()), dist=float(d_r[B].mean())), B)
    (obs, B) = stats(lab_h)
    null = [stats(full[idx[s]][cidx])[0] for s in range(n_spin)]
    return (name, h, obs, null, d_r[B].astype(np.float32))

def f1(d):
    P = d['tpB'] / d['nB']
    R = d['tpR'] / d['nR']
    return (P, R, 2 * P * R / (P + R))

def run(dataset, sets, n_spin, workers):
    jobs = [(nm, lab, gp, h, n_spin) for (nm, lab, gp) in sets for h in C.HEMIS]
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(one_hemi, jobs))
    by = {}
    for (nm, h, obs, null, dv) in res:
        by.setdefault(nm, {})[h] = (obs, null, dv)
    rows = []
    vert = []
    nulls = {}
    for (nm, d) in by.items():
        pool = lambda a, b: {k: a[k] + b[k] for k in ('nB', 'tpB', 'nR', 'tpR')}
        o = pool(d['lh'][0], d['rh'][0])
        (P, R, F) = f1(o)
        dist = (d['lh'][0]['dist'] + d['rh'][0]['dist']) / 2
        nf = np.array([f1(pool(a, b))[2] for (a, b) in zip(d['lh'][1], d['rh'][1])])
        nd = np.array([(a['dist'] + b['dist']) / 2 for (a, b) in zip(d['lh'][1], d['rh'][1])])
        rows.append(dict(dataset=dataset, atlas=nm, precision=P, recall=R, F1=F, F1_null_mean=nf.mean(), F1_null_sd=nf.std(ddof=1), F1_z=(F - nf.mean()) / nf.std(ddof=1), F1_p_one_sided=(1 + np.sum(nf >= F)) / (len(nf) + 1), dist_mm=dist, dist_null_mean=nd.mean(), dist_null_sd=nd.std(ddof=1), dist_z=(dist - nd.mean()) / nd.std(ddof=1), dist_p_one_sided=(1 + np.sum(nd <= dist)) / (len(nd) + 1), n_boundary=o['nB'], n_ridge=o['nR'], n_spin=len(nf), tol_edges=K_TOL, ridge_pct=THR))
        nulls[f'{nm}__F1'] = nf
        nulls[f'{nm}__dist'] = nd
        for h in C.HEMIS:
            vert.append(pd.DataFrame(dict(dataset=dataset, atlas=nm, hemi=h, dist_mm=d[h][2])))
    return (pd.DataFrame(rows), pd.concat(vert), nulls)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-spin', type=int, default=1000)
    ap.add_argument('--workers', type=int, default=24)
    a = ap.parse_args()
    t0 = time.time()
    ds_sets = [(k, C.load_labels(p), os.path.join(C.MV2, 'results', 'gmap_rest_{h}.npy')) for (k, p) in DS.items()]
    hcp = [(nm, lab, os.path.join(C.HREP, f'gmap_{s}_' + '{h}.npy')) for (nm, lab, s) in hcp_sets()]
    out = []
    verts = []
    nulls = {}
    for (dataset, sets) in (('ds005498', ds_sets), ('HCP', hcp)):
        (r, v, n) = run(dataset, sets, a.n_spin, a.workers)
        out.append(r)
        verts.append(v)
        nulls.update({f'{dataset}__{k}': x for (k, x) in n.items()})
        log(dataset, '\n' + r[['atlas', 'precision', 'recall', 'F1', 'F1_z', 'dist_mm', 'dist_z']].to_string(index=False))
    pd.concat(out).to_csv(os.path.join(C.RES, 'boundary_f1_distance.csv'), index=False)
    pd.concat(verts).to_csv(os.path.join(C.SCR, 'boundary_vertex_distances.csv.gz'), index=False)
    np.savez(os.path.join(C.RES, 'boundary_f1_distance_null.npz'), **{k.replace('|', '_').replace(' ', '_').replace('/', '_'): v for (k, v) in nulls.items()})
    log(f'F1_END {time.time() - t0:.0f}s')
if __name__ == '__main__':
    main()
