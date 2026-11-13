import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from common import log
import mrf as M0
import numpy as np, torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
TAG = 'v5m_stim_t03_r30'
BASE = 'v5_stim_t03_r30'
L = M0.L

def signal_mask(hemi, X, rows, zero_frac):
    off = np.concatenate([[0], np.cumsum(rows)])
    cnt = np.zeros(X.shape[1], np.int64)
    for i in range(len(rows)):
        b = np.asarray(X[off[i]:off[i + 1]], dtype=np.float64)
        cnt += (b ** 2).sum(0) < 1e-12
    frac = cnt / len(rows)
    keep = frac <= zero_frac
    return (keep, {'n_vertices': int(len(keep)), 'n_masked': int((~keep).sum()), 'zero_frac_threshold': zero_frac, 'n_vertices_zero_in_any_subject': int((cnt > 0).sum()), 'n_vertices_zero_in_all_subjects': int((cnt == len(rows)).sum()), 'n_subjects': int(len(rows))})

def run(zero_frac):
    cfg = M0.METHODS[BASE]
    rows = json.load(open(os.path.join(C.RESULTS, 'reduce_meta.json')))['rows']
    out = np.zeros(C.NCORT, np.int64)
    rep = {'method': TAG, 'base_method': BASE, 'config': cfg, 'mask': {}}
    t_all = time.time()
    for hemi in C.HEMIS:
        t0 = time.time()
        cidx = C.cortex_idx(hemi)
        X = np.load(os.path.join(C.STACKS, f"X_{cfg['X']}_{hemi}.npy"), mmap_mode='r')
        (keep, minfo) = signal_mask(hemi, X, rows, zero_frac)
        np.save(os.path.join(C.RESULTS, f'signal_mask_{hemi}.npy'), keep)
        kidx = np.where(keep)[0]
        kt = torch.from_numpy(kidx).to(M0.dev)
        Xn = C.unit_cols(torch.from_numpy(np.ascontiguousarray(np.asarray(X)[:, kidx])).to(M0.dev), torch)
        (xyz, _) = C.sphere(hemi)
        xs = xyz[cidx][kidx]
        S = torch.from_numpy((xs / np.linalg.norm(xs, axis=1, keepdims=True)).astype(np.float32)).to(M0.dev)
        lab0 = np.load(os.path.join(C.RESULTS, f'init_{hemi}.npy')).astype(np.int64)[kidx]
        (e, ebase, einfo) = M0.edge_base(hemi)
        newid = -np.ones(len(keep), np.int64)
        newid[kidx] = np.arange(len(kidx))
        me = keep[e[:, 0]] & keep[e[:, 1]]
        e2 = newid[e[me]].astype(np.int32)
        eb2 = ebase[me]
        (PsiG, MPsi, mass, modinfo) = M0.load_modes(hemi, float(cfg['r_s']))
        (PsiG, MPsi, mass) = (PsiG[kt], MPsi[kt], mass[kt])
        (hist, inj) = ([], [])
        t1 = time.time()
        (lab, scal) = M0.solve_v5(Xn, S, PsiG, MPsi, mass, lab0, e2, eb2, cfg['kappa'], cfg['tau'], cfg['c_mult'], hist=hist, inject_log=inj)
        off = 0 if hemi == 'lh' else C.NL
        full = np.zeros(len(keep), np.int64)
        full[kidx] = lab + 1 + (0 if hemi == 'lh' else 100)
        out[off:off + C.HEMI_N[hemi]] = full
        rep['mask'][hemi] = minfo
        rep[hemi] = {'mrf_seconds': round(time.time() - t1, 1), 'iterations': len(hist), 'n_labels_final': int(len(np.unique(lab))), 'converged': bool(hist[-1]['frac_changed'] < M0.STOP and hist[-1]['reinjected'] == 0), 'n_injections': len(inj), 'n_edges_kept': int(me.sum()), 'n_edges_total': int(len(e)), 'edge': einfo, 'modes': modinfo, 'scalars': scal, 'history': hist, 'injections': inj}
        log(f"{TAG} {hemi}: masked {minfo['n_masked']} vertices, {len(hist)} it, labels {rep[hemi]['n_labels_final']}, last change {hist[-1]['frac_changed']:.4f}")
        del Xn, S, PsiG, MPsi, mass
        torch.cuda.empty_cache()
    rep['total_seconds'] = round(time.time() - t_all, 1)
    np.save(os.path.join(C.RESULTS, f'labels_{TAG}.npy'), out.astype(np.int16))
    json.dump(rep, open(os.path.join(C.RESULTS, f'mrf_{TAG}.json'), 'w'), indent=1)
    return out

def comps(lab):
    tot = 0
    off = 0
    for hemi in C.HEMIS:
        n = len(C.cortex_idx(hemi))
        e = C.hemi_edges_cortex(hemi)
        Lh = lab[off:off + n]
        same = (Lh[e[:, 0]] == Lh[e[:, 1]]) & (Lh[e[:, 0]] > 0)
        A = coo_matrix((np.ones(same.sum()), (e[same, 0], e[same, 1])), shape=(n, n))
        (nc, comp) = connected_components(A, directed=False)
        tot += len(np.unique(comp[Lh > 0]))
        off += n
    return tot

def clean_masked(lab):
    out = lab.copy()
    report = []
    off = 0
    for hemi in C.HEMIS:
        n = len(C.cortex_idx(hemi))
        e = C.hemi_edges_cortex(hemi)
        Lh = lab[off:off + n]
        same = Lh[e[:, 0]] == Lh[e[:, 1]]
        A = coo_matrix((np.ones(same.sum()), (e[same, 0], e[same, 1])), shape=(n, n))
        (nc, comp) = connected_components(A, directed=False)
        for lb in np.unique(Lh):
            if lb == 0:
                continue
            (cs, cnt) = np.unique(comp[Lh == lb], return_counts=True)
            if len(cs) > 1:
                keep = cs[np.argmax(cnt)]
                for (c, k) in zip(cs, cnt):
                    if c == keep:
                        continue
                    idx = np.where(comp == c)[0]
                    m = np.isin(e[:, 0], idx) ^ np.isin(e[:, 1], idx)
                    nb = np.concatenate([Lh[e[m, 0]], Lh[e[m, 1]]])
                    nb = nb[(nb != lb) & (nb > 0)]
                    if len(nb) == 0:
                        continue
                    (vals, cn) = np.unique(nb, return_counts=True)
                    new = vals[np.argmax(cn)]
                    out[off + idx] = new
                    report.append(dict(hemi=hemi, label=int(lb), island_size=int(k), main_size=int(cnt.max()), reassigned_to=int(new)))
        off += n
    return (out, report)
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--zero_frac', type=float, default=0.5)
    a = ap.parse_args()
    p = os.path.join(C.RESULTS, f'labels_{TAG}.npy')
    lab = np.load(p).astype(np.int64) if os.path.exists(p) else run(a.zero_frac)
    (cl, isl) = clean_masked(lab)
    np.save(os.path.join(C.RESULTS, f'labels_{TAG}_clean.npy'), cl.astype(np.int16))
    r = {'n_islands': len(isl), 'islands': isl, 'components_before': comps(lab), 'components_after': comps(cl), 'labels_after': int(len(np.unique(cl[cl > 0]))), 'n_unassigned_vertices': int((cl == 0).sum())}
    json.dump(r, open(os.path.join(C.RESULTS, 'islands_v5m.json'), 'w'), indent=1)
    log(f"islands {TAG}: {len(isl)} islands, components {r['components_before']} -> {r['components_after']}, parcels {r['labels_after']}, unassigned {r['n_unassigned_vertices']}")
    log('MRF_V5M_END')
