import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
from pfbn import settings
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, settings.code('evaluation', 'boundary'))
from gradient_tools import *
from scipy.sparse.csgraph import dijkstra
from concurrent.futures import ProcessPoolExecutor
EXP = os.path.join(ROOT, 'boundary_distance')
RES = os.path.join(EXP, 'results')
os.makedirs(RES, exist_ok=True)
GROUPS = [('age', 'children', 'life_children'), ('age', 'adolescents', 'life_adolescents'), ('age', 'young_adults', 'life_young_adults'), ('age', 'older_adults', 'life_older_adults'), ('disease', 'NC', 'pd_NC'), ('disease', 'PD', 'pd_PD')]
WIN = ['Somatomotor', 'Dorsolateral PFC']
CATO = os.path.join(C.SIX, 'assets', 'lausanne250_labels.npz')
NSPIN = int(os.environ.get('NSPIN', 1000))

def atlases(coh):
    return {'Micapipe': C.load_labels(C.MICAPIPE), 'CATO': C.load_labels(CATO), 'PFBN v5': C.load_labels(os.path.join(ATL, f'{coh}_A', 'results', 'labels_v5_stim_t03_r30_clean.npy'))}

def job(args):
    (g, coh, h) = args
    os.environ['OMP_NUM_THREADS'] = '2'
    names = mmp_names()
    A = C.adjacency(h)
    e = C.edges(h)
    cidx = C.cortex_idx(h)
    G = C.mesh_graph(h)
    rois = {r: roi_mask(names[h], ROIS[r], A, 3) for r in WIN}
    rois['Whole cortex'] = np.ones(C.HN[h], bool)
    (gm, nz) = load_gmap(os.path.join(ATL, f'{coh}_B', 'results', 'gmap_{h}.npy'), h)
    ridge = nz & (gm >= np.percentile(gm[nz], THR))
    d_r = dijkstra(G, directed=False, indices=np.flatnonzero(ridge), min_only=True)
    idx = spins(h, NSPIN)
    atl = atlases(coh)
    (obs, null) = ({}, {})
    Rdil = {r: C.dilate(ridge & m, A, K_TOL) for (r, m) in rois.items()}
    for (nm, lab) in atl.items():
        lab_h = C.hemi_lab(lab, h)
        full = full_hemi(lab, h)
        obs[nm] = {r: ridge_stats(lab_h, e, A, ridge, d_r, m)[0] for (r, m) in rois.items()}
        nl = {r: [] for r in rois}
        for s in range(len(idx)):
            Bs = C.boundary_mask(full[idx[s]][cidx], e)
            for (r, m) in rois.items():
                B = Bs & m
                Rr = ridge & m
                nl[r].append(dict(nB=int(B.sum()), tpB=int((B & Rdil[r]).sum()), nR=int(Rr.sum()), tpR=int((Rr & C.dilate(B, A, K_TOL)).sum()), dsum=float(d_r[B].sum())))
        null[nm] = nl
    log(f'{g} {h} done')
    return (g, h, obs, null)

def main():
    jobs = [(g, coh, h) for (_, g, coh) in GROUPS for h in C.HEMIS]
    with ProcessPoolExecutor(int(os.environ.get('NPROC', 12))) as ex:
        res = list(ex.map(job, jobs))
    by = {}
    for (g, h, obs, null) in res:
        by.setdefault(g, {})[h] = (obs, null)
    (rows, nulls) = ([], {})
    for (study, g, coh) in GROUPS:
        d = by[g]
        vals = {}
        for nm in d['lh'][0]:
            for r in d['lh'][0][nm]:
                o = pool(d['lh'][0][nm][r], d['rh'][0][nm][r])
                nl = [pool(a, b) for (a, b) in zip(d['lh'][1][nm][r], d['rh'][1][nm][r])]
                F = f1(o)[2]
                D_ = o['dsum'] / max(o['nB'], 1)
                nf = np.array([f1(x)[2] for x in nl])
                nd = np.array([x['dsum'] / max(x['nB'], 1) for x in nl])
                vals[nm, r] = (F, D_, nf, nd)
                rows.append(dict(study=study, group=g, roi=r, comparison=nm, F1=F, F1_z=(F - nf.mean()) / nf.std(ddof=1), dist_mm=D_, dist_z=(D_ - nd.mean()) / nd.std(ddof=1), p_F1=(1 + (nf >= F).sum()) / (len(nf) + 1), p_dist=(1 + (nd <= D_).sum()) / (len(nd) + 1)))
                nulls[f'{g}|{r}|{nm}|F1'] = nf
                nulls[f'{g}|{r}|{nm}|dist'] = nd
        for ref in ('Micapipe', 'CATO'):
            for r in ['Whole cortex'] + WIN:
                (F5, D5, nf5, nd5) = vals['PFBN v5', r]
                (Fr, Dr, nfr, ndr) = vals[ref, r]
                (dF, dD, nF, nD) = (F5 - Fr, D5 - Dr, nf5 - nfr, nd5 - ndr)
                rows.append(dict(study=study, group=g, roi=r, comparison=f'PFBN v5 - {ref}', F1=dF, dist_mm=dD, F1_z=(dF - nF.mean()) / nF.std(ddof=1), dist_z=(dD - nD.mean()) / nD.std(ddof=1), p_F1=(1 + (nF >= dF).sum()) / (len(nF) + 1), p_dist=(1 + (nD <= dD).sum()) / (len(nD) + 1), p_F1_two_sided_centred=(1 + (np.abs(nF - nF.mean()) >= abs(dF - nF.mean())).sum()) / (1 + len(nF)), p_dist_two_sided_centred=(1 + (np.abs(nD - nD.mean()) >= abs(dD - nD.mean())).sum()) / (1 + len(nD))))
                nulls[f'{g}|{r}|PFBN v5 - {ref}|F1'] = nF
                nulls[f'{g}|{r}|PFBN v5 - {ref}|dist'] = nD
    pd.DataFrame(rows).to_csv(os.path.join(RES, 'dist_f1_groups.csv'), index=False)
    np.savez(os.path.join(RES, 'dist_f1_nulls.npz'), **nulls)
    write_sha([os.path.abspath(__file__), settings.code('evaluation', 'boundary', 'gradient_tools.py'), settings.code('evaluation', 'boundary', 'eval_common.py'), CATO, C.MICAPIPE, C.SPINS] + [os.path.join(ATL, f'{coh}_{x}', 'results', f) for (_, _, coh) in GROUPS for (x, f) in (('A', 'labels_v5_stim_t03_r30_clean.npy'), ('B', 'gmap_lh.npy'), ('B', 'gmap_rh.npy'))], os.path.join(RES, 'dist_inputs_sha256.txt'))
    log('DIST_END')
if __name__ == '__main__':
    main()
