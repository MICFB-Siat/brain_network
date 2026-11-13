import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
from pfbn import settings
import os, sys, json, time
E = settings.out('dynamics_ablation')
ROOT = settings.out()
for p in (settings.code('evaluation', 'boundary'), settings.code('evaluation', 'dcbc'), settings.code('pfbn', 'pipeline')):
    sys.path.insert(0, p)
from boundary_tools import *
import pandas as pd
from scipy.sparse.csgraph import dijkstra
from scipy.stats import wilcoxon
from sklearn.metrics import adjusted_rand_score
RES = os.path.join(E, 'results')
FIG = os.path.join(E, 'figs')
RDR = os.path.join(FIG, 'renders')
for d in (RES, FIG, RDR):
    os.makedirs(d, exist_ok=True)
RU = settings.out()
GROUPS = [('children', 'life_children'), ('adolescents', 'life_adolescents'), ('young_adults', 'life_young_adults'), ('older_adults', 'life_older_adults'), ('NC', 'pd_NC'), ('PD', 'pd_PD')]
GN = {'children': 'Children', 'adolescents': 'Adolescents', 'young_adults': 'Young adults', 'older_adults': 'Older adults', 'NC': 'NC', 'PD': 'PD'}
NAMES = ['Micapipe', 'Without dynamics', 'With dynamics']
WINS = ['Somatomotor', 'Dorsolateral PFC']

def atlases(coh):
    r = os.path.join(E, 'atlases', f'{coh}_A', 'results')
    return {'Micapipe': C.load_labels(C.MICAPIPE), 'Without dynamics': C.load_labels(os.path.join(r, 'labels_v4_stim_t03_c1_clean.npy')), 'With dynamics': C.load_labels(os.path.join(r, 'labels_v5_stim_t03_r30_clean.npy'))}

def build_info():
    rows = []
    for (g, coh) in GROUPS:
        r = os.path.join(E, 'atlases', f'{coh}_A', 'results')
        isl = json.load(open(os.path.join(r, 'islands.json')))
        new = np.load(os.path.join(r, 'labels_v5_stim_t03_r30_clean.npy'))
        frz = np.load(os.path.join(ATL, f'{coh}_A', 'results', 'labels_v5_stim_t03_r30_clean.npy'))
        rows.append(dict(group=g, ari_rebuilt_with_vs_frozen=adjusted_rand_score(frz, new), identical=bool(np.array_equal(frz, new)), **{f'{m}_{k}': isl[t][k] for (m, t) in (('with', 'v5_stim_t03_r30'), ('without', 'v4_stim_t03_c1')) for k in ('components_before', 'components_after', 'labels_after', 'n_islands', 'changed_vertices')}))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, 'build_info.csv'), index=False)
    log('\n' + df.to_string(index=False))

def main_dcbc():
    import torch, cohort_lib as RL, common as PC, dcbc as D
    pairs = {h: D.HemiPairs(h) for h in ('lh', 'rh')}
    rows = []
    t0 = time.time()
    for (g, coh) in GROUPS:
        atl = {k: v.astype(np.int64) for (k, v) in atlases(coh).items()}
        cj = json.load(open(os.path.join(RU, 'cohorts', f'{coh}_B.json')))
        PC.DENOISE = cj.get('denoise', 'none')
        PC.DROP = int(cj.get('drop', 5))
        for s in cj['subjects']:
            runs = [r if isinstance(r, dict) else {'path': r} for r in s['runs']]
            X = np.concatenate([(lambda x: x - x.mean(0))(PC.load_run(r)[0]) for r in runs], 0)
            Xt = torch.from_numpy(X).to(D.dev)
            res = {}
            for (h, sl) in (('lh', slice(0, RL.NL)), ('rh', slice(RL.NL, RL.NCORT))):
                P_ = pairs[h]
                P_.set_subject(Xt[:, sl])
                for k in NAMES:
                    res.setdefault(k, []).append(P_.evaluate(atl[k][sl])['DCBC_weighted'])
            for (k, v) in res.items():
                rows.append(dict(subject=s['sid'], group=g, atlas=k, DCBC=float(np.mean(v)), frames=int(X.shape[0])))
            del Xt
            torch.cuda.empty_cache()
        log(f"DCBC {g}: {len(cj['subjects'])} subjects, {time.time() - t0:.0f}s")
    dc = pd.DataFrame(rows)
    dc.to_csv(os.path.join(RES, 'dcbc_per_subject.csv'), index=False)
    st = {}
    for (g, _) in GROUPS:
        t = dc[dc.group == g].pivot(index='subject', columns='atlas', values='DCBC')
        st[g] = {'n': len(t), **{f'median|{k}': float(t[k].median()) for k in NAMES}}
        for (a, b) in (('With dynamics', 'Without dynamics'), ('With dynamics', 'Micapipe'), ('Without dynamics', 'Micapipe')):
            st[g][f'{a} vs {b}'] = dict(n_greater=int((t[a] > t[b]).sum()), median_diff=float((t[a] - t[b]).median()), p=float(wilcoxon(t[a], t[b]).pvalue))
    json.dump(st, open(os.path.join(RES, 'dcbc_stats.json'), 'w'), indent=1)
    log(json.dumps(st, indent=1))

def spin_job(a):
    (g, coh, h) = a
    os.environ['OMP_NUM_THREADS'] = '2'
    G = C.mesh_graph(h)
    A = C.adjacency(h)
    e = C.edges(h)
    cidx = C.cortex_idx(h)
    (gm, nz) = load_gmap(os.path.join(ATL, f'{coh}_B', 'results', 'gmap_{h}.npy'), h)
    ridge = nz & (gm >= np.percentile(gm[nz], THR))
    d_r = dijkstra(G, directed=False, indices=np.flatnonzero(ridge), min_only=True)
    names = mmp_names()
    rois = {r: roi_mask(names[h], ROIS[r], A, 3) for r in WINS}
    rois['Whole cortex'] = np.ones(C.HN[h], bool)
    idx = spins(h, 1000)
    atl = atlases(coh)
    out = {}
    for (nm, lab) in atl.items():
        lab_h = C.hemi_lab(lab, h)
        full = full_hemi(lab, h)
        obs = {r: ridge_stats(lab_h, e, A, ridge, d_r, m)[0] for (r, m) in rois.items()}
        null = {r: [] for r in rois}
        for s in range(len(idx)):
            Bs = C.boundary_mask(full[idx[s]][cidx], e)
            for (r, m) in rois.items():
                B = Bs & m
                R_ = ridge & m
                null[r].append(dict(nB=int(B.sum()), tpB=int((B & C.dilate(R_, A, K_TOL)).sum()), nR=int(R_.sum()), tpR=int((R_ & C.dilate(B, A, K_TOL)).sum()), dsum=float(d_r[B].sum())))
        out[nm] = (obs, null)
    log('spin', g, h)
    return (g, h, out)

def main_spin():
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(12) as ex:
        res = list(ex.map(spin_job, [(g, coh, h) for (g, coh) in GROUPS for h in C.HEMIS]))
    by = {}
    for (g, h, out) in res:
        by.setdefault(g, {})[h] = out
    (rows, nulls) = ([], {})
    for (g, _) in GROUPS:
        val = {}
        for nm in NAMES:
            for r in by[g]['lh'][nm][0]:
                o = pool(by[g]['lh'][nm][0][r], by[g]['rh'][nm][0][r])
                nl = [pool(a, b) for (a, b) in zip(by[g]['lh'][nm][1][r], by[g]['rh'][nm][1][r])]
                F = f1(o)[2]
                dist = o['dsum'] / max(o['nB'], 1)
                nf = np.array([f1(x)[2] for x in nl])
                nd = np.array([x['dsum'] / max(x['nB'], 1) for x in nl])
                val[nm, r] = (F, nf, dist, nd)
                rows.append(dict(group=g, roi=r, comparison=nm, F1=F, F1_z=(F - nf.mean()) / nf.std(ddof=1), F1_p_one_sided=(1 + np.sum(nf >= F)) / (len(nf) + 1), dist_mm=dist, dist_z=(dist - nd.mean()) / nd.std(ddof=1), dist_p_one_sided=(1 + np.sum(nd <= dist)) / (len(nd) + 1)))
        for (a, b) in (('With dynamics', 'Without dynamics'), ('With dynamics', 'Micapipe'), ('Without dynamics', 'Micapipe')):
            for r in ['Whole cortex'] + WINS:
                (Fa, nfa, da, nda) = val[a, r]
                (Fb, nfb, db, ndb) = val[b, r]
                (dF, nF) = (Fa - Fb, nfa - nfb)
                (dD, nD) = (da - db, nda - ndb)
                rows.append(dict(group=g, roi=r, comparison=f'{a} - {b}', F1=dF, F1_z=(dF - nF.mean()) / nF.std(ddof=1), F1_p_one_sided=(1 + np.sum(nF >= dF)) / (len(nF) + 1), F1_p_two_sided=(1 + np.sum(np.abs(nF - nF.mean()) >= abs(dF - nF.mean()))) / (len(nF) + 1), dist_mm=dD, dist_z=(dD - nD.mean()) / nD.std(ddof=1), dist_p_one_sided=(1 + np.sum(nD <= dD)) / (len(nD) + 1), dist_p_two_sided=(1 + np.sum(np.abs(nD - nD.mean()) >= abs(dD - nD.mean()))) / (len(nD) + 1)))
                nulls[f'{g}|{r}|{a} - {b}|F1'] = nF
                nulls[f'{g}|{r}|{a} - {b}|dist'] = nD
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, 'spin_f1_distance.csv'), index=False)
    np.savez(os.path.join(RES, 'spin_diff_nulls.npz'), **nulls)
    log('\n' + df[df.comparison.str.contains(' - ')][['group', 'roi', 'comparison', 'F1', 'F1_p_one_sided', 'dist_mm', 'dist_p_one_sided']].to_string(index=False))

def render_job(a):
    (g, coh) = a
    os.environ['OMP_NUM_THREADS'] = '2'
    gm = gmap_cortex(os.path.join(ATL, f'{coh}_B', 'results', 'gmap_{h}.npy'))
    allg = np.sort(np.concatenate([gm[h][np.isfinite(gm[h])] for h in C.HEMIS]))
    pct = {}
    for h in C.HEMIS:
        x = np.full(C.HN[h], np.nan)
        ok = np.isfinite(gm[h])
        x[ok] = np.searchsorted(allg, gm[h][ok]) / len(allg) * 100
        pct[h] = x
    atl = atlases(coh)
    for nm in ('Without dynamics', 'With dynamics'):
        L_ = {h: thin_border(C.hemi_lab(atl[nm], h), C.edges(h)) for h in C.HEMIS}
        render_layers(os.path.join(RDR, f'grad_{g}_{nm.split()[0].lower()}.png'), [(L_, (0, 0, 0))], background=pct, vmin=0, vmax=100)
    log('render', g)
    return g

def main_render():
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(6) as ex:
        list(ex.map(render_job, GROUPS))
if __name__ == '__main__':
    stage = sys.argv[1]
    if stage == 'info':
        build_info()
    elif stage == 'dcbc':
        main_dcbc()
    elif stage == 'spin':
        main_spin()
    elif stage == 'render':
        main_render()
