import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
from pfbn import settings
import os, sys, json, time
sys.path[0:0] = [settings.code('evaluation', 'boundary'), settings.code('visualization')]
from boundary_tools import *
import pandas as pd
from scipy.sparse.csgraph import dijkstra
from concurrent.futures import ProcessPoolExecutor
E = settings.out('r1_figure')
RES = os.path.join(E, 'results')
RDR = os.path.join(E, 'figs', 'renders')
ABLATION = settings.out('dynamics_ablation')
for d in (RES, RDR):
    os.makedirs(d, exist_ok=True)
GROUPS = [('children', 'life_children'), ('adolescents', 'life_adolescents'), ('young_adults', 'life_young_adults'), ('older_adults', 'life_older_adults'), ('NC', 'pd_NC'), ('PD', 'pd_PD')]
CATO = os.path.join(C.SIX, 'assets', 'lausanne250_labels.npz')
V5 = 'results/labels_v5_stim_t03_r30_clean.npy'
WO = 'results/labels_v4_stim_t03_c1_clean.npy'

def three(coh):
    return {'CATO': C.load_labels(CATO), 'Micapipe': C.load_labels(C.MICAPIPE), 'PFBN v5': C.load_labels(os.path.join(ATL, f'{coh}_A', V5))}

def pct_map(coh, gpath=None):
    gm = gmap_cortex(gpath or os.path.join(ATL, f'{coh}_B', 'results', 'gmap_{h}.npy'))
    allg = np.sort(np.concatenate([gm[h][np.isfinite(gm[h])] for h in C.HEMIS]))
    pct = {}
    for h in C.HEMIS:
        x = np.full(C.HN[h], np.nan)
        ok = np.isfinite(gm[h])
        x[ok] = np.searchsorted(allg, gm[h][ok]) / len(allg) * 100
        pct[h] = x
    return pct

def render_job(a):
    (g, coh) = a
    os.environ['OMP_NUM_THREADS'] = '2'
    pct = pct_map(coh)
    for (nm, lab) in three(coh).items():
        out = os.path.join(RDR, f"grad_{g}_{nm.replace(' ', '_')}.png")
        if os.path.exists(out):
            continue
        L_ = {h: thin_border(C.hemi_lab(lab, h), C.edges(h)) for h in C.HEMIS}
        render_layers(out, [(L_, (0, 0, 0))], background=pct, vmin=0, vmax=100)
    log('render', g)
    return g

def hnull_job(a):
    (g, coh, h) = a
    os.environ['OMP_NUM_THREADS'] = '2'
    A = C.adjacency(h)
    e = C.edges(h)
    cidx = C.cortex_idx(h)
    la = C.hemi_lab(C.load_labels(os.path.join(ATL, f'{coh}_A', V5)), h)
    lb_full_lab = C.load_labels(os.path.join(ATL, f'{coh}_B', V5))
    B1 = thin_border(la, e)
    D1 = C.dilate(B1, A, K_TOL)

    def counts(lb_h):
        B2 = thin_border(lb_h, e)
        return (int((B1 & C.dilate(B2, A, K_TOL)).sum()), int(B1.sum()), int(B2.sum()))
    obs = counts(C.hemi_lab(lb_full_lab, h))
    full = full_hemi(lb_full_lab, h)
    idx = spins(h, 1000)
    null = [counts(full[idx[s]][cidx]) for s in range(len(idx))]
    log('hnull', g, h)
    return (g, h, obs, null)

def fnull_job(a):
    (g, coh, h) = a
    os.environ['OMP_NUM_THREADS'] = '2'
    G = C.mesh_graph(h)
    A = C.adjacency(h)
    e = C.edges(h)
    cidx = C.cortex_idx(h)
    (gm, nz) = load_gmap(os.path.join(ATL, f'{coh}_B', 'results', 'gmap_{h}.npy'), h)
    ridge = nz & (gm >= np.percentile(gm[nz], THR))
    d_r = dijkstra(G, directed=False, indices=np.flatnonzero(ridge), min_only=True)
    idx = spins(h, 1000)
    r = os.path.join(ABLATION, 'atlases', f'{coh}_A')
    atl = {'Micapipe': C.load_labels(C.MICAPIPE), 'Without dynamics': C.load_labels(os.path.join(r, WO)), 'With dynamics': C.load_labels(os.path.join(r, V5))}
    out = {}
    for (nm, lab) in atl.items():
        lab_h = C.hemi_lab(lab, h)
        full = full_hemi(lab, h)
        o = ridge_stats(lab_h, e, A, ridge, d_r, np.ones(C.HN[h], bool))[0]
        obs = (float(o['dsum']), int(o['nB']))
        nl = []
        for s in range(len(idx)):
            Bs = C.boundary_mask(full[idx[s]][cidx], e)
            nl.append((float(d_r[Bs].sum()), int(Bs.sum())))
        out[nm] = (obs, nl)
    log('fnull', g, h)
    return (g, h, out)

def main_render():
    with ProcessPoolExecutor(6) as ex:
        list(ex.map(render_job, GROUPS))

def main_hnull():
    with ProcessPoolExecutor(12) as ex:
        res = list(ex.map(hnull_job, [(g, coh, h) for (g, coh) in GROUPS for h in C.HEMIS]))
    by = {}
    for (g, h, obs, null) in res:
        by.setdefault(g, {})[h] = (obs, null)
    ref = pd.read_csv(settings.out('reference_figs', 'R1_age_disease', 'results', 'border_agreement.csv')).set_index('tag')
    (rows, nulls) = ([], {})
    agr = lambda s, b1, b2: 2 * s / (b1 + b2)
    for (g, _) in GROUPS:
        o = [sum((by[g][h][0][k] for h in C.HEMIS)) for k in range(3)]
        ob = agr(*o)
        nl = np.array([agr(*[by[g]['lh'][1][s][k] + by[g]['rh'][1][s][k] for k in range(3)]) for s in range(len(by[g]['lh'][1]))])
        tag = f"{('age' if g not in ('NC', 'PD') else 'disease')}_{g}_v5"
        rows.append(dict(group=g, agreement=ob, reference_agreement=float(ref.loc[tag, 'halfA_vs_halfB_border_agreement']), null_mean=nl.mean(), null_sd=nl.std(ddof=1), z=(ob - nl.mean()) / nl.std(ddof=1), p_one_sided=(1 + np.sum(nl >= ob)) / (len(nl) + 1)))
        nulls[g] = nl
    pd.DataFrame(rows).to_csv(os.path.join(RES, 'halves_agreement_spin.csv'), index=False)
    np.savez(os.path.join(RES, 'halves_agreement_nulls.npz'), **nulls)
    log('\n' + pd.DataFrame(rows).to_string(index=False))

def ablation_ready():
    return all((os.path.exists(os.path.join(ABLATION, 'atlases', f'{coh}_A', p)) for (_, coh) in GROUPS for p in (V5, WO)))

def main_fnull():
    if not ablation_ready():
        log('fnull: ablation atlases incomplete, skipped')
        return
    with ProcessPoolExecutor(12) as ex:
        res = list(ex.map(fnull_job, [(g, coh, h) for (g, coh) in GROUPS for h in C.HEMIS]))
    by = {}
    for (g, h, out) in res:
        by.setdefault(g, {})[h] = out
    (rows, nulls) = ([], {})
    for (g, _) in GROUPS:
        for nm in by[g]['lh']:
            ((ol, nl_l), (orr, nl_r)) = (by[g]['lh'][nm], by[g]['rh'][nm])
            ob = (ol[0] + orr[0]) / (ol[1] + orr[1])
            nl = np.array([(a[0] + b[0]) / (a[1] + b[1]) for (a, b) in zip(nl_l, nl_r)])
            rows.append(dict(group=g, atlas=nm, dist_mm=ob, null_mean=nl.mean(), z=(ob - nl.mean()) / nl.std(ddof=1)))
            nulls[f'{g}|{nm}'] = nl
    pd.DataFrame(rows).to_csv(os.path.join(RES, 'dyn_distance_raw.csv'), index=False)
    np.savez(os.path.join(RES, 'dyn_distance_raw_nulls.npz'), **nulls)
    log('\n' + pd.DataFrame(rows).to_string(index=False))

def main_drender():
    from gradient_tools import roi_mask, mmp_names, ROIS
    (g, coh) = ('adolescents', 'life_adolescents')
    atl = {'without': C.load_labels(os.path.join(ABLATION, 'atlases', f'{coh}_A', WO)), 'with': C.load_labels(os.path.join(ATL, f'{coh}_A', V5))}
    pct = pct_map(coh)
    names = mmp_names()
    rows = []
    for (nm, lab) in atl.items():
        out = os.path.join(RDR, f'grad_{g}_{nm}_dynamics.png')
        if not os.path.exists(out):
            render_layers(out, [({h: thin_border(C.hemi_lab(lab, h), C.edges(h)) for h in C.HEMIS}, (0, 0, 0))], background=pct, vmin=0, vmax=100)
        acc = {'Somatomotor': [0.0, 0], 'Whole cortex': [0.0, 0]}
        for h in C.HEMIS:
            G = C.mesh_graph(h)
            A = C.adjacency(h)
            e = C.edges(h)
            (gm, nz) = load_gmap(os.path.join(ATL, f'{coh}_B', 'results', 'gmap_{h}.npy'), h)
            ridge = nz & (gm >= np.percentile(gm[nz], THR))
            d_r = dijkstra(G, directed=False, indices=np.flatnonzero(ridge), min_only=True)
            for (r, m) in (('Somatomotor', roi_mask(names[h], ROIS['Somatomotor'], A, 3)), ('Whole cortex', np.ones(C.HN[h], bool))):
                o = ridge_stats(C.hemi_lab(lab, h), e, A, ridge, d_r, m)[0]
                acc[r][0] += o['dsum']
                acc[r][1] += o['nB']
        for (r, (ds, nb)) in acc.items():
            rows.append(dict(group=g, atlas=f'{nm.capitalize()} dynamics', roi=r, dist_mm=ds / nb))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, 'panel_d_distance.csv'), index=False)
    log('\n' + df.to_string(index=False))

def main_brender(groups=('adolescents', 'older_adults')):
    for (g, coh) in [x for x in GROUPS if x[0] in groups]:
        out = os.path.join(RDR, f'grad_{g}_noborder.png')
        if not os.path.exists(out):
            render_layers(out, [], background=pct_map(coh), vmin=0, vmax=100)
        log('brender', g)

def main_custom(tag, half_a_labels, gmap_b_pattern):
    from gradient_tools import roi_mask, mmp_names, ROIS
    atl = {'CATO': C.load_labels(CATO), 'Micapipe': C.load_labels(C.MICAPIPE), 'PFBN v5': C.load_labels(half_a_labels)}
    pct = pct_map(None, gmap_b_pattern)
    out = os.path.join(RDR, f'grad_{tag}_noborder.png')
    if not os.path.exists(out):
        render_layers(out, [], background=pct, vmin=0, vmax=100)
    names = mmp_names()
    rows = []
    for (nm, lab) in atl.items():
        out = os.path.join(RDR, f"grad_{tag}_{nm.replace(' ', '_')}.png")
        if not os.path.exists(out):
            render_layers(out, [({h: thin_border(C.hemi_lab(lab, h), C.edges(h)) for h in C.HEMIS}, (0, 0, 0))], background=pct, vmin=0, vmax=100)
        WINS_ = ('Somatomotor', 'Dorsolateral PFC', 'Early visual')
        acc = {r: [0.0, 0] for r in WINS_ + ('Whole cortex',)}
        for h in C.HEMIS:
            G = C.mesh_graph(h)
            A = C.adjacency(h)
            e = C.edges(h)
            (gm, nz) = load_gmap(gmap_b_pattern, h)
            ridge = nz & (gm >= np.percentile(gm[nz], THR))
            d_r = dijkstra(G, directed=False, indices=np.flatnonzero(ridge), min_only=True)
            for (r, m) in [(w, roi_mask(names[h], ROIS[w], A, 3)) for w in WINS_] + [('Whole cortex', np.ones(C.HN[h], bool))]:
                o = ridge_stats(C.hemi_lab(lab, h), e, A, ridge, d_r, m)[0]
                acc[r][0] += o['dsum']
                acc[r][1] += o['nB']
        for (r, (ds, nb)) in acc.items():
            rows.append(dict(group=tag, roi=r, comparison=nm, dist_mm=ds / nb))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, f'custom_{tag}_distance.csv'), index=False)
    log('\n' + df.to_string(index=False))
if __name__ == '__main__':
    st = sys.argv[1] if len(sys.argv) > 1 else 'all'
    t0 = time.time()
    if st in ('render', 'all'):
        main_render()
    if st in ('hnull', 'all'):
        main_hnull()
    if st in ('fnull', 'all'):
        main_fnull()
    if st in ('drender', 'all'):
        main_drender()
    if st in ('brender', 'all'):
        main_brender(tuple(sys.argv[2:]) or ('adolescents', 'older_adults'))
    if st == 'custom':
        main_custom(sys.argv[2], sys.argv[3], sys.argv[4])
    if st == 'hcp':
        main_custom('HCP', os.path.join(C.HREP, 'labels_v5_stim_t03_r30_REST1_clean.npy'), os.path.join(C.HREP, 'gmap_REST2_{h}.npy'))
    log(f'r1 figure data {st} done {time.time() - t0:.0f}s')
