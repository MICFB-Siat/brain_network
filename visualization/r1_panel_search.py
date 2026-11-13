import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
from pfbn import settings
import os, sys, json, glob
sys.path[0:0] = [settings.code('evaluation', 'boundary'), settings.code('visualization')]
from boundary_tools import *
from gradient_tools import roi_mask, mmp_names, ROIS, pad_box
import nibabel as nib, pandas as pd
from scipy.sparse.csgraph import dijkstra
E = settings.out('r1_figure')
RES = os.path.join(E, 'results')
RDR = os.path.join(E, 'figs', 'renders')
D2 = os.path.join(E, 'figs', 'panel_a_v2')
CATO = os.path.join(C.SIX, 'assets', 'lausanne250_labels.npz')
H = 'lh'
STEP = 500
ATLS = {'CATO': lambda : C.load_labels(CATO), 'Micapipe': lambda : C.load_labels(C.MICAPIPE), 'PFBN v5': lambda : C.load_labels(os.path.join(ATL, 'pd_PD_A', 'results', 'labels_v5_stim_t03_r30_clean.npy'))}
GPATH = os.path.join(ATL, 'pd_PD_B', 'results', 'gmap_{h}.npy')

def normals_x():
    import wbrender as W_
    pk = os.path.dirname(W_.SURF)
    c = sorted(glob.glob(os.path.join(pk, '*L.very_inflated*32k*.surf.gii'))) or sorted(glob.glob(os.path.join(pk, '*L.inflated*32k*.surf.gii')))
    src = c[0] if c else W_.SURF.format('L')
    g = nib.load(src)
    v = g.darrays[0].data.astype(float)
    f = g.darrays[1].data.astype(int)
    n = np.zeros_like(v)
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    for k in range(3):
        np.add.at(n, f[:, k], fn)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    return (n[C.cortex_idx(H), 0], src)

def setup():
    G = C.mesh_graph(H)
    A = C.adjacency(H)
    e = C.edges(H)
    (gm, nz) = load_gmap(GPATH, H)
    ridge = nz & (gm >= np.percentile(gm[nz], THR))
    d_r = dijkstra(G, directed=False, indices=np.flatnonzero(ridge), min_only=True)
    K = int(roi_mask(mmp_names()[H], ROIS['Somatomotor'], A, 3).sum())
    atl = {k: f() for (k, f) in ATLS.items()}
    B = {k: C.boundary_mask(C.hemi_lab(v, H), e) for (k, v) in atl.items()}
    return (G, A, e, ridge, d_r, K, atl, B)

def window(G, c, K):
    d = dijkstra(G, directed=False, indices=[c], limit=80.0)[0]
    o = np.argsort(d)[:K]
    m = np.zeros(G.shape[0], bool)
    m[o[np.isfinite(d[o])]] = True
    return m

def main_search():
    (G, A, e, ridge, d_r, K, atl, B) = setup()
    (nx, src) = normals_x()
    rows = []
    for c in range(0, G.shape[0], STEP):
        m = window(G, c, K)
        (lat, med) = (np.mean(nx[m] < -0.2), np.mean(nx[m] > 0.2))
        view = 'lateral' if lat >= 0.95 else 'medial' if med >= 0.95 else None
        if view is None:
            continue
        nR = int((ridge & m).sum())
        r = dict(centre=c, view=view, n_vertices=int(m.sum()), n_ridge=nR)
        for k in atl:
            r[f'nB_{k}'] = int((B[k] & m).sum())
            r[f'dist_{k}'] = float(d_r[B[k] & m].mean()) if r[f'nB_{k}'] else np.nan
        rows.append(r)
    df = pd.DataFrame(rows)
    df['score'] = df['dist_Micapipe'] - df['dist_PFBN v5']
    ok = (df.n_ridge >= 30) & (df[[f'nB_{k}' for k in atl]] >= 30).all(1) & (df['dist_PFBN v5'] < df['dist_CATO'])
    df['eligible'] = ok
    df = df.sort_values(['eligible', 'score'], ascending=False).reset_index(drop=True)
    df['rank'] = np.where(df.eligible, np.arange(1, len(df) + 1), 0)
    df.to_csv(os.path.join(RES, 'pd_window_search.csv'), index=False)
    log(f'search: K = {K}, surface {os.path.basename(src)}, {len(df)} visible windows, {int(ok.sum())} eligible\n' + df.head(10).to_string(index=False))

def mask_job(a):
    (rank, c) = a
    os.environ['OMP_NUM_THREADS'] = '2'
    out = os.path.join(RDR, f'pdwin_{rank}.png')
    if not os.path.exists(out):
        G = C.mesh_graph(H)
        A = C.adjacency(H)
        K = int(roi_mask(mmp_names()[H], ROIS['Somatomotor'], A, 3).sum())
        m = window(G, c, K)
        render_layers(out, [({'lh': m, 'rh': np.zeros(C.HN['rh'], bool)}, (255, 0, 0))])
    return rank

def box(rank, shape, view):
    import matplotlib.pyplot as plt
    im = plt.imread(os.path.join(RDR, f'pdwin_{rank}.png'))[..., :3]
    m = (im[..., 0] > 0.8) & (im[..., 1] < 0.25) & (im[..., 2] < 0.25)
    (H_, W_) = m.shape
    q = np.zeros_like(m)
    q[slice(0, H_ // 2) if view == 'lateral' else slice(H_ // 2, H_), :W_ // 2] = True
    m &= q
    (rr, cc) = np.where(m)
    return pad_box((rr.min(), rr.max(), cc.min(), cc.max()), shape, 0.18)

def main_masks(n=6):
    from concurrent.futures import ProcessPoolExecutor
    df = pd.read_csv(os.path.join(RES, 'pd_window_search.csv'))
    top = df[df.eligible].head(n)
    with ProcessPoolExecutor(6) as ex:
        list(ex.map(mask_job, [(int(r['rank']), int(r.centre)) for (_, r) in top.iterrows()]))

def spin(rank):
    (G, A, e, ridge, d_r, K, atl, B) = setup()
    df = pd.read_csv(os.path.join(RES, 'pd_window_search.csv')).set_index('rank')
    c = int(df.loc[rank, 'centre'])
    m = window(G, c, K)
    idx = spins(H, 1000)
    cidx = C.cortex_idx(H)
    null = {}
    obs = {k: float(d_r[B[k] & m].mean()) for k in atl}
    for (k, lab) in atl.items():
        full = full_hemi(lab, H)
        v = []
        for s in range(len(idx)):
            Bs = C.boundary_mask(full[idx[s]][cidx], e) & m
            v.append(d_r[Bs].mean() if Bs.any() else np.nan)
        null[k] = np.array(v)
    out = {'rank': rank, 'centre_vertex_cortex_index': c, 'view': df.loc[rank, 'view'], 'dist_mm': obs}
    for ref in ('Micapipe', 'CATO'):
        dn = null['PFBN v5'] - null[ref]
        dn = dn[np.isfinite(dn)]
        do = obs['PFBN v5'] - obs[ref]
        out[f'p_spin_one_sided_vs_{ref}'] = float((1 + np.sum(dn <= do)) / (len(dn) + 1))
        out[f'diff_vs_{ref}'] = do
        out[f'n_spins_valid_vs_{ref}'] = int(len(dn))
    out['note'] = 'window chosen by an objective search maximising Micapipe - Ours: post-selection P, descriptive only'
    json.dump(out, open(os.path.join(RES, 'pd_window_best.json'), 'w'), indent=1)
    log(json.dumps(out, indent=1))
    return out
if __name__ == '__main__':
    st = sys.argv[1]
    if st == 'search':
        main_search()
    elif st == 'masks':
        main_masks()
    elif st == 'spin':
        spin(int(sys.argv[2]))
