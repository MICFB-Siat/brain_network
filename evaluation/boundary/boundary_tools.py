import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
from pfbn import settings
import os, sys, json, time, hashlib
ROOT = settings.out()
BOUNDARY_CODE = settings.code('evaluation', 'boundary')
if BOUNDARY_CODE not in sys.path:
    sys.path.insert(0, BOUNDARY_CODE)
from gradient_tools import *
EXP = os.path.join(ROOT, 'reference_figs')
ATL5M = os.path.join(ROOT, 'atlases_masked')
TMP35 = settings.tmp('reference_figs')
os.makedirs(TMP35, exist_ok=True)
V5 = 'results/labels_v5_stim_t03_r30_clean.npy'
V5M = 'results/labels_v5m_stim_t03_r30_clean.npy'
WO = 'results/labels_v4_stim_t03_c1_clean.npy'
RGB = {'shared': (150, 150, 150), 'ours': (31, 95, 220), 'mica': (0, 0, 0), 'A': (209, 73, 91), 'B': (31, 95, 220)}

def atlas_v5(cohort, kind='v5'):
    return C.load_labels(os.path.join(ATL if kind == 'v5' else ATL5M, cohort, V5 if kind == 'v5' else V5M))

def micapipe():
    return C.load_labels(C.MICAPIPE)

def thin_border(lab_h, e):
    (a, b) = (lab_h[e[:, 0]], lab_h[e[:, 1]])
    d = (a != b) & (a > 0) & (b > 0)
    m = np.zeros(len(lab_h), bool)
    big = np.where(a > b, e[:, 0], e[:, 1])
    m[big[d]] = True
    return m

def border_categories(lab1_h, lab2_h, e, A, k=K_TOL):
    (B1, B2) = (thin_border(lab1_h, e), thin_border(lab2_h, e))
    (D1, D2) = (C.dilate(B1, A, k), C.dilate(B2, A, k))
    return {'shared': B1 & D2, 'shared_any': B1 & D2 | B2 & D1, 'only1': B1 & ~D2, 'only2': B2 & ~D1, 'B1': B1, 'B2': B2}

def render_layers(out, layers, background=None, cmap='YlOrRd', vmin=None, vmax=None, n_levels=128, size=(4800, 3000), base_rgb=(225, 225, 225)):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.cm as cm
    arr = {}
    colors = {}
    if background is not None:
        allv = np.concatenate([background[h][np.isfinite(background[h])] for h in C.HEMIS])
        vmin = np.percentile(allv, 2) if vmin is None else vmin
        vmax = np.percentile(allv, 98) if vmax is None else vmax
        cm_ = cm.get_cmap(cmap, n_levels)
        colors.update({k: cm_(k - 1)[:3] for k in range(1, n_levels + 1)})
    else:
        colors[1] = tuple(np.asarray(base_rgb) / 255)
    for h in C.HEMIS:
        x = np.zeros(C.HN[h], int)
        if background is None:
            x[:] = 1
        else:
            g = background[h]
            ok = np.isfinite(g)
            x[ok] = 1 + np.clip(((g[ok] - vmin) / (vmax - vmin) * n_levels).astype(int), 0, n_levels - 1)
        for (j, (m, rgb)) in enumerate(layers):
            key = n_levels + 10 + j
            x[m[h] & (x > 0)] = key
            colors[key] = tuple(np.asarray(rgb) / 255)
        arr[h] = to32k(x, h, 0).astype(int)
    W.labels_png(arr['lh'], arr['rh'], out, colors=colors, borders=False, size=size)
    return (vmin, vmax)

def render_parcels_with_ref(out, ours, ref, seed=3, size=(2400, 1500)):
    keys = np.unique(ref[ref > 0])
    pc = W.parcel_colors(len(keys), seed)
    rcol = dict(zip(keys.tolist(), pc))
    colors = {}
    arr = {}
    for k in np.unique(ours[ours > 0]):
        r = ref[(ours == k) & (ref > 0)]
        colors[int(k)] = rcol[int(np.bincount(r).argmax())] if len(r) else (0.8, 0.8, 0.8)
    colors[901] = (1, 1, 1)
    colors[902] = (0, 0, 0)
    for h in C.HEMIS:
        e = C.edges(h)
        (o, r) = (C.hemi_lab(ours, h).copy(), C.hemi_lab(ref, h))
        x = o.copy()
        x[thin_border(o, e)] = 901
        x[thin_border(r, e)] = 902
        arr[h] = to32k(x, h, 0).astype(int)
    W.labels_png(arr['lh'], arr['rh'], out, colors=colors, borders=False, size=size)

def gmap_cortex(path):
    out = {}
    for h in C.HEMIS:
        (g, nz) = load_gmap(path, h)
        out[h] = np.where(nz, g, np.nan)
    return out

def window_f1(atlases, gpath, rois, nspin=1000):
    res = {}
    names = list(atlases)
    for h in C.HEMIS:
        A = C.adjacency(h)
        e = C.edges(h)
        cidx = C.cortex_idx(h)
        (g, nz) = load_gmap(gpath, h)
        ridge = nz & (g >= np.percentile(g[nz], THR))
        masks = {r: m[h] for (r, m) in rois.items()}
        masks['Whole cortex'] = np.ones(C.HN[h], bool)
        Rd = {r: C.dilate(ridge & m, A, K_TOL) for (r, m) in masks.items()}
        Rn = {r: ridge & m for (r, m) in masks.items()}
        idx = spins(h, nspin)
        for nm in names:
            lab_h = C.hemi_lab(atlases[nm], h)
            full = full_hemi(atlases[nm], h)

            def counts(Bfull):
                o = {}
                for (r, m) in masks.items():
                    B = Bfull & m
                    o[r] = np.array([B.sum(), (B & Rd[r]).sum(), Rn[r].sum(), (Rn[r] & C.dilate(B, A, K_TOL)).sum()], float)
                return o
            obs = counts(C.boundary_mask(lab_h, e))
            nul = [counts(C.boundary_mask(full[idx[s]][cidx], e)) for s in range(len(idx))]
            for r in masks:
                key = (nm, r)
                (o, n_) = (obs[r], np.stack([x[r] for x in nul]))
                if key in res:
                    res[key][0] += o
                    res[key][1] += n_
                else:
                    res[key] = [o.copy(), n_.copy()]

    def f1v(c):
        c = np.atleast_2d(c)
        P_ = c[:, 1] / np.maximum(c[:, 0], 1)
        R_ = c[:, 3] / np.maximum(c[:, 2], 1)
        return np.where(P_ + R_ > 0, 2 * P_ * R_ / np.maximum(P_ + R_, 1e-12), 0)
    rows = []
    for r in list(rois) + ['Whole cortex']:
        f = {nm: (f1v(res[nm, r][0])[0], f1v(res[nm, r][1])) for nm in names}
        for nm in names:
            (o, n_) = f[nm]
            rows.append(dict(roi=r, atlas=nm, F1=o, F1_z=(o - n_.mean()) / n_.std(), p_spin=(1 + (n_ >= o).sum()) / (1 + len(n_))))
        if len(names) == 2:
            (a, b) = names
            d_obs = f[a][0] - f[b][0]
            d_null = f[a][1] - f[b][1]
            rows.append(dict(roi=r, atlas=f'{a} - {b}', F1=d_obs, F1_z=(d_obs - d_null.mean()) / d_null.std(), p_spin=(1 + (np.abs(d_null - d_null.mean()) >= abs(d_obs - d_null.mean())).sum()) / (1 + len(d_null))))
    return rows

def roi_masks(names, rset, dil=3):
    return {r: {h: roi_mask(names[h], ROIS[r], C.adjacency(h), dil) for h in C.HEMIS} for r in rset}

def sha_list(paths, out):
    write_sha(paths, out)
