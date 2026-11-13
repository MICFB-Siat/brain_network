import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gradient_tools import *
from gradient_zoom import hcp_atlases, GM_HCP
EXP = os.path.join(ROOT, 'gradient_zoom')
RD = os.path.join(EXP, 'figs', 'renders')
os.makedirs(RD, exist_ok=True)
SIZE = (4800, 3000)
CMAP = 'YlOrRd'
HUES = {1: 0.0, 2: 0.333, 3: 0.667, 4: 0.1667}

def g32(path, kind):
    out = {}
    for (h, H) in (('lh', 'L'), ('rh', 'R')):
        (g, nz) = load_gmap(path, h) if kind == 'npy' else load_func_cortex(path.format(H=H), h)
        g = np.where(nz, g, np.nan)
        out[H] = to32k(g, h)
    return out

def main():
    only = sys.argv[1:]
    names = mmp_names()
    A = {h: C.adjacency(h) for h in C.HEMIS}
    if not only or 'mask' in only:
        lh = np.full(C.NV, 0, int)
        lh[C.cortex_idx('lh')] = 5
        for (k, (r, ar)) in enumerate(ROIS.items(), 1):
            m = roi_mask(names['lh'], ar)
            v = C.cortex_idx('lh')[m]
            lh[v] = k
        rh = np.zeros(C.NV, int)
        rh[C.cortex_idx('rh')] = 5
        cols = {k: tuple(__import__('colorsys').hsv_to_rgb(HUES[k], 1, 1)) for k in HUES}
        cols[5] = (0.6, 0.6, 0.6)
        W.labels_png(lh, rh, os.path.join(RD, 'roi_mask.png'), colors=cols, borders=False, size=SIZE)
    gm = g32(GM_HCP, 'npy')
    atl = hcp_atlases()
    allg = np.concatenate([gm['L'], gm['R']])
    (vmin, vmax) = (np.nanpercentile(allg, 2), np.nanpercentile(allg, 98))
    rng = {'HCP_gmap_REST2': [float(vmin), float(vmax)]}
    if not only or 'hcp' in only:
        for (nm, lab) in atl.items():
            tag = nm.replace(' ', '_')
            W.scalar_with_borders_png(gm['L'], gm['R'], lab32k(lab, 'lh'), lab32k(lab, 'rh'), os.path.join(RD, f'gmapREST2_{tag}.png'), cmap=CMAP, vmin=vmin, vmax=vmax, size=SIZE)
        none = np.zeros(C.NV, int)
        W.scalar_with_borders_png(gm['L'], gm['R'], none, none, os.path.join(RD, 'gmapREST2_noborders.png'), cmap=CMAP, vmin=vmin, vmax=vmax, size=SIZE)
    if not only or 'tyf' in only:
        can = {}
        for (h, H) in (('lh', 'L'), ('rh', 'R')):
            x = np.zeros(C.HN[h], int)
            n = names[h]
            x[n == 'V1'] = 1
            x[n == 'V2'] = 2
            x[n == '4'] = 3
            x[np.isin(n, ['3a', '3b'])] = 4
            can[H] = to32k(x, h, 0).astype(int)
        for (tag, path, kind) in (('tyf_young_adults_iter8', os.path.join(TYF, 'young_adults_iter8_{H}.func.gii'), 'gii'), ('tyf_young_adults_iter0', os.path.join(TYF, 'young_adults_iter0_{H}.func.gii'), 'gii'), ('HCP_gmap_REST2', GM_HCP, 'npy')):
            g = gm if kind == 'npy' else g32(path, kind)
            a_ = np.concatenate([g['L'], g['R']])
            (lo, hi) = (np.nanpercentile(a_, 2), np.nanpercentile(a_, 98))
            rng[tag] = [float(lo), float(hi)]
            W.scalar_with_borders_png(g['L'], g['R'], can['L'], can['R'], os.path.join(RD, f'canonical_{tag}.png'), cmap='viridis', vmin=lo, vmax=hi, border_rgb=(255, 255, 255), size=(2400, 1500), thin=False)
        g = g32(os.path.join(TYF, 'young_adults_iter8_{H}.func.gii'), 'gii')
        a_ = np.concatenate([g['L'], g['R']])
        (lo, hi) = (np.nanpercentile(a_, 2), np.nanpercentile(a_, 98))
        for (nm, lab) in (('Micapipe', C.load_labels(C.MICAPIPE)), ('PFBN_v5_A', C.load_labels(os.path.join(ATL, 'life_young_adults_A/results/labels_v5_stim_t03_r30_clean.npy')))):
            W.scalar_with_borders_png(g['L'], g['R'], lab32k(lab, 'lh'), lab32k(lab, 'rh'), os.path.join(RD, f'tyf_young_iter8_{nm}.png'), cmap=CMAP, vmin=lo, vmax=hi, size=(2400, 1500))
    json.dump(rng, open(os.path.join(RD, 'color_ranges.json'), 'w'), indent=1)
    log('RENDER_END')
if __name__ == '__main__':
    main()
