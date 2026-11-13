import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
from pfbn import settings
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, settings.code('visualization'))
os.environ.setdefault('WBR_TMP', settings.tmp())
import eval_common as C
import wbrender as W
from boundary_f1 import DS
from hcp_retest import hcp_atlases
from concurrent.futures import ProcessPoolExecutor
OUT = os.path.join(C.FIGS, 'renders')
os.makedirs(OUT, exist_ok=True)

def full(lab):
    return (C.cortex_to_hemi_full(lab, 'lh'), C.cortex_to_hemi_full(lab, 'rh'))

def gmap(path):
    g = [np.load(path.format(h=h)).astype(float) for h in C.HEMIS]
    for x in g:
        x[(x == 0) | ~np.isfinite(x)] = np.nan
    return g

def job(args):
    (kind, name, lab, gpath, vr, out) = args
    if os.path.exists(out):
        return out
    (l, r) = full(lab)
    if kind == 'atlas':
        W.labels_png(l, r, out, seed=7)
    else:
        (gl, gr) = gmap(gpath)
        W.scalar_with_borders_png(gl, gr, l, r, out, cmap='viridis', vmin=vr[0], vmax=vr[1])
    return out

def jobs():
    J = []
    ds = {k: C.load_labels(p) for (k, p) in DS.items() if k != 'v4_filt'}
    g_ds = os.path.join(C.MV2, 'results', 'gmap_rest_{h}.npy')
    gv = np.concatenate([g for g in gmap(g_ds)])
    vr_ds = tuple(np.nanpercentile(gv, [2, 98]))
    for (k, lab) in ds.items():
        tag = k.replace(' ', '_').replace('/', '')
        J.append(('atlas', k, lab, None, None, os.path.join(OUT, f'ds005498_{tag}.png')))
        J.append(('grad', k, lab, g_ds, vr_ds, os.path.join(OUT, f'ds005498_{tag}_on_gradient.png')))
    atl = hcp_atlases()
    g_h = os.path.join(C.HREP, 'gmap_REST2_{h}.npy')
    gv = np.concatenate(gmap(g_h))
    vr_h = tuple(np.nanpercentile(gv, [2, 98]))
    for k in ('Micapipe', 'PFBN v0', 'PFBN v5|REST1', 'PFBN v5|REST2', 'CATO (Lausanne-219)'):
        tag = k.replace(' ', '_').replace('|', '_').replace('(', '').replace(')', '')
        J.append(('atlas', k, atl[k], None, None, os.path.join(OUT, f'HCP_{tag}.png')))
        if k != 'PFBN v5|REST2':
            J.append(('grad', k, atl[k], g_h, vr_h, os.path.join(OUT, f'HCP_{tag}_on_gradient.png')))
    np.save(os.path.join(OUT, 'gradient_ranges.npy'), np.array([vr_ds, vr_h]))
    return J
if __name__ == '__main__':
    J = jobs()
    with ProcessPoolExecutor(8) as ex:
        for o in ex.map(job, J):
            print(o, flush=True)
