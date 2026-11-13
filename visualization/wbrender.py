import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
from pfbn import settings
import os, shutil, subprocess, tempfile, colorsys, numpy as np, nibabel as nib
WB = settings.get('wb_command')
PKG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wbplot_pkg', 'wbplot', 'data')
SURF = os.path.join(PKG, 'HumanCorticalParcellations', 'S1200.{}.midthickness_MSMAll.32k_fs_LR.surf.gii')
BILATERAL = (1600, 1000)
NV = 32492

def _run(cmd):
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(' '.join(map(str, cmd)) + '\n' + r.stdout[-2000:] + r.stderr[-2000:])

def _scene_dir():
    t = tempfile.mkdtemp(prefix='wbr_', dir=os.environ.get('WBR_TMP', settings.tmp()))
    shutil.copy(os.path.join(PKG, 'Human.scene'), t)
    os.symlink(os.path.join(PKG, 'HumanCorticalParcellations'), os.path.join(t, 'HumanCorticalParcellations'))
    for f in ('ImageDense.dscalar.nii', 'ImageParcellated.dlabel.nii'):
        shutil.copy(os.path.join(PKG, f), t)
    return t

def _capture(t, scene, out, size):
    _run([WB, '-scene-capture-image', os.path.join(t, 'Human.scene'), scene, os.path.abspath(out), '-size-width-height', size[0], size[1]])

def _metric(path, x):
    g = nib.gifti.GiftiImage(darrays=[nib.gifti.GiftiDataArray(np.asarray(x, np.float32))])
    nib.save(g, path)

def _dense_from_arrays(t, lh, rh, out):
    _metric(os.path.join(t, 'l.func.gii'), lh)
    _metric(os.path.join(t, 'r.func.gii'), rh)
    _run([WB, '-cifti-create-dense-from-template', os.path.join(PKG, 'ImageDense.dscalar.nii'), out, '-metric', 'CORTEX_LEFT', os.path.join(t, 'l.func.gii'), '-metric', 'CORTEX_RIGHT', os.path.join(t, 'r.func.gii')])
_ADJ = {}

def adjacency(hemi):
    if hemi not in _ADJ:
        f = nib.load(SURF.format(hemi)).darrays[1].data.astype(int)
        e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
        _ADJ[hemi] = e
    return _ADJ[hemi]

def border_mask(lab, hemi, thin=True):
    e = adjacency(hemi)
    (a, b) = (lab[e[:, 0]], lab[e[:, 1]])
    m = np.zeros(NV, bool)
    d = (a != b) & (a > 0) & (b > 0)
    if thin:
        big = np.where(a > b, e[:, 0], e[:, 1])
        m[big[d]] = True
    else:
        m[e[d, 0]] = True
        m[e[d, 1]] = True
    return m

def dlabel_to_arrays(path):
    img = nib.load(path)
    ax = img.header.get_axis(1)
    d = np.asarray(img.get_fdata()[0]).astype(int)
    out = {}
    for (name, sl, bm) in ax.iter_structures():
        h = {'CIFTI_STRUCTURE_CORTEX_LEFT': 'L', 'CIFTI_STRUCTURE_CORTEX_RIGHT': 'R'}.get(name)
        if h:
            x = np.zeros(NV, int)
            x[bm.vertex] = d[sl]
            out[h] = x
    return (out['L'], out['R'])

def parcel_colors(n, seed=0):
    rng = np.random.default_rng(seed)
    h = np.arange(n) * 0.618033988749895 % 1.0
    cols = [colorsys.hls_to_rgb(hh, 0.45 + 0.2 * rng.random(), 0.55 + 0.35 * rng.random()) for hh in h]
    return np.array(cols)[rng.permutation(n)]

def labels_png(lh, rh, out, colors=None, borders=True, size=BILATERAL, seed=0, thin=True):
    lh = np.asarray(lh, int).copy()
    rh = np.asarray(rh, int).copy()
    keys = np.unique(np.concatenate([lh[lh > 0], rh[rh > 0]]))
    if colors is not None and (not borders):
        keys = np.array(sorted(set(keys.tolist()) | set((k for k in colors))))
    if colors is None:
        colors = dict(zip(keys.tolist(), parcel_colors(len(keys), seed)))
    bkey = int(keys.max()) + 1
    if borders:
        (bl, br) = (border_mask(lh, 'L', thin), border_mask(rh, 'R', thin))
        lh[bl] = bkey
        rh[br] = bkey
    t = _scene_dir()
    try:
        tbl = os.path.join(t, 'labels.txt')
        with open(tbl, 'w') as f:
            for k in keys:
                (r, g, b) = np.clip(np.asarray(colors[int(k)]) * 255, 0, 255).astype(int)
                f.write(f'P{int(k)}\n{int(k)} {r} {g} {b} 255\n')
            f.write(f'BORDER\n{bkey} 0 0 0 255\n')
        ds = os.path.join(t, 'lab.dscalar.nii')
        _dense_from_arrays(t, lh, rh, ds)
        _run([WB, '-cifti-label-import', ds, tbl, os.path.join(t, 'ImageParcellated.dlabel.nii'), '-discard-others'])
        _capture(t, 1, out, size)
    finally:
        shutil.rmtree(t, ignore_errors=True)

def dlabel_png(dlabel, out, **kw):
    (lh, rh) = dlabel_to_arrays(dlabel)
    labels_png(lh, rh, out, **kw)

def scalar_png(lh, rh, out, palette='magma', pos_range=None, neg_range=None, size=BILATERAL, extra=()):
    t = _scene_dir()
    try:
        ds = os.path.join(t, 'tmp.dscalar.nii')
        _dense_from_arrays(t, lh, rh, ds)
        cmd = [WB, '-cifti-palette', ds, 'MODE_USER_SCALE', os.path.join(t, 'ImageDense.dscalar.nii'), '-palette-name', palette, '-disp-pos', 'true', '-disp-neg', 'true' if neg_range else 'false']
        if pos_range:
            cmd += ['-pos-user', pos_range[0], pos_range[1]]
        if neg_range:
            cmd += ['-neg-user', neg_range[0], neg_range[1]]
        _run(cmd + list(extra))
        _capture(t, 6, out, size)
    finally:
        shutil.rmtree(t, ignore_errors=True)

def scalar_with_borders_png(g_lh, g_rh, lab_lh, lab_rh, out, cmap='viridis', vmin=None, vmax=None, n_levels=128, border_rgb=(0, 0, 0), size=BILATERAL, thin=True):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.cm as cm
    g = np.concatenate([g_lh, g_rh]).astype(float)
    ok = np.isfinite(g)
    vmin = np.nanpercentile(g[ok], 2) if vmin is None else vmin
    vmax = np.nanpercentile(g[ok], 98) if vmax is None else vmax
    q = np.zeros(len(g), int)
    q[ok] = 1 + np.clip(((g[ok] - vmin) / (vmax - vmin) * n_levels).astype(int), 0, n_levels - 1)
    (lh, rh) = (q[:NV].copy(), q[NV:].copy())
    bl = border_mask(np.asarray(lab_lh, int), 'L', thin)
    br = border_mask(np.asarray(lab_rh, int), 'R', thin)
    bkey = n_levels + 1
    lh[bl & (lh > 0)] = bkey
    rh[br & (rh > 0)] = bkey
    cmap_ = cm.get_cmap(cmap, n_levels)
    colors = {k: cmap_(k - 1)[:3] for k in range(1, n_levels + 1)}
    colors[bkey] = tuple(np.asarray(border_rgb) / 255.0)
    labels_png(lh, rh, out, colors=colors, borders=False, size=size)
    return (vmin, vmax)
