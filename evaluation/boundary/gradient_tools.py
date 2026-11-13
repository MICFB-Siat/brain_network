import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
from pfbn import settings
import os, sys, json, time
ROOT = settings.out()
EVAL_CODE = settings.code('evaluation', 'boundary')
WBR = settings.code('visualization')
for p in (EVAL_CODE, WBR, settings.code('evaluation', 'dcbc')):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault('WBR_TMP', settings.tmp())
os.environ.setdefault('MPLCONFIGDIR', settings.tmp('mpl'))
import numpy as np, nibabel as nib
import eval_common as C
from eval_common import log
import plotstyle as P
import wbrender as W
MMP = os.path.join(WBR, 'wbplot_pkg/wbplot/data/HumanCorticalParcellations/Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii')
SPHERE = os.path.join(C.S4TMP, 'S1200.{H}.sphere.32k_fs_LR.surf.gii')
TYF = settings.path('msm_group')
ATL = os.path.join(ROOT, 'atlases')
TMP = settings.tmp('gradient_zoom')
os.makedirs(TMP, exist_ok=True)
(K_TOL, THR) = (2, 85)
ROIS = {'Somatomotor': ['4', '3a', '3b', '1', '2'], 'Early visual': ['V1', 'V2', 'V3'], 'Temporo-parietal': ['PGi', 'PGs', 'PFm', 'PF', 'TPOJ1', 'TPOJ2', 'TPOJ3', 'STV', 'PSL', 'PGp'], 'Dorsolateral PFC': ['46', '9-46d', 'a9-46v', 'p9-46v', '8C', '8Av', 'IFJa', 'IFJp', '8Ad', '9a', '9p', 'p47r']}
VISUAL = ['V1', 'V2', 'V3', 'V4', 'V3A', 'V3B', 'V6', 'V6A', 'V7', 'IPS1', 'V3CD', 'LO1', 'LO2', 'LO3', 'V8', 'VMV1', 'VMV2', 'VMV3', 'PIT', 'FFC', 'MST', 'MT', 'V4t', 'PH', 'VVC', 'FST', 'ProS', 'DVT', 'POS1']

def mmp_names():
    img = nib.load(MMP)
    ax = img.header.get_axis(1)
    lab = img.header.get_axis(0).label[0]
    d = np.asarray(img.get_fdata()[0]).astype(int)
    out = {}
    for (name, sl, bm) in ax.iter_structures():
        h = {'CIFTI_STRUCTURE_CORTEX_LEFT': 'lh', 'CIFTI_STRUCTURE_CORTEX_RIGHT': 'rh'}.get(name)
        if not h:
            continue
        full = np.full(C.NV, '', dtype=object)
        vals = d[sl]
        nm = np.array([lab[v][0] if v in lab else '' for v in vals], dtype=object)
        nm = np.array([s[2:-4] if s.endswith('_ROI') else '' for s in nm], dtype=object)
        full[bm.vertex] = nm
        out[h] = full[C.cortex_idx(h)]
    return out

def roi_mask(names_h, areas, A=None, dil=0):
    m = np.isin(names_h, areas)
    if dil and A is not None:
        m = C.dilate(m, A, dil)
    return m

def load_gmap(path, h):
    g = np.load(path.format(h=h)).astype(np.float64)
    g = g[C.cortex_idx(h)] if g.shape[0] == C.NV else g
    return (g, np.isfinite(g) & (g != 0))

def load_func_cortex(path, h):
    g = np.asarray(nib.load(path).darrays[0].data, np.float64)
    assert g.shape[0] == C.NV, g.shape
    g = g[C.cortex_idx(h)]
    return (g, np.isfinite(g) & (g != 0))

def spins(h, n=1000):
    return np.load(C.SPINS)['idx_' + C.HL[h]][:n]

def ridge_stats(lab_h, e, A, ridge, d_r, roi, k=K_TOL):
    B = C.boundary_mask(lab_h, e) & roi
    R = ridge & roi
    B_dil = C.dilate(B, A, k)
    R_dil = C.dilate(R, A, k)
    return (dict(nB=int(B.sum()), tpB=int((B & R_dil).sum()), nR=int(R.sum()), tpR=int((R & B_dil).sum()), dsum=float(d_r[B].sum())), B)

def f1(d):
    P_ = d['tpB'] / max(d['nB'], 1)
    R_ = d['tpR'] / max(d['nR'], 1)
    return (P_, R_, 2 * P_ * R_ / (P_ + R_) if P_ + R_ > 0 else 0.0)

def pool(a, b):
    return {k: a[k] + b[k] for k in ('nB', 'tpB', 'nR', 'tpR', 'dsum')}

def full_hemi(lab, h):
    out = np.zeros(C.NV, np.int64)
    out[C.cortex_idx(h)] = C.hemi_lab(lab, h)
    return out

def to32k(x_h, h, fill=np.nan):
    out = np.full(C.NV, fill, dtype=float)
    out[C.cortex_idx(h)] = x_h
    return out

def lab32k(lab, h):
    return full_hemi(lab, h)

def crop_boxes(mask_png, n_keys, hues, min_sat=0.35):
    import matplotlib.pyplot as plt, matplotlib.colors as mc
    from scipy import ndimage
    im = plt.imread(mask_png)[..., :3]
    hsv = mc.rgb_to_hsv(im)
    out = {}
    for (k, hue) in hues.items():
        dh = np.abs((hsv[..., 0] - hue + 0.5) % 1.0 - 0.5)
        m = (dh < 0.04) & (hsv[..., 1] > min_sat) & (hsv[..., 2] > 0.15)
        (lab, n) = ndimage.label(ndimage.binary_closing(m, iterations=3))
        if n == 0:
            continue
        sz = ndimage.sum(m, lab, range(1, n + 1))
        j = int(np.argmax(sz)) + 1
        (rr, cc) = np.where(lab == j)
        out[k] = (rr.min(), rr.max(), cc.min(), cc.max())
    return out

def pad_box(b, shape, frac=0.25, square=True):
    (r0, r1, c0, c1) = b
    (h, w) = (r1 - r0, c1 - c0)
    if square:
        s = max(h, w)
        (rc, cc) = ((r0 + r1) / 2, (c0 + c1) / 2)
        h = w = s
        (r0, r1, c0, c1) = (rc - s / 2, rc + s / 2, cc - s / 2, cc + s / 2)
    r0 -= h * frac
    r1 += h * frac
    c0 -= w * frac
    c1 += w * frac
    return (int(max(r0, 0)), int(min(r1, shape[0])), int(max(c0, 0)), int(min(c1, shape[1])))

def write_sha(paths, out):
    with open(out, 'w') as f:
        for p in paths:
            f.write(f"{(C.sha256(p) if os.path.isfile(p) else 'MISSING')}  {p}\n")
