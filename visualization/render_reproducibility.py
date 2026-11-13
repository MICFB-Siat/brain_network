import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
from pfbn import settings
import os, sys, json
os.environ['WBR_TMP'] = settings.tmp('reproducibility', 'wbr')
os.environ['MPLCONFIGDIR'] = settings.tmp('reproducibility', 'mpl')
RDR = settings.tmp('reproducibility', 'renders')
for d in (os.environ['WBR_TMP'], os.environ['MPLCONFIGDIR'], RDR):
    os.makedirs(d, exist_ok=True)
ROOT = settings.out()
sys.path[0:0] = [settings.code('evaluation', 'boundary'), settings.code('visualization')]
from boundary_tools import C, render_layers, thin_border
import numpy as np
E = f'{ROOT}/reproducibility'
TAG = sys.argv[2] if len(sys.argv) > 2 else 'life_young_adults_n5_d1'
M = [('Without dynamics', 'v4_stim_t03_c1'), ('With dynamics', 'v5_stim_t03_r30')]
RED = (192, 57, 43)

def dil(m, e):
    n = m.copy()
    n[e[m[e[:, 0]], 1]] = True
    n[e[m[e[:, 1]], 0]] = True
    return n

def render():
    meta = {}
    for (nm, m) in M:
        A = C.load_labels(f'{E}/atlases/{TAG}_A/results/labels_{m}_clean.npy')
        B = C.load_labels(f'{E}/atlases/{TAG}_B/results/labels_{m}_clean.npy')
        (sh, od) = ({}, {})
        tot = 0
        bad = 0
        for h in C.HEMIS:
            e = C.edges(h)
            ba = thin_border(C.hemi_lab(A, h), e)
            bb = thin_border(C.hemi_lab(B, h), e)
            s = ba & dil(bb, e) | bb & dil(ba, e)
            o = ba & ~dil(bb, e) | bb & ~dil(ba, e)
            sh[h] = s
            od[h] = o
            tot += ba.sum() + bb.sum()
            bad += (ba & ~dil(bb, e)).sum() + (bb & ~dil(ba, e)).sum()
        render_layers(f'{RDR}/{TAG}_{m}.png', [(sh, (0, 0, 0)), (od, RED)], background=None, base_rgb=(235, 235, 235))
        meta[nm] = dict(frac_irreproducible=float(bad / tot))
        print(nm, meta[nm], flush=True)
    json.dump(meta, open(f'{E}/results/brain_{TAG}.json', 'w'), indent=1)

def render2():
    from scipy.optimize import linear_sum_assignment
    meta = {}
    for (nm, m) in M:
        A = C.load_labels(f'{E}/atlases/{TAG}_A/results/labels_{m}_clean.npy')
        B = C.load_labels(f'{E}/atlases/{TAG}_B/results/labels_{m}_clean.npy')
        (ka, kb) = (np.unique(A[A > 0]), np.unique(B[B > 0]))
        X = np.zeros((len(ka), len(kb)))
        k = (A > 0) & (B > 0)
        np.add.at(X, (np.searchsorted(ka, A[k]), np.searchsorted(kb, B[k])), 1)
        (r, c) = linear_sum_assignment(-X)
        mp = dict(zip(kb[c], ka[r]))
        Bm = np.array([mp.get(x, -1) if x > 0 else 0 for x in B])
        dis = k & (Bm != A)
        lay_d = {h: C.hemi_lab(dis.astype(int), h) > 0 for h in C.HEMIS}
        bord = {h: thin_border(C.hemi_lab(A, h), C.edges(h)) for h in C.HEMIS}
        render_layers(f'{RDR}/{TAG}_{m}_dis.png', [(lay_d, RED), (bord, (120, 120, 120))], background=None, base_rgb=(235, 235, 235))
        meta[nm] = dict(frac_irreproducible=float(dis.sum() / k.sum()))
        print(nm, meta[nm], flush=True)
    json.dump(meta, open(f'{E}/results/brain_{TAG}.json', 'w'), indent=1)

def fig():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    plt.rcParams.update({'font.family': 'Liberation Sans', 'font.weight': 'bold', 'font.size': 7.5, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    meta = json.load(open(f'{E}/results/brain_{TAG}.json'))
    ims = {nm: plt.imread(f'{RDR}/{TAG}_{m}_dis.png')[..., :3] for (nm, m) in M}
    red = {k: (v[..., 0] > 0.6) & (v[..., 1] < 0.35) & (v[..., 2] < 0.3) for (k, v) in ims.items()}
    d = red['Without dynamics'].astype(float) - red['With dynamics'].astype(float)
    (H, W) = d.shape
    WIN = 560
    from scipy.ndimage import uniform_filter
    s = uniform_filter(d, WIN)
    s[:WIN // 2] = s[-WIN // 2:] = -1000000000.0
    s[:, :WIN // 2] = s[:, -WIN // 2:] = -1000000000.0
    (rc, cc) = np.unravel_index(np.argmax(s), s.shape)
    (r0, c0) = (rc - WIN // 2, cc - WIN // 2)
    fig = plt.figure(figsize=(7.2, 5.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.12, wspace=0.06, left=0.02, right=0.98, top=0.9, bottom=0.03)
    for (j, (nm, m)) in enumerate(M):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(ims[nm])
        ax.axis('off')
        ax.add_patch(Rectangle((c0, r0), WIN, WIN, fill=False, ls='--', lw=1.3, ec='k'))
        f = meta[nm]['frac_irreproducible']
        ax.set_title(f'{nm}\nvertices in a different parcel between halves: {100 * f:.0f}%', fontsize=8.5, color='#C0392B' if j == 0 else '#1F5F86')
        z = fig.add_subplot(gs[1, j])
        z.imshow(ims[nm][r0:r0 + WIN, c0:c0 + WIN])
        z.set_xticks([])
        z.set_yticks([])
        [s_.set_linewidth(2.2) or s_.set_edgecolor('#555555' if j == 0 else '#8CC3DE') for s_ in z.spines.values()]
    fig.text(0.02, 0.965, 'Young adults, atlas from 5 participants: time half A vs half B of the same people', fontsize=8.5, ha='left')
    fig.text(0.98, 0.965, 'red = parcel assignment not reproduced', fontsize=7.5, ha='right', color='#C0392B')
    for e in ('png', 'pdf', 'svg'):
        fig.savefig(f'{E}/figs/fig_brain_{TAG}.{e}', dpi=300)
if __name__ == '__main__':
    {'render': render, 'render2': render2, 'fig': fig}[sys.argv[1]]()
