import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
from pfbn import settings
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, settings.code('pfbn', 'pipeline'))
sys.path.insert(0, settings.code('visualization'))
import common as C, wbrender as W
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
plt.rcParams.update({'font.family': 'Arial', 'font.size': 7, 'axes.linewidth': 0.8, 'pdf.fonttype': 42})
E = settings.out('reproducibility')
(OURS, WO, FRAG) = ('#8CC3DE', '#B3B3B3', '#C0392B')
EX = sys.argv[1] if len(sys.argv) > 1 else 'life_adolescents_n5_d0_A'
os.environ.setdefault('WBR_TMP', settings.tmp())

def fragments(lab):
    out = np.zeros(len(lab), bool)
    for h in C.HEMIS:
        l = lab[C.HEMI_SL[h]]
        e = C.hemi_edges_cortex(h)
        same = l[e[:, 0]] == l[e[:, 1]]
        (k, cc) = sp.csgraph.connected_components(sp.coo_matrix((np.ones(same.sum()), (e[same, 0], e[same, 1])), shape=(len(l), len(l))), directed=False)
        df = pd.DataFrame({'l': l, 'c': cc})
        big = df.groupby(['l', 'c']).size().reset_index(name='n').sort_values('n').drop_duplicates('l', keep='last')
        keep = set(zip(big.l, big.c))
        out[C.HEMI_SL[h]] = [(a, b) not in keep for (a, b) in zip(l, cc)]
    return out

def render(lab, png):
    fr = fragments(lab)
    full = {}
    for (h, L) in (('lh', 'L'), ('rh', 'R')):
        x = np.zeros(C.NV, int)
        ci = C.cortex_idx(h)
        l = lab[C.HEMI_SL[h]]
        f = fr[C.HEMI_SL[h]]
        v = np.ones(len(l), int)
        bm = np.zeros(C.NV, bool)
        tmp = np.zeros(C.NV, int)
        tmp[ci] = l
        bm = W.border_mask(tmp, L, thin=True)
        x[ci] = v
        x[bm & (x > 0)] = 3
        x[ci[f]] = 2
        full[h] = x
    W.labels_png(full['lh'], full['rh'], png, colors={1: (0.93, 0.93, 0.93), 2: tuple((int(FRAG[i:i + 2], 16) / 255 for i in (1, 3, 5))), 3: (0.3, 0.3, 0.3)}, borders=False)
    return int(fr.sum())
os.makedirs(f'{E}/figs/png_cache', exist_ok=True)
imgs = {}
for (m, nm) in (('v4_stim_t03_c1', 'Without dynamics'), ('v5_stim_t03_r30', 'With dynamics')):
    lab = np.load(f'{E}/atlases/{EX}/results/labels_{m}.npy').astype(int)
    png = f'{E}/figs/png_cache/{EX}_{m}.png'
    nf = render(lab, png)
    imgs[nm] = (plt.imread(png), nf)
d = pd.read_csv(f'{E}/results/reliability.csv')
d.loc[d.part == 'A', 'n'] = 50
st = [('life_children', 'Children'), ('life_adolescents', 'Adolescents'), ('life_young_adults', 'Young adults'), ('life_older_adults', 'Older adults')]
fig = plt.figure(figsize=(7.2, 6.0))
gs = fig.add_gridspec(3, 4, height_ratios=[1.35, 1, 1], hspace=0.55, wspace=0.45)
for (j, nm) in enumerate(['Without dynamics', 'With dynamics']):
    ax = fig.add_subplot(gs[0, 2 * j:2 * j + 2])
    (im, nf) = imgs[nm]
    ax.imshow(im)
    ax.axis('off')
    ax.set_title(f'{nm}\n{nf} fragment vertices', fontsize=8, fontweight='bold', color=FRAG if j == 0 else '#1F5F86')
    (H, Wd) = im.shape[:2]
    for (x0, y0, w, h) in [(0.03, 0.05, 0.44, 0.42), (0.53, 0.05, 0.44, 0.42), (0.03, 0.53, 0.44, 0.42), (0.53, 0.53, 0.44, 0.42)][:0]:
        ax.add_patch(Rectangle((x0 * Wd, y0 * H), w * Wd, h * H, fill=False, ls='--', lw=0.8, ec='k'))
for (row, (met, ylab, logy)) in enumerate([('ARI', 'Split-half reproducibility (ARI)', False), ('island_vertices_raw', 'Fragment vertices', True)], start=1):
    for (k, (c, t)) in enumerate(st):
        ax = fig.add_subplot(gs[row, k])
        g = d[d.cohort == c]
        for (meth, col, mk) in (('Without dynamics', WO, 's'), ('With dynamics', OURS, 'o')):
            s = g[g.method == meth].groupby('n')[met].agg(['mean', 'sem', 'count']).reset_index()
            if len(s) == 0:
                continue
            ax.errorbar(s.n, s['mean'], yerr=s['sem'].fillna(0), color=col if meth.startswith('With') else '#7F7F7F', mfc=col, mec='k', mew=0.5, marker=mk, ms=4, lw=1.4, capsize=2, label=meth)
        ax.set_xscale('log')
        ax.set_xticks([5, 10, 20, 50])
        ax.set_xticklabels(['5', '10', '20', '50'])
        ax.minorticks_off()
        if logy:
            ax.set_yscale('log')
        ax.set_title(t, fontsize=7.5, fontweight='bold')
        ax.spines[['top', 'right']].set_visible(False)
        if k == 0:
            ax.set_ylabel(ylab)
        if row == 2:
            ax.set_xlabel('Subjects in atlas')
        if row == 1 and k == 0:
            ax.legend(frameon=False, fontsize=6, loc='lower right')
        ax.axvspan(4, 12, color='#F2F2F2', zorder=0)
fig.savefig(f'{E}/figs/fig_dynamics_reliability.png', dpi=300, bbox_inches='tight')
fig.savefig(f'{E}/figs/fig_dynamics_reliability.pdf', bbox_inches='tight')
print({k: v[1] for (k, v) in imgs.items()})
