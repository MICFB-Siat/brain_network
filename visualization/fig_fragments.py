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
os.environ['WBR_TMP'] = settings.tmp('reproducibility', 'wbr')
os.environ['MPLCONFIGDIR'] = settings.tmp('reproducibility', 'mpl')

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
fig = plt.figure(figsize=(7.2, 4.8))
gs = fig.add_gridspec(2, 4, height_ratios=[1.35, 1], hspace=0.5, wspace=0.4)
for (j, nm) in enumerate(['Without dynamics', 'With dynamics']):
    ax = fig.add_subplot(gs[0, 2 * j:2 * j + 2])
    (im, nf) = imgs[nm]
    ax.imshow(im)
    ax.axis('off')
    ax.set_title(f'{nm}: {nf} fragment vertices', fontsize=8, fontweight='bold', color=FRAG if j == 0 else '#1F5F86')
for (k, (c, t)) in enumerate(st):
    ax = fig.add_subplot(gs[1, k])
    g = d[d.cohort == c]
    for (meth, col, mk) in (('Without dynamics', '#555555', 's'), ('With dynamics', OURS, 'o')):
        s_ = g[g.method == meth].groupby('n').island_vertices_raw.agg(['mean', 'sem']).reset_index()
        ax.errorbar(s_.n, s_['mean'], yerr=s_['sem'].fillna(0), color=col, mfc=col, mec='k', mew=0.5, marker=mk, ms=4, lw=1.6, capsize=2, label=meth)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks([5, 10, 20, 50])
    ax.set_xticklabels(['5', '10', '20', '50'])
    ax.minorticks_off()
    ax.set_title(t, fontsize=7.5, fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlabel('Subjects')
    if k == 0:
        ax.set_ylabel('Fragment vertices')
        ax.legend(frameon=False, fontsize=5.5, loc='lower left')
fig.text(0.02, 0.97, f"Adolescents, atlas from 5 participants ({EX}); red = vertices outside their parcel's main piece", fontsize=7)
for e in ('png', 'pdf', 'svg'):
    fig.savefig(f'{E}/figs/fig_fragments.{e}', dpi=300, bbox_inches='tight')
print({k: v[1] for (k, v) in imgs.items()})
