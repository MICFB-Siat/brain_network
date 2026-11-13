import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
from pfbn import settings
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
E = settings.out('reproducibility')
plt.rcParams.update({'font.family': 'Liberation Sans', 'font.weight': 'bold', 'axes.labelweight': 'bold', 'pdf.fonttype': 42, 'svg.fonttype': 'none', 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 1.2})
(OURS, WO, RED) = ('#8CC3DE', '#555555', '#C0392B')
EX = 'life_adolescents_n5_d0_A'
fig = plt.figure(figsize=(7.4, 5.4))
gs = fig.add_gridspec(2, 4, height_ratios=[1.25, 1], hspace=0.5, wspace=0.12, left=0.09, right=0.99, top=0.93, bottom=0.09)
for (j, (nm, m, col, nf)) in enumerate([('Without dynamics', 'v4_stim_t03_c1', RED, 9230), ('With dynamics', 'v5_stim_t03_r30', '#1F5F86', 1899)]):
    ax = fig.add_subplot(gs[0, 2 * j:2 * j + 2])
    im = plt.imread(f'{E}/figs/png_cache/{EX}_{m}.png')[..., :3]
    mk = (im < 0.97).any(2)
    (r, c) = (np.where(mk.any(1))[0], np.where(mk.any(0))[0])
    ax.imshow(im[r[0]:r[-1] + 1, c[0]:c[-1] + 1])
    ax.axis('off')
    ax.set_title(f'{nm}\n{nf:,} fragmented vertices', fontsize=10, color=col)
fig.text(0.54, 0.475, 'Atlas from 5 participants (adolescents); red = fragmented vertices', fontsize=8.5, ha='center')
d = pd.read_csv(f'{E}/results/reliability.csv')
d = d[d.part == 'B']
ST = [('life_children', 'Children'), ('life_adolescents', 'Adolescents'), ('life_young_adults', 'Young adults'), ('life_older_adults', 'Older adults')]
for (k, (c, t)) in enumerate(ST):
    ax = fig.add_subplot(gs[1, k], sharey=None if k == 0 else fig.axes[2])
    g = d[d.cohort == c]
    for (meth, col, mk_, lw) in (('Without dynamics', WO, 's', 1.6), ('With dynamics', OURS, 'o', 2.6)):
        s = g[g.method == meth].groupby('n').ARI.agg(['mean', 'sem']).reset_index()
        ax.errorbar(s.n, s['mean'], yerr=s['sem'], color=col, mfc=col, mec='k', mew=0.6, marker=mk_, ms=5.5, lw=lw, capsize=2.5, label=meth)
    ax.set_xscale('log')
    ax.set_xticks([5, 10, 20])
    ax.set_xticklabels(['5', '10', '20'])
    ax.minorticks_off()
    ax.set_xlim(4.2, 24)
    ax.set_title(t, fontsize=9)
    ax.set_xlabel('Participants', fontsize=8.5)
    if k == 0:
        ax.set_ylabel('Split-half reproducibility\n(ARI)', fontsize=8.5)
    else:
        ax.tick_params(labelleft=False)
fig.axes[2].set_ylim(0.36, 0.6)
x = d.pivot_table(index=['cohort', 'n', 'draw'], columns='method', values='ARI')
dd = x['With dynamics'] - x['Without dynamics']
fig.axes[2].text(4.5, 0.595, f'With > Without: {(dd > 0).sum()}/{len(dd)} atlases\nP = {wilcoxon(dd).pvalue:.0e}', fontsize=7.5, va='top')
fig.axes[-1].legend(frameon=False, fontsize=8, loc='lower center', bbox_to_anchor=(-1.15, 1.13), ncol=2)
for e in ('png', 'pdf', 'svg'):
    fig.savefig(f'{E}/figs/fig_dynamics_final.{e}', dpi=300)
