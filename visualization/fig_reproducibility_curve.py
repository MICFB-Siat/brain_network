import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
from pfbn import settings
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
E = settings.out('reproducibility')
plt.rcParams.update({'font.family': 'Liberation Sans', 'font.weight': 'bold', 'axes.labelweight': 'bold', 'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 1.3})
(OURS, WO) = ('#8CC3DE', '#555555')
d = pd.read_csv(f'{E}/results/reliability.csv')
d.loc[d.part == 'A', 'n'] = 50
ST = [('life_children', 'Children'), ('life_adolescents', 'Adolescents'), ('life_young_adults', 'Young adults'), ('life_older_adults', 'Older adults')]
(fig, axs) = plt.subplots(1, 4, figsize=(8.4, 2.6), sharey=True)
for (ax, (c, t)) in zip(axs, ST):
    g = d[d.cohort == c]
    ax.axvspan(4, 24, color='#EFEFEF', zorder=0)
    for (meth, col, mk, lw) in (('Without dynamics', WO, 's', 1.6), ('With dynamics', OURS, 'o', 2.4)):
        s = g[g.method == meth].groupby('n').ARI.agg(['mean', 'sem']).reset_index()
        ax.errorbar(s.n, s['mean'], yerr=s['sem'].fillna(0), color=col, mfc=col, mec='k', mew=0.6, marker=mk, ms=5, lw=lw, capsize=2.5, label=meth)
    ax.set_xscale('log')
    ax.set_xticks([5, 10, 20, 50])
    ax.set_xticklabels(['5', '10', '20', '50'])
    ax.minorticks_off()
    ax.set_title(t, fontsize=10)
    ax.set_xlabel('Subjects', fontsize=9)
    ax.set_xlim(4, 60)
axs[0].set_ylabel('Split-half reproducibility\n(ARI)', fontsize=9.5)
axs[0].set_ylim(0.37, 0.62)
b = d[d.part == 'B'].pivot_table(index=['cohort', 'n', 'draw'], columns='method', values='ARI')
x = b['With dynamics'] - b['Without dynamics']
axs[0].text(4.4, 0.605, f'5-20 subjects: With > Without\n{(x > 0).sum()}/{len(x)} atlas pairs, P = {wilcoxon(x).pvalue:.0e}', fontsize=7, va='top')
axs[-1].legend(frameon=False, fontsize=7.5, loc='lower right')
fig.subplots_adjust(left=0.1, right=0.99, top=0.88, bottom=0.2, wspace=0.12)
for e in ('png', 'pdf'):
    fig.savefig(f'{E}/figs/fig_dynamics_reproducibility_curve.{e}', dpi=300, transparent=e == 'pdf')
d.to_csv(f'{E}/results/prism_R1_reproducibility_curve.csv', index=False)
