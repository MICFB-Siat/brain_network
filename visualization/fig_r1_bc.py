import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
from pfbn import settings
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.stats import wilcoxon
R = settings.out()
S = os.path.join(R, 'boundary_distance', 'results')
OUT = os.path.join(R, 'prism_export', 'figs')
os.makedirs(OUT, exist_ok=True)
fam = 'Liberation Sans' if any(('Liberation Sans' in f.name for f in font_manager.fontManager.ttflist)) else 'DejaVu Sans'
plt.rcParams.update({'font.family': fam, 'font.weight': 'bold', 'axes.labelweight': 'bold', 'axes.linewidth': 1.2, 'xtick.major.width': 1.2, 'ytick.major.width': 1.2, 'font.size': 8, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
COL = {'CATO': '#555555', 'Micapipe': '#E2B6C1', 'PFBN': '#8CC3DE'}
G = [('children', 'Children'), ('adolescents', 'Adolescents'), ('young_adults', 'Young adults'), ('older_adults', 'Older adults'), ('NC', 'NC'), ('PD', 'PD')]
stars = lambda p: '****' if p < 0.0001 else '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'

def bracket(ax, i, j, y, h, txt):
    ax.plot([i, i, j, j], [y - h, y, y, y - h], color='k', lw=1.2)
    ax.text((i + j) / 2, y + (h * 0.2 if txt == 'n.s.' else -h * 0.6), txt, ha='center', va='bottom', fontsize=10 if txt != 'n.s.' else 7.5, weight='bold')

def outline_violin(ax, data, cols, fill=False):
    for (i, (v, c)) in enumerate(zip(data, cols)):
        b = ax.violinplot(v, positions=[i], widths=0.75, showextrema=False)['bodies'][0]
        b.set_facecolor('#E6E6E6' if fill else 'white')
        b.set_edgecolor(c)
        b.set_linewidth(1.8)
        b.set_alpha(1)
        if not fill:
            (q1, q2, q3) = np.percentile(v, [25, 50, 75])
            for (q, ls) in ((q1, ':'), (q2, '--'), (q3, ':')):
                ax.plot([i - 0.2, i + 0.2], [q, q], color='0.2', ls=ls, lw=1.0)

def header(fig, axs, text):
    x0 = axs[0].get_position().x0
    x1 = axs[-1].get_position().x1
    y = axs[0].get_position().y1 + 0.13
    fig.add_artist(plt.Line2D([x0, x1], [y, y], color='k', lw=1.2))
    for x in (x0, x1):
        fig.add_artist(plt.Line2D([x, x], [y, y - 0.025], color='k', lw=1.2))
    fig.add_artist(plt.Line2D([(x0 + x1) / 2] * 2, [y, y + 0.02], color='k', lw=1.2))
    fig.text((x0 + x1) / 2, y + 0.03, text, ha='center', va='bottom', fontsize=10, weight='bold')

def finish(ax, labels):
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8, weight='bold')
    ax.spines[['top', 'right']].set_visible(False)

def panel_c():
    d = pd.read_csv(os.path.join(S, 'dcbc_groups_with_cato.csv'))
    (fig, axs) = plt.subplots(1, 6, figsize=(10, 3.3))
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.25, top=0.72, wspace=0.55)
    for (ax, (g, gn)) in zip(axs, G):
        x = d[d.group == g].pivot(index='subject', columns='atlas', values='DCBC')
        (c, m, o) = (x['CATO'].values, x['Micapipe'].values, x['Own PFBN v5'].values)
        outline_violin(ax, [c, m, o], [COL['CATO'], COL['Micapipe'], COL['PFBN']])
        (lo, hi) = (min(c.min(), m.min(), o.min()), max(c.max(), m.max(), o.max()))
        rg = hi - lo
        bracket(ax, 1, 2, hi + rg * 0.1, rg * 0.04, stars(wilcoxon(o, m).pvalue))
        bracket(ax, 0, 2, hi + rg * 0.28, rg * 0.04, stars(wilcoxon(o, c).pvalue))
        ax.set_ylim(lo - rg * 0.05, hi + rg * 0.45)
        ax.set_title(f'{gn}\n(n = {len(o)})', fontsize=8.5, weight='bold', pad=4)
        finish(ax, ['CATO', 'Micapipe', 'PFBN'])
    axs[0].set_ylabel('DCBC (held-out data)', fontsize=9)
    header(fig, axs[:4], 'Age groups')
    header(fig, axs[4:], 'Disease')
    fig.text(0.005, 0.97, 'c', fontsize=14, weight='bold', va='top')
    for e in ('png', 'pdf', 'svg'):
        fig.savefig(os.path.join(OUT, f'fig_R1_c_DCBC.{e}'), dpi=600)
    plt.close(fig)

def panel_b():
    t = pd.read_csv(os.path.join(S, 'dist_f1_groups.csv'))
    t = t[t.roi == 'Whole cortex']
    nl = np.load(os.path.join(S, 'dist_f1_nulls.npz'))
    (fig, axs) = plt.subplots(1, 6, figsize=(10, 3.3))
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.25, top=0.72, wspace=0.55)
    for (ax, (g, gn)) in zip(axs, G):
        A = ['CATO', 'Micapipe', 'PFBN v5']
        nulls = [nl[f'{g}|Whole cortex|{a}|dist'] for a in A]
        obs = [float(t[(t.group == g) & (t.comparison == a)].dist_mm.iloc[0]) for a in A]
        outline_violin(ax, nulls, ['#9E9E9E'] * 3, fill=True)
        for (i, (o, c)) in enumerate(zip(obs, [COL['CATO'], COL['Micapipe'], COL['PFBN']])):
            ax.scatter([i], [o], marker='D', s=46, color=c, edgecolor='black', lw=0.9, zorder=3)
        lo = min(min(map(np.min, nulls)), min(obs))
        hi = max(max(map(np.max, nulls)), max(obs))
        rg = hi - lo
        pm = float(t[(t.group == g) & (t.comparison == 'PFBN v5 - Micapipe')].p_dist.iloc[0])
        pc = float(t[(t.group == g) & (t.comparison == 'PFBN v5 - CATO')].p_dist.iloc[0])
        bracket(ax, 1, 2, hi + rg * 0.1, rg * 0.04, stars(pm))
        bracket(ax, 0, 2, hi + rg * 0.28, rg * 0.04, stars(pc))
        ax.set_ylim(lo - rg * 0.05, hi + rg * 0.45)
        ax.set_title(gn, fontsize=8.5, weight='bold', pad=4)
        finish(ax, ['CATO', 'Micapipe', 'PFBN'])
    axs[0].set_ylabel('Boundary-to-gradient\ndistance (mm)', fontsize=9)
    header(fig, axs[:4], 'Age groups')
    header(fig, axs[4:], 'Disease')
    fig.text(0.005, 0.97, 'b', fontsize=14, weight='bold', va='top')
    fig.text(0.99, 0.01, 'Grey violins: 1000 spin rotations; diamonds: observed; one-sided paired spin test (PFBN closer)', ha='right', fontsize=6.5, weight='normal')
    for e in ('png', 'pdf', 'svg'):
        fig.savefig(os.path.join(OUT, f'fig_R1_b_distance.{e}'), dpi=600)
    plt.close(fig)
if __name__ == '__main__':
    panel_b()
    panel_c()
    print('ok', fam)
