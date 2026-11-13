import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
from pfbn import settings
import os
os.environ.setdefault('MPLCONFIGDIR', settings.tmp('mpl'))
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from scipy.stats import wilcoxon
for f in (settings.path('fonts', 'Arial.ttf'), settings.path('fonts', 'Arial_Bold.ttf')):
    if os.path.exists(f):
        fm.fontManager.addfont(f)
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'], 'font.size': 7, 'axes.titlesize': 7.5, 'axes.labelsize': 7, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none', 'legend.fontsize': 6.5, 'legend.frameon': False})
COL = {'Micapipe': '#8C8C8C', 'PFBN v0': '#5B8FD0', 'PFBN v5': '#D1495B', 'PFBN v5 w/o dynamics': '#E8A33D', 'CATO (Lausanne-219)': '#6A8D73'}

def xlab(m):
    return {'PFBN v5 w/o dynamics': 'PFBN v5\nw/o dyn.', 'CATO (Lausanne-219)': 'CATO\n(L-219)'}.get(m, m.replace('PFBN ', 'PFBN\n').replace(' (', '\n('))

def pstr(p):
    return 'n.s.' if p >= 0.05 else '*' if p >= 0.01 else '**' if p >= 0.001 else '***'

def violins(ax, data, order, ylabel, title=None, paired=True, ref='Micapipe', points=True, show_stats=True, width=0.7, point_size=5, stats_out=None, key=None):
    pos = np.arange(len(order))
    for (i, m) in enumerate(order):
        y = np.asarray(data[m], float)
        y = y[np.isfinite(y)]
        v = ax.violinplot(y, positions=[i], widths=width, showextrema=False)
        for b in v['bodies']:
            b.set_facecolor(COL.get(m, '#999999'))
            b.set_edgecolor('none')
            b.set_alpha(0.35)
        ax.boxplot(y, positions=[i], widths=0.16, showfliers=False, patch_artist=True, medianprops=dict(color='black', lw=0.9), boxprops=dict(facecolor='white', edgecolor='black', lw=0.6), whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6))
        if points:
            jit = (np.random.default_rng(i).random(len(y)) - 0.5) * 0.18
            ax.scatter(i + 0.22 + jit * 0.5, y, s=point_size, color=COL.get(m, '#999999'), edgecolor='none', alpha=0.8, zorder=3)
    if paired and points and (len(order) > 1):
        n = len(np.asarray(data[order[0]]))
        if all((len(np.asarray(data[m])) == n for m in order)):
            for j in range(n):
                ax.plot(pos + 0.22, [data[m][j] for m in order], color='#BBBBBB', lw=0.25, alpha=0.5, zorder=1)
    ax.set_xticks(pos)
    ax.set_xticklabels([xlab(m) for m in order], rotation=0)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    res = {}
    if show_stats and paired and (ref in order):
        top = max((np.nanmax(np.asarray(data[m], float)) for m in order))
        rng = top - min((np.nanmin(np.asarray(data[m], float)) for m in order))
        k = 0
        for (i, m) in enumerate(order):
            if m == ref:
                continue
            (a, b) = (np.asarray(data[m], float), np.asarray(data[ref], float))
            w = wilcoxon(a, b)
            res[m] = dict(n=int(len(a)), W=float(w.statistic), p=float(w.pvalue), median_diff=float(np.median(a - b)), n_greater=int((a > b).sum()))
            y = top + rng * (0.06 + 0.09 * k)
            i0 = order.index(ref)
            k += 1
            ax.plot([i0, i0, i, i], [y - rng * 0.02, y, y, y - rng * 0.02], color='black', lw=0.5)
            ax.text((i0 + i) / 2, y + rng * 0.005, pstr(w.pvalue), ha='center', va='bottom', fontsize=6.5)
        if stats_out is not None:
            stats_out[key or ylabel] = res
    return res

def null_violins(ax, nulls, obs, order, ylabel, title=None, higher_better=True, zs=None):
    for (i, m) in enumerate(order):
        v = ax.violinplot(np.asarray(nulls[m]), positions=[i], widths=0.7, showextrema=False)
        for b in v['bodies']:
            b.set_facecolor('#D9D9D9')
            b.set_edgecolor('none')
            b.set_alpha(1)
        ax.scatter([i], [obs[m]], marker='D', s=22, color=COL.get(m, '#999'), edgecolor='black', linewidth=0.5, zorder=3)
        if zs is not None:
            ax.text(i, obs[m], f'  z={zs[m]:.1f}', fontsize=6, va='center', ha='left')
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([xlab(m) for m in order])
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

def save(fig, path_noext):
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(f'{path_noext}.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)

def panel_label(ax, s, x=-0.18, y=1.08):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight='bold', va='bottom', ha='left')

def image_panel(ax, png, title=None):
    ax.imshow(plt.imread(png))
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=7.5)
