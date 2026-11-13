import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
from pfbn import settings
import os, sys, json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
EXP = settings.out('dcbc_curve')
RES = os.path.join(EXP, 'results')
FIG = os.path.join(EXP, 'figs')
sys.path.insert(0, os.path.join(EXP, 'code'))
from dcbc_curve_plot import plot_wb_curve
STATS = json.load(open(settings.out('boundary_distance', 'results', 'dcbc_cato_stats.json')))
COL = {'CATO': '#555555', 'Micapipe': '#E2B6C1', 'Our Method': '#8CC3DE'}
KEY = {'CATO': 'median|CATO', 'Micapipe': 'median|Micapipe', 'Our Method': 'median|Own PFBN v5'}
TITLE = {'children': 'Children', 'adolescents': 'Adolescents', 'young_adults': 'Young adults', 'older_adults': 'Older adults', 'NC': 'NC', 'PD': 'PD'}
plt.rcParams.update({'font.family': 'Liberation Sans', 'font.weight': 'bold', 'axes.labelweight': 'bold', 'axes.titleweight': 'bold', 'font.size': 7, 'axes.linewidth': 0.8, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'mathtext.fontset': 'custom', 'mathtext.rm': 'Liberation Sans:bold', 'mathtext.it': 'Liberation Sans:bold:italic', 'mathtext.bf': 'Liberation Sans:bold', 'xtick.major.width': 0.8, 'ytick.major.width': 0.8})

def fmt_p(p):
    if p >= 0.001:
        return f'{p:.3f}'
    (m, e) = f'{p:.1e}'.split('e')
    return f'{m} $\\times$ 10$^{{{int(e)}}}$'

def T_of(z, atlas):
    T = {}
    for (h, H) in (('lh', 'L'), ('rh', 'R')):
        (cw, cb) = (z[f'author|{atlas}|{h}|corr_within'], z[f'author|{atlas}|{h}|corr_between'])
        for (i, s) in enumerate(z['subjects']):
            T[f'{s}|{H}'] = dict(binWidth=1, maxDist=35, corr_within=cw[i], corr_between=cb[i], hemisphere=H)
    return T

def panel(ax, g, src):
    z = np.load(os.path.join(RES, f'bins_{g}.npz'))
    subs = [str(s) for s in z['subjects']]
    cur = {}
    for (k, a) in enumerate(('CATO', 'Micapipe', 'Our Method')):
        (x, w, b) = plot_wb_curve(T_of(z, a), None, sub_list=[f'{s}|' for s in subs], within_color=COL[a], between_color=COL[a], ax=ax, sem=True, label=a, lw=1.1, zorder=3 + k)
        cur[a] = (w, b)
        for (typ, y) in (('within', w), ('between', b)):
            for (xi, yi) in zip(x, y):
                src.append(dict(group=g, atlas=a, curve=typ, distance_mm=xi, mean_corr=yi))
    ax.fill_between(x, cur['Our Method'][0], cur['Our Method'][1], color='#8CC3DE', alpha=0.25, lw=0, zorder=1)
    st = STATS[g]
    txt = 'DCBC (median)\n' + '\n'.join((f'{a}: {st[KEY[a]]:.3f}' for a in ('CATO', 'Micapipe', 'Our Method')))
    txt += f"\nOur vs Micapipe: P = {fmt_p(st['p_own_vs_mica'])}\nOur vs CATO: P = {fmt_p(st['p_own_vs_cato'])}"
    ax.text(0.97, 0.97, txt, transform=ax.transAxes, ha='right', va='top', fontsize=6, linespacing=1.35)
    ax.set_title(f'{TITLE[g]} (n = {len(subs)})', fontsize=7.5, pad=3)
    ax.set_xlim(0, 35)
    ax.set_xticks([0, 5, 10, 15, 20, 25, 30, 35])
    ax.set_xlabel('Spatial distance (mm)')
    ax.spines[['top', 'right']].set_visible(False)

def figure(groups, name):
    (f, axs) = plt.subplots(1, 3, figsize=(7.2, 2.75), sharey=True)
    src = []
    for (ax, g) in zip(axs, groups):
        panel(ax, g, src)
    axs[0].set_ylabel('Correlation between vertex pairs')
    h = [Line2D([], [], color=COL[a], lw=2, label=a) for a in COL] + [Line2D([], [], color='k', lw=1.1, ls='-', label='Within-parcel'), Line2D([], [], color='k', lw=1.1, ls='--', label='Between-parcel'), Patch(color='#8CC3DE', alpha=0.25, lw=0, label='DCBC (Our Method)')]
    f.legend(handles=h, loc='upper center', ncol=6, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, 1.0), handlelength=2.2, columnspacing=1.2)
    f.tight_layout(rect=(0, 0, 1, 0.92), w_pad=0.8)
    for e in ('png', 'pdf'):
        f.savefig(os.path.join(FIG, f'{name}.{e}'), dpi=300)
    plt.close(f)
    return src
src = figure(['adolescents', 'older_adults', 'PD'], 'fig_DCBC_curve')
src += figure(['children', 'young_adults', 'NC'], 'fig_DCBC_curve_supp')
pd.DataFrame(src).to_csv(os.path.join(RES, 'fig_source_curves.csv'), index=False)
print('FIG_END')
