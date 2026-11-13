import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
from pfbn import settings
import os, sys, json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib as mpl
from scipy.stats import wilcoxon
sys.path[0:0] = [settings.code('evaluation', 'boundary'), settings.code('visualization'), settings.code('evaluation', 'dcbc')]
E = settings.out('dynamics_ablation')
RES = os.path.join(E, 'results')
FIG = os.path.join(E, 'figs')
RDR = os.path.join(FIG, 'renders')
COL = {'Micapipe': '#D9A5B3', 'Without dynamics': '#A0A0A0', 'With dynamics': '#5DA9D6'}
G = ['children', 'adolescents', 'young_adults', 'older_adults', 'NC', 'PD']
GN = {'children': 'Children', 'adolescents': 'Adolescents', 'young_adults': 'Young adults', 'older_adults': 'Older adults', 'NC': 'NC', 'PD': 'PD'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 6.5, 'axes.linewidth': 0.8, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
stars = lambda p: '****' if p < 0.0001 else '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'

def save(f, n):
    for e in ('png', 'pdf', 'svg'):
        f.savefig(os.path.join(FIG, f'{n}.{e}'), dpi=600, bbox_inches='tight')
    plt.close(f)

def fig_dcbc():
    dc = pd.read_csv(os.path.join(RES, 'dcbc_per_subject.csv'))
    names = list(COL)
    (f, axs) = plt.subplots(2, 3, figsize=(5.6, 4.8))
    for (ax, g) in zip(axs.ravel(), G):
        t = dc[dc.group == g].pivot(index='subject', columns='atlas', values='DCBC')
        data = [t[n].values for n in names]
        for (i, (v, n)) in enumerate(zip(data, names)):
            b = ax.violinplot(v, positions=[i], widths=0.78, showextrema=False)['bodies'][0]
            b.set_facecolor('white')
            b.set_edgecolor(COL[n])
            b.set_linewidth(1.4)
            b.set_alpha(1)
            (q1, q2, q3) = np.percentile(v, [25, 50, 75])
            for (q, ls) in ((q1, ':'), (q2, '--'), (q3, ':')):
                ax.plot([i - 0.2, i + 0.2], [q, q], color='0.2', ls=ls, lw=0.8)
        for (a, b_) in zip(data[1], data[2]):
            ax.plot([1.08, 1.92], [a, b_], color='0.75', lw=0.3, zorder=0)
        (lo, hi) = (min(map(np.min, data)), max(map(np.max, data)))
        rg = hi - lo
        for (k, (i, j)) in enumerate(((1, 2), (0, 2))):
            p = wilcoxon(data[j], data[i]).pvalue
            y = hi + rg * (0.07 + 0.15 * k)
            ax.plot([i, i, j, j], [y - rg * 0.025, y, y, y - rg * 0.025], color='k', lw=0.7)
            ax.text((i + j) / 2, y, stars(p), ha='center', va='bottom', fontsize=6.5)
        ax.set_ylim(lo - rg * 0.06, hi + rg * 0.45)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['Micapipe', 'Without\ndynamics', 'With\ndynamics'], fontsize=5.6)
        ax.set_title(f"{GN[g]} (n = {len(t)})\nWith > Without: {int((t['With dynamics'] > t['Without dynamics']).sum())}/{len(t)}", fontsize=6.3, pad=2)
        ax.spines[['top', 'right']].set_visible(False)
    axs[0, 0].set_ylabel('DCBC (held-out half B)')
    axs[1, 0].set_ylabel('DCBC (held-out half B)')
    f.tight_layout(w_pad=1.0, h_pad=1.2)
    save(f, 'fig_ablation_dcbc')

def fig_spin():
    sp = pd.read_csv(os.path.join(RES, 'spin_f1_distance.csv'))
    nl = np.load(os.path.join(RES, 'spin_diff_nulls.npz'))
    (f, axs) = plt.subplots(2, 6, figsize=(7.2, 3.6))
    for (row, (met, yl, pc)) in enumerate((('dist', 'Δ border-to-ridge distance (mm)\nWith − Without', 'dist'), ('F1', 'Δ Boundary F1\nWith − Without', 'F1'))):
        for (ax, g) in zip(axs[row], G):
            r = sp[(sp.group == g) & (sp.roi == 'Whole cortex') & (sp.comparison == 'With dynamics - Without dynamics')].iloc[0]
            nd = nl[f'{g}|Whole cortex|With dynamics - Without dynamics|{met}']
            obs = r['dist_mm' if met == 'dist' else 'F1']
            b = ax.violinplot(nd, positions=[0], widths=0.8, showextrema=False)['bodies'][0]
            b.set_facecolor('#E3E3E3')
            b.set_edgecolor('#9E9E9E')
            b.set_alpha(1)
            ax.scatter([0], [obs], marker='D', s=24, color=COL['With dynamics'], edgecolor='black', lw=0.6, zorder=3)
            ax.axhline(0, color='0.5', lw=0.5, ls='--')
            ax.set_xticks([])
            ax.spines[['top', 'right']].set_visible(False)
            ax.set_title(f"{GN[g]}\n{obs:+.3f}, P = {r[f'{pc}_p_one_sided']:.3f}", fontsize=5.8, pad=2)
            if g == G[0]:
                ax.set_ylabel(yl, fontsize=6)
    f.text(0.5, -0.01, 'Whole cortex vs held-out half-B gradient ridges; grey = 1000 spins (both atlases rotated identically), ◆ = observed; one-sided P (With better)', ha='center', fontsize=5.5)
    f.tight_layout(w_pad=0.8)
    save(f, 'fig_ablation_spin')

def fig_render():
    from gradient_zoom_figs import crops, trim, HUES
    bx = crops()
    W_ = ['Somatomotor', 'Dorsolateral PFC']
    (f, axs) = plt.subplots(6, 4, figsize=(5.2, 7.6))
    for (i, g) in enumerate(G):
        for (j, roi) in enumerate(W_):
            for (k, m) in enumerate(('without', 'with')):
                ax = axs[i, j * 2 + k]
                im = plt.imread(os.path.join(RDR, f'grad_{g}_{m}.png'))
                (r0, r1, c0, c1) = bx[roi]
                ax.imshow(trim(im[r0:r1, c0:c1], 2))
                ax.set_xticks([])
                ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(True)
                    s.set_linewidth(0.8)
                    s.set_edgecolor(mpl.colors.hsv_to_rgb((HUES[roi], 0.9, 0.9)))
                if i == 0:
                    ax.set_title(f"{roi}\n{('Without' if m == 'without' else 'With')} dynamics", fontsize=6)
                if j == 0 and k == 0:
                    ax.set_ylabel(GN[g], fontsize=6.5)
    cax = f.add_axes([0.3, 0.035, 0.4, 0.008])
    cb = mpl.colorbar.ColorbarBase(cax, cmap=mpl.colormaps['YlOrRd'], norm=mpl.colors.Normalize(0, 100), orientation='horizontal')
    cb.ax.tick_params(labelsize=5, length=1.5)
    cb.outline.set_linewidth(0.4)
    cb.set_label('Held-out (half B) local gradient, percentile within group; black = parcel borders', fontsize=5.5)
    f.subplots_adjust(left=0.06, right=0.99, top=0.95, bottom=0.07, wspace=0.04, hspace=0.06)
    save(f, 'fig_ablation_render')
if __name__ == '__main__':
    for s in sys.argv[1:] or ['dcbc', 'spin', 'render']:
        {'dcbc': fig_dcbc, 'spin': fig_spin, 'render': fig_render}[s]()
