import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
from pfbn import settings
import os, json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
R = settings.out()
EXP = os.path.join(R, 'boundary_distance')
RES = os.path.join(EXP, 'results')
FIG = os.path.join(EXP, 'figs')
GROUPS = [('age_boundary', 'children', 'Children'), ('age_boundary', 'adolescents', 'Adolescents'), ('age_boundary', 'young_adults', 'Young adults'), ('age_boundary', 'older_adults', 'Older adults'), ('pd_boundary', 'NC', 'NC'), ('pd_boundary', 'PD', 'PD')]
COL = {'CATO': '#6E6E6E', 'mica': '#D9A5B3', 'other': '#A0A0A0', 'own': '#5DA9D6'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 6.5, 'axes.linewidth': 0.8, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
stars = lambda p: '****' if p < 0.0001 else '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'

def violins(ax, data, cols, labels, pairs):
    for (i, (v, c)) in enumerate(zip(data, cols)):
        b = ax.violinplot(v, positions=[i], widths=0.78, showextrema=False)['bodies'][0]
        b.set_facecolor('white')
        b.set_edgecolor(c)
        b.set_linewidth(1.4)
        b.set_alpha(1)
        (q1, q2, q3) = np.percentile(v, [25, 50, 75])
        for (q, ls) in ((q1, ':'), (q2, '--'), (q3, ':')):
            ax.plot([i - 0.2, i + 0.2], [q, q], color='0.2', ls=ls, lw=0.8)
    (lo, hi) = (min(map(np.min, data)), max(map(np.max, data)))
    rg = hi - lo
    for (k, (i, j, p)) in enumerate(pairs):
        y = hi + rg * (0.06 + 0.12 * k)
        ax.plot([i, i, j, j], [y - rg * 0.02, y, y, y - rg * 0.02], color='k', lw=0.7)
        ax.text((i + j) / 2, y, stars(p), ha='center', va='bottom', fontsize=6.5)
    ax.set_ylim(lo - rg * 0.06, hi + rg * (0.12 + 0.12 * len(pairs)))
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels(labels, fontsize=5.6, rotation=45, ha='right', rotation_mode='anchor')
    ax.spines[['top', 'right']].set_visible(False)

def save(f, name):
    for e in ('png', 'pdf', 'svg'):
        f.savefig(os.path.join(FIG, f'{name}.{e}'), dpi=600, bbox_inches='tight')
    plt.close(f)

def main():
    (rows, st) = ([], {})
    for (exp, g, gt) in GROUPS:
        d = pd.read_csv(os.path.join(R, exp, 'results', 'dcbc_per_subject.csv'), dtype={'subject': str})
        d = d[d.group == g]
        own = d[d.own == True].set_index('subject').DCBC
        mic = d[d.atlas == 'Micapipe'].set_index('subject').DCBC
        oth = d[(d.own == False) & (d.atlas != 'Micapipe')].groupby('subject').DCBC.mean()
        c = pd.read_csv(os.path.join(RES, f'dcbc_cato_{g}.csv'), dtype={'subject': str})
        cat = c[c.atlas == 'CATO'].set_index('subject').DCBC
        mic2 = c[c.atlas == 'Micapipe'].set_index('subject').DCBC
        s = own.index.intersection(cat.index)
        t = pd.DataFrame({'subject': s, 'group': g, 'CATO': cat[s].values, 'Micapipe': mic[s].values, 'Other-group PFBN v5 (mean)': oth[s].values, 'Own PFBN v5': own[s].values})
        rows.append(t)
        st[g] = dict(n=len(s), micapipe_reproduction_max_abs_diff=float(np.abs(mic[s].values - mic2[s].values).max()), **{f'median|{k}': float(t[k].median()) for k in ('CATO', 'Micapipe', 'Other-group PFBN v5 (mean)', 'Own PFBN v5')}, p_own_vs_cato=float(wilcoxon(t['Own PFBN v5'], t['CATO']).pvalue), n_own_gt_cato=int((t['Own PFBN v5'] > t['CATO']).sum()), p_mica_vs_cato=float(wilcoxon(t['Micapipe'], t['CATO']).pvalue), n_mica_gt_cato=int((t['Micapipe'] > t['CATO']).sum()), p_own_vs_mica=float(wilcoxon(t['Own PFBN v5'], t['Micapipe']).pvalue), p_own_vs_other=float(wilcoxon(t['Own PFBN v5'], t['Other-group PFBN v5 (mean)']).pvalue))
    long = pd.concat(rows).melt(id_vars=['subject', 'group'], var_name='atlas', value_name='DCBC')
    long.to_csv(os.path.join(RES, 'dcbc_groups_with_cato.csv'), index=False)
    json.dump(st, open(os.path.join(RES, 'dcbc_cato_stats.json'), 'w'), indent=1)
    (f, axs) = plt.subplots(1, 6, figsize=(7.2, 2.4))
    for (ax, (t, (_, g, gt))) in zip(axs, zip(rows, GROUPS)):
        violins(ax, [t['CATO'], t['Micapipe'], t['Other-group PFBN v5 (mean)'], t['Own PFBN v5']], [COL['CATO'], COL['mica'], COL['other'], COL['own']], ['CATO', 'Micapipe', 'Other group', 'Own group'], [(2, 3, st[g]['p_own_vs_other']), (1, 3, st[g]['p_own_vs_mica']), (0, 3, st[g]['p_own_vs_cato'])])
        ax.set_title(f'{gt}\nn = {len(t)}', fontsize=6.3, pad=2)
    axs[0].set_ylabel('DCBC (held-out half)')
    f.tight_layout(w_pad=0.6)
    save(f, 'fig_dcbc_groups')
    d = pd.read_csv(os.path.join(RES, 'dist_f1_groups.csv'))
    nl = np.load(os.path.join(RES, 'dist_f1_nulls.npz'))
    (f, axs) = plt.subplots(2, 6, figsize=(7.2, 3.6))
    for (row, ref) in enumerate(('Micapipe', 'CATO')):
        for (k, (_, g, gt)) in enumerate(GROUPS):
            ax = axs[row, k]
            r = d[(d.group == g) & (d.roi == 'Whole cortex') & (d.comparison == f'PFBN v5 - {ref}')].iloc[0]
            nd = nl[f'{g}|Whole cortex|PFBN v5 - {ref}|dist']
            b = ax.violinplot(nd, positions=[0], widths=0.8, showextrema=False)['bodies'][0]
            b.set_facecolor('#E3E3E3')
            b.set_edgecolor('#9E9E9E')
            b.set_alpha(1)
            ax.scatter([0], [r.dist_mm], marker='D', s=24, color=COL['own'], edgecolor='black', lw=0.6, zorder=3)
            ax.axhline(0, color='0.5', lw=0.5, ls='--')
            ax.set_xticks([])
            ax.spines[['top', 'right']].set_visible(False)
            ax.set_title(f'{gt}\n{r.dist_mm:+.2f} mm, P = {r.p_dist:.3f}', fontsize=5.8, pad=2)
            if k == 0:
                ax.set_ylabel(f'Δ distance to ridge (mm)\nPFBN v5 − {ref}', fontsize=6)
    f.tight_layout(w_pad=0.5, h_pad=1.0)
    save(f, 'fig_distance_groups')
    for (g, v) in st.items():
        print(g, {k: round(x, 4) if isinstance(x, float) else x for (k, x) in v.items()})
if __name__ == '__main__':
    main()
