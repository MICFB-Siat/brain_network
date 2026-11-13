import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
from pfbn import settings
import os, sys, json, glob, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib as mpl
from matplotlib.gridspec import GridSpec
from scipy.stats import wilcoxon
sys.path[0:0] = [settings.code('evaluation', 'boundary'), settings.code('visualization'), settings.code('evaluation', 'dcbc')]
from gradient_zoom_figs import crops, trim
R = settings.out()
E = os.path.join(R, 'r1_figure')
DIST_RES = os.path.join(R, 'boundary_distance', 'results')
ABLATION = os.path.join(R, 'dynamics_ablation')
RES = os.path.join(E, 'results')
FIG = os.path.join(E, 'figs')
RDR = os.path.join(FIG, 'renders')
HCP_SCR = settings.scratch('validation', 'subjects')
COL = {'CATO': '#555555', 'Micapipe': '#E2B6C1', 'Ours': '#8CC3DE', 'fourth': '#B3B3B3'}
G6 = ['children', 'adolescents', 'young_adults', 'older_adults', 'NC', 'PD']
GN = {'children': 'Children', 'adolescents': 'Adolescents', 'young_adults': 'Young adults', 'older_adults': 'Older adults', 'NC': 'NC', 'PD': 'PD'}
FONT = 'Liberation Sans'
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': [FONT, 'DejaVu Sans'], 'font.weight': 'bold', 'axes.labelweight': 'bold', 'axes.titleweight': 'bold', 'font.size': 6.5, 'axes.linewidth': 1.0, 'xtick.major.width': 1.0, 'ytick.major.width': 1.0, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
STATS = {}

def stars(p):
    return 'n.s.' if p is None or not np.isfinite(p) or p >= 0.05 else '****' if p < 0.0001 else '***' if p < 0.001 else '**' if p < 0.01 else '*'

def clean(ax):
    ax.spines[['top', 'right']].set_visible(False)

def xlabels(ax, labels, rot=45):
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=rot, ha='right' if rot else 'center', rotation_mode='anchor', fontsize=5.6, weight='bold')

def width_at(body, y, pos):
    v = body.get_paths()[0].vertices
    left = v[v[:, 0] <= pos]
    if len(left) < 2:
        return 0.2
    o = np.argsort(left[:, 1])
    return max(pos - np.interp(y, left[o, 1], left[o, 0]), 0.02)

def brackets(ax, pairs, top, rg, step=0.19, fs=7.5):
    for (k, (i, j, p)) in enumerate(pairs):
        y = top + rg * (0.07 + step * k)
        s = stars(p)
        ax.plot([i, j], [y, y], color='black', lw=1.0, solid_capstyle='butt')
        ax.text((i + j) / 2, y + rg * 0.005, s, ha='center', va='bottom', fontsize=fs if s != 'n.s.' else 6, weight='bold')
    return top + rg * (0.07 + step * len(pairs) + 0.06)

def violins(ax, data, cols, labels, pairs=(), ylabel=None, rot=45, step=0.19):
    data = [np.asarray(d, float)[np.isfinite(d)] for d in data]
    for (i, (d, c)) in enumerate(zip(data, cols)):
        b = ax.violinplot(d, positions=[i], widths=0.8, showextrema=False)['bodies'][0]
        b.set_facecolor('white')
        b.set_edgecolor(c)
        b.set_linewidth(1.6)
        b.set_alpha(1)
        for (q, ls) in zip(np.percentile(d, [25, 50, 75]), (':', '--', ':')):
            w = width_at(b, q, i)
            ax.plot([i - w, i + w], [q, q], color='0.15', ls=ls, lw=0.8)
    (lo, hi) = (min(map(np.min, data)), max(map(np.max, data)))
    rg = hi - lo
    ytop = brackets(ax, pairs, hi, rg, step) if pairs else hi + rg * 0.05
    ax.set_ylim(lo - rg * 0.05, ytop)
    ax.set_xlim(-0.6, len(data) - 0.4)
    xlabels(ax, labels, rot)
    clean(ax)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=6.3)

def spin_violins(ax, nulls, obs, cols, labels, pairs=(), ylabel=None, rot=45):
    for (i, (n, o, c)) in enumerate(zip(nulls, obs, cols)):
        b = ax.violinplot(n, positions=[i], widths=0.8, showextrema=False)['bodies'][0]
        b.set_facecolor('#EDEDED')
        b.set_edgecolor('#B5B5B5')
        b.set_linewidth(0.8)
        b.set_alpha(1)
        ax.scatter([i], [o], marker='D', s=26, color=c, edgecolor='black', lw=0.7, zorder=4)
    vals = np.concatenate([np.concatenate(nulls), obs])
    (lo, hi) = (vals.min(), vals.max())
    rg = hi - lo
    ytop = brackets(ax, pairs, hi, rg) if pairs else hi + rg * 0.05
    ax.set_ylim(lo - rg * 0.1, ytop)
    ax.set_xlim(-0.6, len(nulls) - 0.4)
    xlabels(ax, labels, rot)
    clean(ax)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=6.3)

def placeholder(ax, text='pending'):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linestyle('--')
        s.set_edgecolor('0.6')
    ax.text(0.5, 0.5, text, ha='center', va='center', transform=ax.transAxes, fontsize=7, color='0.45')

def letter(fig, ax, s, dx=-0.045, dy=0.012):
    p = ax.get_position()
    fig.text(p.x0 + dx, p.y1 + dy, s, fontsize=12, weight='bold', va='bottom', ha='left')

def brace(fig, axes, text, dy=0.03):
    (p0, p1) = (axes[0].get_position(), axes[-1].get_position())
    y = max(p0.y1, p1.y1) + dy
    (x0, x1) = (p0.x0, p1.x1)
    t = 0.006
    fig.add_artist(mpl.lines.Line2D([x0, x0, x1, x1], [y - t, y, y, y - t], color='black', lw=0.9, transform=fig.transFigure))
    fig.add_artist(mpl.lines.Line2D([(x0 + x1) / 2] * 2, [y, y + t], color='black', lw=0.9, transform=fig.transFigure))
    fig.text((x0 + x1) / 2, y + t + 0.002, text, ha='center', va='bottom', fontsize=7, weight='bold')

def group_axes(fig, spec, n, wspace=0.75):
    sub = spec.subgridspec(1, n, wspace=wspace)
    return [fig.add_subplot(sub[i]) for i in range(n)]

def headers(fig, axs, dy=0.032):
    brace(fig, axs[:4], 'Age groups', dy)
    brace(fig, axs[4:], 'Disease', dy)

def load_distance():
    d = pd.read_csv(os.path.join(DIST_RES, 'dist_f1_groups.csv'))
    n = np.load(os.path.join(DIST_RES, 'dist_f1_nulls.npz'))
    c = pd.read_csv(os.path.join(DIST_RES, 'dcbc_groups_with_cato.csv'))
    return (d, n, c)

def ablation_ok():
    return all((os.path.exists(os.path.join(ABLATION, 'results', f)) for f in ('dcbc_per_subject.csv', 'spin_f1_distance.csv')))

def hcp_crossday():
    keys = {'CATO': 'CATO (Lausanne-219)', 'Micapipe': 'Micapipe', 'Ours': 'PFBN v5|REST1'}
    out = {k: {m: [] for m in keys} for k in ('dcbc', 'retest')}
    pairs = [(a, b) for a in ('rfMRI_REST1_LR', 'rfMRI_REST1_RL') for b in ('rfMRI_REST2_LR', 'rfMRI_REST2_RL')]
    for f in sorted(glob.glob(os.path.join(HCP_SCR, '*.npz'))):
        z = np.load(f)
        for (m, k) in keys.items():
            out['dcbc'][m].append(np.mean([z[f'dcbc|{k}|REST2|{h}'] for h in ('lh', 'rh')]))
            out['retest'][m].append(np.mean([np.arctanh(np.corrcoef(z[f'fc|{k}|{a}'], z[f'fc|{k}|{b}'])[0, 1]) for (a, b) in pairs]))
    return {k: {m: np.array(v) for (m, v) in d.items()} for (k, d) in out.items()}

def wx(a, b):
    return float(wilcoxon(a, b).pvalue)

def panel_a(fig, spec, groups=('adolescents', 'older_adults')):
    bx = crops()
    (r0, r1, c0, c1) = bx['Somatomotor']
    dist = pd.read_csv(os.path.join(DIST_RES, 'dist_f1_groups.csv'))
    atl = [('CATO', 'CATO'), ('Micapipe', 'Micapipe'), ('PFBN v5', 'Our Method')]
    sub = spec.subgridspec(1, 3 * len(groups) + len(groups) - 1, width_ratios=([1, 1, 1, 0.25] * len(groups))[:-1], wspace=0.04)
    axs = []
    for (gi, g) in enumerate(groups):
        for (k, (nm, lab)) in enumerate(atl):
            ax = fig.add_subplot(sub[0, gi * 4 + k])
            axs.append(ax)
            im = plt.imread(os.path.join(RDR, f"grad_{g}_{nm.replace(' ', '_')}.png"))
            ax.imshow(trim(im[r0:r1, c0:c1], 2))
            ax.set_xticks([])
            ax.set_yticks([])
            for s_ in ax.spines.values():
                s_.set_visible(True)
                s_.set_linewidth(1.6)
                s_.set_edgecolor(COL['Ours'] if nm == 'PFBN v5' else COL[nm])
            dm = dist[(dist.group == g) & (dist.roi == 'Somatomotor') & (dist.comparison == nm)].dist_mm.iloc[0]
            ax.set_title(lab, fontsize=6.3, pad=2)
            ax.set_xlabel(f'{dm:.2f} mm', fontsize=6.3, labelpad=2)
    return axs

def trim_off(im, pad=6):
    m = (im[..., :3] < 0.97).any(2)
    r = np.where(m.any(1))[0]
    c = np.where(m.any(0))[0]
    (r0, c0) = (max(r[0] - pad, 0), max(c[0] - pad, 0))
    return (im[r0:r[-1] + pad, c0:c[-1] + pad], r0, c0)

def a_dist(g):
    c = os.path.join(RES, f'custom_{g}_distance.csv')
    d = pd.read_csv(c) if os.path.exists(c) else pd.read_csv(os.path.join(DIST_RES, 'dist_f1_groups.csv'))
    return d[(d.group == g) & (d.roi == 'Somatomotor')].set_index('comparison').dist_mm

def panel_a_v2(fig, spec, groups=('adolescents', 'older_adults')):
    from matplotlib.patches import Rectangle, ConnectionPatch
    bx = crops()
    (r0, r1, c0, c1) = bx['Somatomotor']
    dist = pd.read_csv(os.path.join(DIST_RES, 'dist_f1_groups.csv'))
    atl = [('CATO', 'CATO'), ('Micapipe', 'Micapipe'), ('PFBN v5', 'Our Method')]
    sub = spec.subgridspec(len(groups), 6, width_ratios=[1.75, 0.18, 1, 1, 1, 0.22], wspace=0.05, hspace=0.55)
    axs = []
    for (gi, g) in enumerate(groups):
        full = plt.imread(os.path.join(RDR, f'grad_{g}_noborder.png'))
        (H, W) = full.shape[:2]
        assert r1 <= H // 2 and c1 <= W // 2, 'Somatomotor window not in the left lateral quadrant'
        (im, ro, co) = trim_off(full[:H // 2, :W // 2])
        ab = fig.add_subplot(sub[gi, 0])
        ab.imshow(im)
        ab.axis('off')
        axs.append(ab)
        rect = Rectangle((c0 - co, r0 - ro), c1 - c0, r1 - r0, fill=False, lw=1.4, edgecolor='#1F5FBF')
        ab.add_patch(rect)
        ab.set_title(f'{GN.get(g, g)}: held-out gradient (left hemisphere)', fontsize=6.3, pad=2)
        zs = []
        for (k, (nm, lab)) in enumerate(atl):
            ax = fig.add_subplot(sub[gi, 2 + k])
            axs.append(ax)
            zs.append(ax)
            z = plt.imread(os.path.join(RDR, f"grad_{g}_{nm.replace(' ', '_')}.png"))
            ax.imshow(z[r0:r1, c0:c1])
            ax.set_xticks([])
            ax.set_yticks([])
            for s_ in ax.spines.values():
                s_.set_visible(True)
                s_.set_linewidth(1.6)
                s_.set_edgecolor(COL['Ours'] if nm == 'PFBN v5' else COL[nm])
            ax.set_title(lab, fontsize=6.3, pad=2)
            ax.set_xlabel(f'{a_dist(g)[nm]:.2f} mm', fontsize=6.3, labelpad=2)
        for (ya, yb) in ((r0 - ro, 0), (r1 - ro, r1 - r0)):
            fig.add_artist(ConnectionPatch(xyA=(c1 - co, ya), coordsA=ab.transData, xyB=(0, yb), coordsB=zs[0].transData, color='#1F5FBF', lw=0.7, alpha=0.8))
    cs = fig.add_subplot(sub[:, 5])
    cs.axis('off')
    p = cs.get_position()
    hh = min(p.height * 0.6, 1.3 / fig.get_figheight())
    colorbar(fig, [p.x0 + p.width * 0.25, p.y0 + (p.height - hh) / 2, p.width * 0.3, hh], vertical=True)
    return axs

def panel_b(fig, spec):
    (d, n, _) = load_distance()
    axs = group_axes(fig, spec, 6)
    st = {}
    for (ax, g) in zip(axs, G6):
        w = d[(d.group == g) & (d.roi == 'Whole cortex')].set_index('comparison')
        nulls = [n[f'{g}|Whole cortex|{a}|dist'] for a in ('CATO', 'Micapipe', 'PFBN v5')]
        obs = [w.loc[a, 'dist_mm'] for a in ('CATO', 'Micapipe', 'PFBN v5')]
        (pm, pc) = (w.loc['PFBN v5 - Micapipe', 'p_dist'], w.loc['PFBN v5 - CATO', 'p_dist'])
        spin_violins(ax, nulls, obs, [COL['CATO'], COL['Micapipe'], COL['Ours']], ['CATO', 'Micapipe', 'Our Method'], [(1, 2, pm), (0, 2, pc)], 'Boundary-to-gradient\ndistance (mm)' if g == 'children' else None)
        ax.set_title(GN[g], fontsize=6.5, pad=2)
        st[g] = dict(obs=dict(zip(('CATO', 'Micapipe', 'Ours'), map(float, obs))), p_vs_micapipe=float(pm), p_vs_cato=float(pc))
    STATS['b'] = st
    headers(fig, axs)
    return axs

def panel_c(fig, spec):
    (_, _, c) = load_distance()
    axs = group_axes(fig, spec, 6)
    st = {}
    for (ax, g) in zip(axs, G6):
        t = c[c.group == g].pivot(index='subject', columns='atlas', values='DCBC')
        (own, oth, mi, ca) = (t['Own PFBN v5'], t['Other-group PFBN v5 (mean)'], t['Micapipe'], t['CATO'])
        pairs = [(2, 3, wx(own, oth)), (1, 3, wx(own, mi)), (0, 3, wx(own, ca))]
        violins(ax, [ca, mi, oth, own], [COL['CATO'], COL['Micapipe'], COL['fourth'], COL['Ours']], ['CATO', 'Micapipe', 'Other group', 'Own group'], pairs, 'DCBC (held-out data)' if g == 'children' else None)
        ax.set_title(f'{GN[g]}\nn = {len(t)}', fontsize=6.2, pad=2)
        st[g] = {k: p for (k, (_, _, p)) in zip(('own_vs_other', 'own_vs_micapipe', 'own_vs_cato'), pairs)}
    STATS['c'] = st
    headers(fig, axs)
    return axs

def panel_d(fig, spec, g='adolescents'):
    sub = spec.subgridspec(1, 2, wspace=0.05)
    axs = [fig.add_subplot(sub[i]) for i in range(2)]
    bx = crops()
    (r0, r1, c0, c1) = bx['Somatomotor']
    dd = pd.read_csv(os.path.join(RES, 'panel_d_distance.csv')).set_index(['atlas', 'roi'])
    for (ax, k, t) in zip(axs, ('without', 'with'), ('Without dynamics', 'With dynamics')):
        ax.imshow(trim(plt.imread(os.path.join(RDR, f'grad_{g}_{k}_dynamics.png'))[r0:r1, c0:c1], 2))
        ax.set_xticks([])
        ax.set_yticks([])
        for s_ in ax.spines.values():
            s_.set_visible(True)
            s_.set_linewidth(1.6)
            s_.set_edgecolor(COL['Ours'] if k == 'with' else COL['fourth'])
        ax.set_title(t, fontsize=6.3, pad=2)
        ax.set_xlabel(f"{dd.loc[(t, 'Somatomotor'), 'dist_mm']:.2f} mm", fontsize=6.3, labelpad=2)
    axs[0].set_ylabel(GN[g], fontsize=6.5)
    return axs

def export_panels():
    fig = plt.figure(figsize=(7.2, 1.75))
    gs = GridSpec(1, 1, figure=fig, top=0.74, bottom=0.12, left=0.03, right=0.99)
    A = panel_a(fig, gs[0])
    for (k, g) in enumerate(('adolescents', 'older_adults')):
        brace(fig, A[3 * k:3 * k + 3], GN[g], 0.06)
    (p2, p3) = (A[2].get_position(), A[3].get_position())
    colorbar(fig, [(p2.x1 + p3.x0) / 2 - 0.004, p2.y0 + p2.height * 0.08, 0.007, p2.height * 0.84], vertical=True)
    for e in ('png', 'pdf'):
        fig.savefig(os.path.join(FIG, f'fig_r1_panel_a.{e}'), dpi=600)
    plt.close(fig)
    fig = plt.figure(figsize=(2.9, 1.75))
    gs = GridSpec(1, 1, figure=fig, top=0.86, bottom=0.12, left=0.08, right=0.86)
    D = panel_d(fig, gs[0])
    p = D[1].get_position()
    colorbar(fig, [p.x1 + 0.05, p.y0 + p.height * 0.08, 0.025, p.height * 0.84], vertical=True)
    for e in ('png', 'pdf'):
        fig.savefig(os.path.join(FIG, f'fig_r1_panel_d.{e}'), dpi=600)
    plt.close(fig)

def panel_e(fig, spec):
    axs = group_axes(fig, spec, 6, 0.8)
    st = {}
    if not ablation_ok():
        for (ax, g) in zip(axs, G6):
            placeholder(ax)
            ax.set_title(GN[g], fontsize=6.5, pad=2)
        headers(fig, axs)
        return axs
    d = pd.read_csv(os.path.join(ABLATION, 'results', 'dcbc_per_subject.csv'))
    for (ax, g) in zip(axs, G6):
        t = d[d.group == g].pivot(index='subject', columns='atlas', values='DCBC')
        (wi, wo, mi) = (t['With dynamics'], t['Without dynamics'], t['Micapipe'])
        pairs = [(1, 2, wx(wi, wo)), (0, 2, wx(wi, mi))]
        violins(ax, [mi, wo, wi], [COL['Micapipe'], COL['fourth'], COL['Ours']], ['Micapipe', 'Without dyn.', 'With dyn.'], pairs, 'DCBC (held-out data)' if g == 'children' else None)
        ax.set_title(GN[g], fontsize=6.5, pad=2)
        st[g] = dict(with_vs_without=pairs[0][2], with_vs_micapipe=pairs[1][2], n_with_gt_without=int((wi > wo).sum()), n=len(t))
    STATS['e'] = st
    headers(fig, axs)
    return axs

def panel_f(fig, spec):
    axs = group_axes(fig, spec, 6, 0.8)
    st = {}
    raw = os.path.join(RES, 'dyn_distance_raw.csv')
    if not ablation_ok() or not os.path.exists(raw):
        for (ax, g) in zip(axs, G6):
            placeholder(ax)
            ax.set_title(GN[g], fontsize=6.5, pad=2)
        headers(fig, axs)
        return axs
    r = pd.read_csv(raw).set_index(['group', 'atlas'])
    n = np.load(os.path.join(RES, 'dyn_distance_raw_nulls.npz'))
    sp = pd.read_csv(os.path.join(ABLATION, 'results', 'spin_f1_distance.csv'))
    sp = sp[sp.roi == 'Whole cortex'].set_index(['group', 'comparison'])
    names = ['Micapipe', 'Without dynamics', 'With dynamics']
    for (ax, g) in zip(axs, G6):
        (pw, pm) = (sp.loc[(g, 'With dynamics - Without dynamics'), 'dist_p_one_sided'], sp.loc[(g, 'With dynamics - Micapipe'), 'dist_p_one_sided'])
        spin_violins(ax, [n[f'{g}|{a}'] for a in names], [r.loc[(g, a), 'dist_mm'] for a in names], [COL['Micapipe'], COL['fourth'], COL['Ours']], ['Micapipe', 'Without dyn.', 'With dyn.'], [(1, 2, pw), (0, 2, pm)], 'Boundary-to-gradient\ndistance (mm)' if g == 'children' else None)
        ax.set_title(GN[g], fontsize=6.5, pad=2)
        st[g] = dict(p_with_vs_without=float(pw), p_with_vs_micapipe=float(pm))
    STATS['f'] = st
    headers(fig, axs)
    return axs

def panel_g(fig, spec):
    ax = fig.add_subplot(spec)
    rs = pd.read_csv(os.path.join(R, 'validation', 'results', 'ds_rs_scan_dcbc.csv'), dtype={'r_s': str}).groupby(['subject', 'r_s']).DCBC.mean().unstack()
    order = [c for c in rs.columns if c.startswith('8.5')] + ['15', '20', '30', '45']
    b = order.index('30')
    pairs = [(i, b, wx(rs['30'], rs[c])) for (i, c) in sorted(enumerate(order), key=lambda x: abs(x[0] - b)) if i != b]
    violins(ax, [rs[c] for c in order], [COL['Ours'] if c == '30' else COL['fourth'] for c in order], [c.replace(' (fitted)', '\n(fitted)') for c in order], pairs, 'DCBC (held-out rest)', rot=0, step=0.26)
    ax.set_xlabel('Wave scale r_s (mm)', fontsize=6.3)
    ax.set_title(f'ds005498 (n = {len(rs)})', fontsize=6.5, pad=2)
    STATS['g'] = {order[i]: p for (i, _, p) in pairs}
    return [ax]

def panel_h(fig, spec):
    ax = fig.add_subplot(spec)
    s = pd.read_csv(os.path.join(RES, 'halves_agreement_spin.csv')).set_index('group')
    n = np.load(os.path.join(RES, 'halves_agreement_nulls.npz'))
    for (i, g) in enumerate(G6):
        b = ax.violinplot(n[g], positions=[i], widths=0.8, showextrema=False)['bodies'][0]
        b.set_facecolor('#EDEDED')
        b.set_edgecolor('#B5B5B5')
        b.set_alpha(1)
        ax.scatter([i], [s.loc[g, 'agreement']], marker='D', s=26, color=COL['Ours'], edgecolor='black', lw=0.7, zorder=4)
        ax.text(i, s.loc[g, 'agreement'] + 0.03, stars(s.loc[g, 'p_one_sided']), ha='center', va='bottom', fontsize=7, weight='bold')
    ax.set_ylim(min((n[g].min() for g in G6)) - 0.03, s.agreement.max() + 0.12)
    xlabels(ax, [GN[g] for g in G6])
    clean(ax)
    ax.set_ylabel('Half A vs half B\nborder agreement', fontsize=6.3)
    ax.set_title('Our Method vs spin null', fontsize=6.5, pad=2)
    STATS['h'] = s[['agreement', 'null_mean', 'z', 'p_one_sided']].to_dict('index')
    return [ax]

def panel_ijk(fig, spec_i, spec_j, spec_k):
    t = pd.read_csv(os.path.join(R, 'hcp7t', 'results', 'per_subject.csv'))
    ai = group_axes(fig, spec_i, 2, 0.9)
    st = {}
    for (ax, (m, yl)) in zip(ai, (('DCBC', 'DCBC'), ('fc_retest', 'FC test–retest (z)'))):
        p = t.pivot(index='subject', columns='atlas', values=m)
        (a, b) = (p['Micapipe'], p['PFBN v5'])
        pv = wx(b, a)
        violins(ax, [a, b], [COL['Micapipe'], COL['Ours']], ['Micapipe', 'Our Method'], [(0, 1, pv)], yl)
        st[f'i_{m}'] = dict(p=pv, n_gt=int((b > a).sum()), n=len(p))
    brace(fig, ai, f'HCP 7T, another scanner (n = {len(t.subject.unique())})', 0.012)
    h = hcp_crossday()
    aj = group_axes(fig, spec_j, 2, 0.9)
    for (ax, (m, yl)) in zip(aj, (('dcbc', 'DCBC'), ('retest', 'FC test–retest (z)'))):
        (ca, mi, ou) = (h[m]['CATO'], h[m]['Micapipe'], h[m]['Ours'])
        pairs = [(1, 2, wx(ou, mi)), (0, 2, wx(ou, ca))]
        violins(ax, [ca, mi, ou], [COL['CATO'], COL['Micapipe'], COL['Ours']], ['CATO', 'Micapipe', 'Our Method'], pairs, yl)
        st[f'j_{m}'] = dict(p_vs_micapipe=pairs[0][2], p_vs_cato=pairs[1][2], n_gt_micapipe=int((ou > mi).sum()), n=len(ou))
    brace(fig, aj, f"HCP, another day (n = {len(h['dcbc']['Ours'])})", 0.012)
    g = pd.read_csv(os.path.join(R, 'gsp', 'results', 'per_subject.csv'))
    g = g[(g.data == 'half B (held out)') & (g.metric == 'dcbc')].pivot(index='subject', columns='atlas', values='value')
    (a, b) = (g['Micapipe'], g['PFBN v5'])
    ak = fig.add_subplot(spec_k)
    pv = wx(b, a)
    violins(ak, [a, b], [COL['Micapipe'], COL['Ours']], ['Micapipe', 'Our Method'], [(0, 1, pv)], 'DCBC')
    brace(fig, [ak], f'GSP (n = {len(g)})', 0.012)
    st['k'] = dict(p=pv, n_gt=int((b > a).sum()), n=len(g))
    STATS['ijk'] = st
    return (ai, aj, [ak])

def colorbar(fig, rect, vertical=False):
    cax = fig.add_axes(rect)
    cb = mpl.colorbar.ColorbarBase(cax, cmap=mpl.colormaps['YlOrRd'], norm=mpl.colors.Normalize(0, 100), orientation='vertical' if vertical else 'horizontal')
    cb.ax.tick_params(labelsize=4.5, length=1.5)
    cb.outline.set_linewidth(0.5)
    cb.set_ticks([0, 50, 100])
    cb.set_label('Gradient (pct)' if vertical else 'Held-out local gradient (percentile)', fontsize=5)
    if vertical:
        cb.ax.yaxis.set_label_position('left')
        cb.ax.yaxis.set_ticks_position('right')

def main(variant='v1'):
    v2 = variant == 'a_v2'
    fig = plt.figure(figsize=(7.2, 12.4 if v2 else 11.2))
    gs = GridSpec(6, 1, figure=fig, height_ratios=[2.1 if v2 else 1.0, 1.2, 1.25, 1.2, 1.2, 1.2], hspace=1.25 if not v2 else 1.1, top=0.965 if v2 else 0.955, bottom=0.05, left=0.075, right=0.99)
    if v2:
        A = panel_a_v2(fig, gs[0])
    else:
        A = panel_a(fig, gs[0])
        for (k, g) in enumerate(('adolescents', 'older_adults')):
            brace(fig, A[3 * k:3 * k + 3], GN[g], 0.014)
        (p2, p3) = (A[2].get_position(), A[3].get_position())
        colorbar(fig, [(p2.x1 + p3.x0) / 2 - 0.004, p2.y0 + p2.height * 0.08, 0.007, p2.height * 0.84], vertical=True)
    B = panel_b(fig, gs[1])
    Cc = panel_c(fig, gs[2])
    r3 = gs[3].subgridspec(1, 2, width_ratios=[1.0, 3.0], wspace=0.2)
    D = panel_d(fig, r3[0])
    Ee = panel_e(fig, r3[1])
    r4 = gs[4].subgridspec(1, 2, width_ratios=[3.0, 1.15], wspace=0.25)
    Ff = panel_f(fig, r4[0])
    Gg = panel_g(fig, r4[1])
    r5 = gs[5].subgridspec(1, 4, width_ratios=[1.5, 1.25, 1.55, 0.55], wspace=0.45)
    H = panel_h(fig, r5[0])
    (I, J, K) = panel_ijk(fig, r5[1], r5[2], r5[3])
    for (s, ax, dx) in (('a', A[0], -0.05), ('b', B[0], -0.06), ('c', Cc[0], -0.06), ('d', D[0], -0.04), ('e', Ee[0], -0.06), ('f', Ff[0], -0.06), ('g', Gg[0], -0.06), ('h', H[0], -0.06), ('i', I[0], -0.06), ('j', J[0], -0.06), ('k', K[0], -0.06)):
        letter(fig, ax, s, dx, (0.012 if v2 else 0.03) if s == 'a' else 0.03 if s in 'bcefij' else 0.012)
    if v2:
        d2 = os.path.join(FIG, 'panel_a_v2')
        os.makedirs(d2, exist_ok=True)
        for e in ('png', 'pdf'):
            fig.savefig(os.path.join(d2, f'fig_r1_main_a_v2.{e}'), dpi=600)
        plt.close(fig)
        fig = plt.figure(figsize=(7.2, 3.3))
        gs = GridSpec(1, 1, figure=fig, top=0.93, bottom=0.1, left=0.02, right=0.99)
        panel_a_v2(fig, gs[0])
        for e in ('png', 'pdf'):
            fig.savefig(os.path.join(d2, f'fig_r1_panel_a_v2.{e}'), dpi=600)
        plt.close(fig)
        return
    for e in ('png', 'pdf', 'svg'):
        fig.savefig(os.path.join(FIG, f'fig_r1_main.{e}'), dpi=600)
    plt.close(fig)
    export_panels()
    json.dump(STATS, open(os.path.join(RES, 'panel_stats.json'), 'w'), indent=1, default=float)
    print(json.dumps(STATS, indent=1, default=float)[:4000])

def hcp_clean(name='fig_r1_HCP_clean', rows=('Dorsolateral PFC', 'Early visual')):
    from matplotlib.patches import Rectangle, ConnectionPatch
    bx = crops()
    dist = pd.read_csv(os.path.join(RES, 'custom_HCP_distance.csv')).set_index(['roi', 'comparison']).dist_mm
    atl = [('Micapipe', 'Fixed atlas (Micapipe)'), ('CATO', 'CATO'), ('PFBN v5', 'Our Method')]
    fig = plt.figure(figsize=(7.2, 1.55 * len(rows) + 0.2))
    gs = GridSpec(1, 1, figure=fig, top=1 - 0.32 / (1.55 * len(rows) + 0.2), bottom=0.25 / (1.55 * len(rows) + 0.2), left=0.02, right=0.99)
    sub = gs[0].subgridspec(len(rows), 6, width_ratios=[1.75, 0.18, 1, 1, 1, 0.22], wspace=0.05, hspace=0.55)
    full = plt.imread(os.path.join(RDR, 'grad_HCP_noborder.png'))
    (H, W) = full.shape[:2]
    for (ri, roi) in enumerate(rows):
        (r0, r1, c0, c1) = bx[roi]
        assert c1 <= W // 2, roi
        top = r1 <= H // 2
        q = full[:H // 2, :W // 2] if top else full[H // 2:, :W // 2]
        qo = 0 if top else H // 2
        (im, ro, co) = trim_off(q)
        ro += qo
        ab = fig.add_subplot(sub[ri, 0])
        ab.imshow(im)
        ab.axis('off')
        ab.add_patch(Rectangle((c0 - co, r0 - ro), c1 - c0, r1 - r0, fill=False, lw=1.4, edgecolor='#1F5FBF'))
        ab.set_xlim(min(0, c0 - co) - 6, max(im.shape[1], c1 - co) + 6)
        ab.set_ylim(max(im.shape[0], r1 - ro) + 6, min(0, r0 - ro) - 6)
        ab.set_title(f"HCP {roi}: held-out REST2 gradient (left, {('lateral' if top else 'medial')})", fontsize=6.0, pad=2)
        zs = []
        for (k, (nm, lab)) in enumerate(atl):
            ax = fig.add_subplot(sub[ri, 2 + k])
            zs.append(ax)
            z = plt.imread(os.path.join(RDR, f"grad_HCP_{nm.replace(' ', '_')}.png"))
            ax.imshow(z[r0:r1, c0:c1])
            ax.set_xticks([])
            ax.set_yticks([])
            for s_ in ax.spines.values():
                s_.set_visible(True)
                s_.set_linewidth(1.6)
                s_.set_edgecolor(COL['Ours'] if nm == 'PFBN v5' else COL[nm])
            ax.set_title(lab, fontsize=6.3, pad=2)
            ax.set_xlabel(f'{dist[roi, nm]:.2f} mm', fontsize=6.3, labelpad=2)
        for (ya, yb) in ((r0 - ro, 0), (r1 - ro, r1 - r0)):
            fig.add_artist(ConnectionPatch(xyA=(c1 - co, ya), coordsA=ab.transData, xyB=(0, yb), coordsB=zs[0].transData, color='#1F5FBF', lw=0.7, alpha=0.8))
    cs = fig.add_subplot(sub[:, 5])
    cs.axis('off')
    p = cs.get_position()
    hh = min(p.height * 0.6, 1.3 / fig.get_figheight())
    colorbar(fig, [p.x0 + p.width * 0.25, p.y0 + (p.height - hh) / 2, p.width * 0.3, hh], vertical=True)
    d2 = os.path.join(FIG, 'panel_a_v2')
    os.makedirs(d2, exist_ok=True)
    for e in ('png', 'pdf'):
        fig.savefig(os.path.join(d2, f'{name}.{e}'), dpi=600)
    plt.close(fig)
    print(os.path.join(d2, name))

def draw_row(fig, sub, ri, title, ov_png, zooms, bx_):
    from matplotlib.patches import Rectangle, ConnectionPatch
    (r0, r1, c0, c1) = bx_
    full = plt.imread(ov_png)
    (H, W) = full.shape[:2]
    top = (r0 + r1) / 2 < H // 2
    q = full[:H // 2, :W // 2] if top else full[H // 2:, :W // 2]
    (im, ro, co) = trim_off(q)
    ro += 0 if top else H // 2
    ab = fig.add_subplot(sub[ri, 0])
    ab.imshow(im)
    ab.axis('off')
    ab.add_patch(Rectangle((c0 - co, r0 - ro), c1 - c0, r1 - r0, fill=False, lw=1.4, edgecolor='#1F5FBF'))
    ab.set_xlim(min(0, c0 - co) - 6, max(im.shape[1], c1 - co) + 6)
    ab.set_ylim(max(im.shape[0], r1 - ro) + 6, min(0, r0 - ro) - 6)
    ab.set_title(title, fontsize=6.0, pad=2)
    zs = []
    for (k, (lab, png, ck, dm)) in enumerate(zooms):
        ax = fig.add_subplot(sub[ri, 2 + k])
        zs.append(ax)
        z = plt.imread(png)
        ax.imshow(z[r0:r1, c0:c1])
        ax.set_xticks([])
        ax.set_yticks([])
        for s_ in ax.spines.values():
            s_.set_visible(True)
            s_.set_linewidth(1.6)
            s_.set_edgecolor(COL[ck])
        ax.set_title(lab, fontsize=6.3, pad=2)
        ax.set_xlabel(f'{dm:.2f} mm', fontsize=6.3, labelpad=2)
    for (ya, yb) in ((r0 - ro, 0), (r1 - ro, r1 - r0)):
        fig.add_artist(ConnectionPatch(xyA=(c1 - co, ya), coordsA=ab.transData, xyB=(0, yb), coordsB=zs[0].transData, color='#1F5FBF', lw=0.7, alpha=0.8))

def rows_figure(path, rows, nz=3):
    hgt = 1.55 * len(rows) + 0.2
    fig = plt.figure(figsize=(7.2 if nz == 3 else 5.4, hgt))
    gs = GridSpec(1, 1, figure=fig, top=1 - 0.32 / hgt, bottom=0.25 / hgt, left=0.02, right=0.99)
    sub = gs[0].subgridspec(len(rows), 3 + nz, width_ratios=[1.75, 0.18] + [1] * nz + [0.22], wspace=0.05, hspace=0.55)
    for (ri, (t, ov, zs, b)) in enumerate(rows):
        draw_row(fig, sub, ri, t, ov, zs, b)
    cs = fig.add_subplot(sub[:, 2 + nz])
    cs.axis('off')
    p = cs.get_position()
    hh = min(p.height * 0.6, 1.3 / fig.get_figheight())
    colorbar(fig, [p.x0 + p.width * 0.25, p.y0 + (p.height - hh) / 2, p.width * 0.3, hh], vertical=True)
    for e in ('png',) if path.endswith('gallery_PD_windows') else ('png', 'pdf'):
        fig.savefig(f'{path}.{e}', dpi=600 if not path.endswith('gallery_PD_windows') else 300)
    plt.close(fig)
    print(path)

def pd_rows(which, rank=None):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import r1_panel_search as PS
    df = pd.read_csv(os.path.join(RES, 'pd_window_search.csv'))
    el = df[df.eligible]
    d2 = os.path.join(FIG, 'panel_a_v2')
    os.makedirs(d2, exist_ok=True)
    shp = plt.imread(os.path.join(RDR, 'grad_PD_noborder.png')).shape
    zp = lambda g, a: os.path.join(RDR, f"grad_{g}_{a.replace(' ', '_')}.png")
    if which == 'gallery':
        rows = []
        for (_, r) in el.head(6).iterrows():
            k = int(r['rank'])
            rows.append((f"PD window rank {k} ({r['view']}): Micapipe - Ours = {r['score']:.2f} mm", os.path.join(RDR, 'grad_PD_noborder.png'), [('Micapipe', zp('PD', 'Micapipe'), 'Micapipe', r['dist_Micapipe']), ('Our Method', zp('PD', 'PFBN v5'), 'Ours', r['dist_PFBN v5'])], PS.box(k, shp, r['view'])))
        rows_figure(os.path.join(d2, 'gallery_PD_windows'), rows, nz=2)
        return
    r = el.set_index('rank').loc[rank]
    b = PS.box(rank, shp, r['view'])
    pdrow = (f"PD: held-out gradient (left, {r['view']}; searched window)", os.path.join(RDR, 'grad_PD_noborder.png'), [(lab, zp('PD', a), ck, r[f'dist_{a}']) for (lab, a, ck) in (('CATO', 'CATO', 'CATO'), ('Micapipe', 'Micapipe', 'Micapipe'), ('Our Method', 'PFBN v5', 'Ours'))], b)
    rows_figure(os.path.join(d2, 'fig_r1_panel_a_v2_PD_best'), [pdrow])
    sm = crops()['Somatomotor']
    std = []
    for g in ('adolescents', 'older_adults'):
        dd = a_dist(g)
        std.append((f'{GN[g]}: held-out gradient (left hemisphere)', os.path.join(RDR, f'grad_{g}_noborder.png'), [(lab, zp(g, a), ck, dd[a]) for (lab, a, ck) in (('CATO', 'CATO', 'CATO'), ('Micapipe', 'Micapipe', 'Micapipe'), ('Our Method', 'PFBN v5', 'Ours'))], sm))
    rows_figure(os.path.join(d2, 'fig_r1_panel_a_v2_3rows_PDbest'), std + [pdrow])

def a_v2_groups(name, groups):
    d2 = os.path.join(FIG, 'panel_a_v2')
    os.makedirs(d2, exist_ok=True)
    fig = plt.figure(figsize=(7.2, 1.55 * len(groups) + 0.2))
    gs = GridSpec(1, 1, figure=fig, top=1 - 0.32 / (1.55 * len(groups) + 0.2), bottom=0.25 / (1.55 * len(groups) + 0.2), left=0.02, right=0.99)
    panel_a_v2(fig, gs[0], groups)
    for e in ('png', 'pdf'):
        fig.savefig(os.path.join(d2, f'{name}.{e}'), dpi=600)
    plt.close(fig)
    print(os.path.join(d2, name))
if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'a_v2_groups':
        a_v2_groups(sys.argv[2], sys.argv[3:])
    elif len(sys.argv) > 1 and sys.argv[1] == 'hcp_clean':
        hcp_clean()
    elif len(sys.argv) > 1 and sys.argv[1] == 'pd_gallery':
        pd_rows('gallery')
    elif len(sys.argv) > 1 and sys.argv[1] == 'pd_best':
        pd_rows('best', int(sys.argv[2]))
    else:
        main(sys.argv[1] if len(sys.argv) > 1 else 'v1')
