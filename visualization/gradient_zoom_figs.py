import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gradient_tools import *
import pandas as pd, matplotlib.pyplot as plt, matplotlib as mpl
from matplotlib.gridspec import GridSpec
EXP = os.path.join(ROOT, 'gradient_zoom')
RES = os.path.join(EXP, 'results')
FIG = os.path.join(EXP, 'figs')
RD = os.path.join(FIG, 'renders')
PANELS = os.path.join(FIG, 'panels')
os.makedirs(PANELS, exist_ok=True)
HUES = {'Somatomotor': 0.0, 'Early visual': 0.333, 'Temporo-parietal': 0.667, 'Dorsolateral PFC': 0.1667}
ORDER = ['Micapipe', 'PFBN v0', 'PFBN v5']
RO = list(ROIS)
P.COL.update({'PFBN v5 (A)': '#D1495B', 'PFBN v5 (B)': '#E58E99'})

def trim(im, pad=10):
    m = (im[..., :3] < 0.98).any(2)
    r = np.where(m.any(1))[0]
    c = np.where(m.any(0))[0]
    return im[max(r[0] - pad, 0):r[-1] + pad, max(c[0] - pad, 0):c[-1] + pad]

def crops():
    boxes = crop_boxes(os.path.join(RD, 'roi_mask.png'), 4, HUES)
    im = plt.imread(os.path.join(RD, 'gmapREST2_PFBN_v5.png'))
    return {k: pad_box(b, im.shape, 0.18) for (k, b) in boxes.items()}

def colorbar(ax, cmap, lo, hi, label):
    cb = mpl.colorbar.ColorbarBase(ax, cmap=mpl.cm.get_cmap(cmap), norm=mpl.colors.Normalize(lo, hi), orientation='horizontal')
    cb.set_label(label, fontsize=6.5)
    cb.ax.tick_params(labelsize=6, width=0.5, length=2)
    cb.outline.set_linewidth(0.5)

def dist_violins(ax, vd, roi, order, rows):
    data = {m: vd[(vd.atlas == m) & (vd.roi == roi)].dist_mm.values for m in order}
    P.violins(ax, data, order, 'Boundary-to-ridge distance (mm)', None, paired=False, points=False, show_stats=False)
    top = max((np.percentile(v, 99) for v in data.values()))
    ax.set_ylim(-0.3, top * 1.12)
    for (i, m) in enumerate(order):
        r = rows[(rows.atlas == m) & (rows.roi == roi)].iloc[0]
        ax.scatter([i], [r.dist_mm], marker='D', s=14, color=P.COL[m], edgecolor='black', lw=0.4, zorder=4)
        ax.text(i, top * 1.04, f'{r.dist_mm:.2f}\nz={r.dist_z:.1f}', ha='center', va='bottom', fontsize=5.5)

def main_fig():
    df = pd.read_csv(os.path.join(RES, 'roi_f1_distance_HCP.csv'))
    nl = np.load(os.path.join(RES, 'roi_nulls_HCP.npz'))
    vd = pd.read_csv(os.path.join(RES, 'roi_vertex_dist_HCP.csv.gz'))
    rng = json.load(open(os.path.join(RD, 'color_ranges.json')))['HCP_gmap_REST2']
    bx = crops()
    ims = {m: plt.imread(os.path.join(RD, f"gmapREST2_{m.replace(' ', '_')}.png")) for m in ORDER}
    fig = plt.figure(figsize=(7.2, 9.4))
    gs = GridSpec(6, 4, figure=fig, height_ratios=[1.25, 0.16, 0.95, 0.95, 0.95, 0.95], hspace=0.42, wspace=0.35)
    for (j, m) in enumerate(('PFBN v5', 'Micapipe')):
        ax = fig.add_subplot(gs[0, 2 * j:2 * j + 2])
        ax.imshow(trim(ims[m]))
        ax.axis('off')
        ax.set_title(f'{m} borders', fontsize=7.5)
        if j == 0:
            P.panel_label(ax, 'a', -0.06, 1.0)
    cax = fig.add_subplot(gs[1, 1:3])
    pos = cax.get_position()
    cax.set_position([pos.x0 + pos.width * 0.15, pos.y0 + pos.height * 0.55, pos.width * 0.7, pos.height * 0.25])
    colorbar(cax, 'YlOrRd', rng[0], rng[1], 'Held-out local RSFC gradient (HCP REST2, n = 50)')
    for (i, m) in enumerate(('PFBN v5', 'Micapipe')):
        for (k, roi) in enumerate(RO):
            (r0, r1, c0, c1) = bx[roi]
            ax = fig.add_subplot(gs[2 + i, k])
            ax.imshow(trim(ims[m][r0:r1, c0:c1], 2))
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(True)
                s.set_linewidth(1.2)
                s.set_edgecolor(mpl.colors.hsv_to_rgb((HUES[roi], 0.9, 0.9)))
            if i == 0:
                ax.set_title(roi, fontsize=7)
            if k == 0:
                ax.set_ylabel(m, fontsize=7)
            if i == 0 and k == 0:
                P.panel_label(ax, 'b', -0.25, 1.05)
    for (k, roi) in enumerate(RO):
        ax = fig.add_subplot(gs[4, k])
        sub = df[df.roi == roi].set_index('atlas')
        P.null_violins(ax, {m: nl[f'HCP gmap_REST2|{m}|{roi}|F1'] for m in ORDER}, {m: sub.loc[m, 'F1'] for m in ORDER}, ORDER, 'Boundary F1' if k == 0 else '', None, zs={m: sub.loc[m, 'F1_z'] for m in ORDER})
        ax.tick_params(axis='x', labelsize=5.5)
        if k == 0:
            P.panel_label(ax, 'c', -0.5, 1.05)
        ax2 = fig.add_subplot(gs[5, k])
        dist_violins(ax2, vd, roi, ORDER, df)
        ax2.tick_params(axis='x', labelsize=5.5)
        if k:
            ax2.set_ylabel('')
        if k == 0:
            P.panel_label(ax2, 'd', -0.5, 1.12)
    P.save(fig, os.path.join(FIG, 'fig_gradient_zoom_main'))
    for m in ('PFBN v5', 'Micapipe', 'PFBN v0'):
        (f, ax) = plt.subplots(figsize=(3.6, 2.4))
        ax.imshow(trim(ims[m]))
        ax.axis('off')
        ax.set_title(f'{m} borders, HCP REST2 local gradient')
        P.save(f, os.path.join(PANELS, f"whole_{m.replace(' ', '_')}"))
    for roi in RO:
        (r0, r1, c0, c1) = bx[roi]
        (f, axs) = plt.subplots(1, 3, figsize=(5.4, 1.9))
        for (ax, m) in zip(axs, ORDER):
            ax.imshow(trim(ims[m][r0:r1, c0:c1], 2))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(m)
        f.suptitle(roi, fontsize=7.5)
        P.save(f, os.path.join(PANELS, f"inset_{roi.replace(' ', '_')}"))
    (f, axs) = plt.subplots(2, 5, figsize=(7.2, 3.6))
    for (k, roi) in enumerate(RO + ['Whole cortex']):
        sub = df[df.roi == roi].set_index('atlas')
        P.null_violins(axs[0, k], {m: nl[f'HCP gmap_REST2|{m}|{roi}|F1'] for m in ORDER}, {m: sub.loc[m, 'F1'] for m in ORDER}, ORDER, 'Boundary F1' if k == 0 else '', roi, zs={m: sub.loc[m, 'F1_z'] for m in ORDER})
        dist_violins(axs[1, k], vd, roi, ORDER, df)
        if k:
            axs[1, k].set_ylabel('')
    P.save(f, os.path.join(PANELS, 'roi_quant_all'))
    return df

def honey_fig():
    hc = pd.read_csv(os.path.join(RES, 'honeycomb_canonical_borders.csv'))
    nl = np.load(os.path.join(RES, 'honeycomb_nulls.npz'))
    from gradient_zoom import GM_HCP
    maps = [('HCP gmap_REST2', GM_HCP, 'npy'), ('Lifespan cohort gmap', os.path.join(ATL, 'lifespan_all/results/gmap_{h}.npy'), 'npy'), ('tyf young_adults iter0', os.path.join(TYF, 'young_adults_iter0_{H}.func.gii'), 'gii')]
    maps += [(f'tyf {g} iter8', os.path.join(TYF, f'{g}_iter8_' + '{H}.func.gii'), 'gii') for g in ('children', 'adolescents', 'young_adults', 'older_adults', 'age_independent')]
    short = {t: t.replace('tyf ', 'MSM ').replace('_adults', '').replace('age_independent', 'age-indep.').replace('HCP gmap_REST2', 'HCP REST2\ngradient').replace('Lifespan cohort gmap', 'Lifespan\ngradient').replace(' iter', '\niter') for (t, _, _) in maps}
    vals = {}
    for (t, p, kind) in maps:
        v = []
        for h in C.HEMIS:
            (g, nz) = load_gmap(p, h) if kind == 'npy' else load_func_cortex(p.format(H=C.HL[h]), h)
            x = g[nz]
            top = x[x >= np.percentile(x, THR)]
            v.append(top / top.mean())
        vals[t] = np.concatenate(v)
    for (t, _, _) in maps:
        P.COL.setdefault(t, '#D1495B' if 'iter8' in t else '#E8A33D' if 'iter0' in t else '#5B8FD0')
    fig = plt.figure(figsize=(7.2, 6.0))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[0.95, 1, 1.1], hspace=0.6, wspace=0.25)
    for (j, (f_, ttl)) in enumerate((('canonical_tyf_young_adults_iter8.png', 'MSM group boundary map (young adults, iter 8)'), ('canonical_HCP_gmap_REST2.png', 'HCP REST2 local gradient'))):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(trim(plt.imread(os.path.join(RD, f_))))
        ax.axis('off')
        ax.set_title(ttl + '\nwhite: HCP-MMP V1|V2 and 4|3a+3b borders', fontsize=6.5)
        if j == 0:
            P.panel_label(ax, 'a', -0.25, 1.02)
    ax = fig.add_subplot(gs[1, :])
    order = [t for (t, _, _) in maps]
    for (i, t) in enumerate(order):
        v = ax.violinplot(vals[t], positions=[i], widths=0.7, showextrema=False)
        for b in v['bodies']:
            b.set_facecolor(P.COL[t])
            b.set_alpha(0.4)
            b.set_edgecolor('none')
        ax.boxplot(vals[t], positions=[i], widths=0.15, showfliers=False, medianprops=dict(color='k', lw=0.8), boxprops=dict(lw=0.6), whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6))
    yl = ax.get_ylim()
    for (i, t) in enumerate(order):
        r = hc[hc['map'] == t].iloc[0]
        ax.text(i, yl[1], f'CV={r.ridge_cv:.2f}\nr={r.r_with_HCP_gmap_REST2:.2f}', ha='center', va='bottom', fontsize=5.5)
    ax.set_ylim(yl[0], yl[1] * 1.15)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([short[t] for t in order], fontsize=5.8)
    ax.set_ylabel('Ridge strength / mean\n(top 15 % vertices)')
    ax.set_title('Ridge-strength uniformity (CV) and similarity to the HCP REST2 gradient map (r)')
    P.panel_label(ax, 'b', -0.07, 1.05)
    for (j, b) in enumerate(('V1|V2', 'All HCP-MMP borders')):
        ax = fig.add_subplot(gs[2, j])
        sub = hc[hc.border == b].set_index('map')
        nn = {t: nl[f'{t}|{b}'] for t in order}
        ob = {t: sub.loc[t, 'AUC'] for t in order}
        for (i, t) in enumerate(order):
            v = ax.violinplot(nn[t][np.isfinite(nn[t])], positions=[i], widths=0.7, showextrema=False)
            for bb in v['bodies']:
                bb.set_facecolor('#D9D9D9')
                bb.set_alpha(1)
                bb.set_edgecolor('none')
            ax.scatter([i], [ob[t]], marker='D', s=16, color=P.COL[t], edgecolor='k', lw=0.4, zorder=3)
            ax.text(i, ob[t], f" {sub.loc[t, 'AUC_z']:.1f}", fontsize=5, va='center')
        ax.axhline(0.5, color='k', lw=0.4, ls=':')
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([short[t].replace(chr(10), ' ') for t in order], fontsize=5, rotation=40, ha='right')
        ax.set_ylabel('AUC (border vs rest)')
        ax.set_title(f'{b} border strength vs spin null (z)')
        if j == 0:
            P.panel_label(ax, 'c', -0.15, 1.05)
    P.save(fig, os.path.join(FIG, 'fig_gradient_zoom_honeycomb'))

def lifespan_fig():
    df = pd.read_csv(os.path.join(RES, 'lifespan_vs_tyf_maps.csv'))
    nl = np.load(os.path.join(RES, 'lifespan_vs_tyf_nulls.npz'))
    groups = ['children', 'adolescents', 'young_adults', 'older_adults', 'age_independent']
    fig = plt.figure(figsize=(7.2, 4.0))
    gs = GridSpec(2, 5, figure=fig, height_ratios=[0.9, 1], hspace=0.45, wspace=0.75)
    for (j, m) in enumerate(('PFBN_v5_A', 'Micapipe')):
        ax = fig.add_subplot(gs[0, 2 * j:2 * j + 2])
        ax.imshow(trim(plt.imread(os.path.join(RD, f'tyf_young_iter8_{m}.png'))))
        ax.axis('off')
        ax.set_title(f"{m.replace('_', ' ')} borders on MSM young-adult map", fontsize=6.5)
        if j == 0:
            P.panel_label(ax, 'a', -0.04, 1.02)
    for (k, g) in enumerate(groups):
        ax = fig.add_subplot(gs[1, k])
        sub = df[df.group == g].set_index('atlas')
        order = list(sub.index)
        P.null_violins(ax, {m: nl[f'tyf iter8|{g}|{m}|Whole cortex|F1'] for m in order}, {m: sub.loc[m, 'F1'] for m in order}, order, 'Boundary F1' if k == 0 else '', g.replace('_', ' '), zs={m: sub.loc[m, 'F1_z'] for m in order})
        ax.tick_params(axis='x', labelsize=5, rotation=45)
        if k == 0:
            P.panel_label(ax, 'b', -0.4, 1.05)
    P.save(fig, os.path.join(FIG, 'fig_gradient_zoom_lifespan_tyf'))
if __name__ == '__main__':
    df = main_fig()
    honey_fig()
    lifespan_fig()
    write_sha([os.path.join(RES, f) for f in sorted(os.listdir(RES)) if not f.startswith('inputs_sha')] + [os.path.join(RD, f) for f in sorted(os.listdir(RD))] + [os.path.abspath(__file__)], os.path.join(RES, 'figs_inputs_sha256.txt'))
    log('FIGS_END')
