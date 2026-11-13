import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gradient_zoom_figs import *
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D

def trim_off(im, pad=10):
    m = (im[..., :3] < 0.98).any(2)
    r = np.where(m.any(1))[0]
    c = np.where(m.any(0))[0]
    (r0, c0) = (max(r[0] - pad, 0), max(c[0] - pad, 0))
    return (im[r0:r[-1] + pad, c0:c[-1] + pad], r0, c0)

def dist_violins_small(ax, vd, roi, order, rows):
    data = {m: vd[(vd.atlas == m) & (vd.roi == roi)].dist_mm.values for m in order}
    P.violins(ax, data, order, 'Boundary-to-ridge distance (mm)', None, paired=False, points=False, show_stats=False)
    top = max((np.percentile(v, 99) for v in data.values()))
    ax.set_ylim(-0.3, top * 1.16)
    for (i, m) in enumerate(order):
        r = rows[(rows.atlas == m) & (rows.roi == roi)].iloc[0]
        ax.scatter([i], [r.dist_mm], marker='D', s=12, color=P.COL[m], edgecolor='black', lw=0.4, zorder=4)
        ax.text(i, top * 1.03, f'{r.dist_mm:.2f}\nz={r.dist_z:.1f}', ha='center', va='bottom', fontsize=4.6, linespacing=1.0)

def main():
    df = pd.read_csv(os.path.join(RES, 'roi_f1_distance_HCP.csv'))
    nl = np.load(os.path.join(RES, 'roi_nulls_HCP.npz'))
    vd = pd.read_csv(os.path.join(RES, 'roi_vertex_dist_HCP.csv.gz'))
    rng = json.load(open(os.path.join(RD, 'color_ranges.json')))['HCP_gmap_REST2']
    bx = crops()
    ims = {m: plt.imread(os.path.join(RD, f"gmapREST2_{m.replace(' ', '_')}.png")) for m in ORDER}
    fig = plt.figure(figsize=(7.2, 8.5))
    gs = GridSpec(6, 4, figure=fig, height_ratios=[1.2, 0.86, 0.86, 0.95, 0.14, 0.95], hspace=0.34, wspace=0.35, top=0.97, bottom=0.05, left=0.08, right=0.98)
    axa = []
    for (j, m) in enumerate(('PFBN v5', 'Micapipe')):
        ax = fig.add_subplot(gs[0, 2 * j:2 * j + 2])
        (im, ro, co) = trim_off(ims[m])
        ax.imshow(im)
        ax.axis('off')
        axa.append(ax)
        ax.set_title(f'{m} borders', fontsize=7.5, pad=2)
        for roi in RO:
            (r0, r1, c0, c1) = bx[roi]
            (x0, y0) = (max(c0 - co, 1), max(r0 - ro, 1))
            (x1, y1) = (min(c1 - co, im.shape[1] - 2), min(r1 - ro, im.shape[0] - 2))
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, lw=1.0, edgecolor=mpl.colors.hsv_to_rgb((HUES[roi], 0.9, 0.9))))
        if j == 0:
            P.panel_label(ax, 'a', -0.06, 1.0)
    (p0, p1) = (axa[0].get_position(), axa[1].get_position())
    cax = fig.add_axes([0.528, p0.y0 + p0.height * 0.18, 0.0065, p0.height * 0.62])
    cb = mpl.colorbar.ColorbarBase(cax, cmap=mpl.colormaps['YlOrRd'], norm=mpl.colors.Normalize(rng[0], rng[1]), orientation='vertical')
    cb.ax.tick_params(labelsize=5.5, width=0.5, length=2)
    cb.outline.set_linewidth(0.5)
    cb.set_label('Held-out local RSFC gradient\n(HCP REST2, n = 50)', fontsize=6)
    for (i, m) in enumerate(('PFBN v5', 'Micapipe')):
        for (k, roi) in enumerate(RO):
            (r0, r1, c0, c1) = bx[roi]
            ax = fig.add_subplot(gs[1 + i, k])
            ax.imshow(trim(ims[m][r0:r1, c0:c1], 2))
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(True)
                s.set_linewidth(1.2)
                s.set_edgecolor(mpl.colors.hsv_to_rgb((HUES[roi], 0.9, 0.9)))
            if i == 0:
                ax.set_title(roi, fontsize=7, pad=3)
            if k == 0:
                ax.set_ylabel(m, fontsize=7)
            if i == 0 and k == 0:
                P.panel_label(ax, 'b', -0.3, 1.08)
    cax_list = []
    for (k, roi) in enumerate(RO):
        ax = fig.add_subplot(gs[3, k])
        sub = df[df.roi == roi].set_index('atlas')
        cax_list.append(ax)
        P.null_violins(ax, {m: nl[f'HCP gmap_REST2|{m}|{roi}|F1'] for m in ORDER}, {m: sub.loc[m, 'F1'] for m in ORDER}, ORDER, 'Boundary F1' if k == 0 else '', None, zs={m: sub.loc[m, 'F1_z'] for m in ORDER})
        ax.tick_params(axis='x', labelsize=5.5)
        if k == 0:
            P.panel_label(ax, 'c', -0.5, 1.05)
        ax2 = fig.add_subplot(gs[5, k])
        dist_violins_small(ax2, vd, roi, ORDER, df)
        ax2.tick_params(axis='x', labelsize=5.5)
        if k:
            ax2.set_ylabel('')
        if k == 0:
            P.panel_label(ax2, 'd', -0.5, 1.12)
    lax = fig.add_subplot(gs[4, :])
    lax.axis('off')
    lax.legend(handles=[Patch(facecolor='#D9D9D9', edgecolor='none'), Line2D([], [], marker='D', ls='none', color='#777777', markeredgecolor='black', markersize=4)], labels=['grey violin = 1000 spin null', 'diamond = observed'], loc='center', ncol=2, frameon=False, fontsize=6, handlelength=1.2, columnspacing=1.5)
    P.save(fig, os.path.join(FIG, 'fig_gradient_zoom_main_v2'))
    write_sha([os.path.join(RES, 'roi_f1_distance_HCP.csv'), os.path.join(RES, 'roi_nulls_HCP.npz'), os.path.join(RES, 'roi_vertex_dist_HCP.csv.gz'), os.path.join(RD, 'roi_mask.png'), os.path.join(RD, 'color_ranges.json')] + [os.path.join(RD, f"gmapREST2_{m.replace(' ', '_')}.png") for m in ORDER] + [os.path.abspath(__file__), os.path.join(EXP, 'gradient_zoom_figs.py')], os.path.join(RES, 'fig_main_v2_inputs_sha256.txt'))
    log('FIG_V2_END')
if __name__ == '__main__':
    main()
