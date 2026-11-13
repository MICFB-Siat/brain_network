import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from boundary_tools import *
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
OUT = os.path.join(EXP, 'R1_age_disease')
RDR = os.path.join(OUT, 'renders')
RES = os.path.join(OUT, 'results')
FIG = os.path.join(OUT, 'figs')
for d in (RDR, RES, FIG):
    os.makedirs(d, exist_ok=True)
GROUPS = {'age': [('children', 'life_children'), ('adolescents', 'life_adolescents'), ('young_adults', 'life_young_adults'), ('older_adults', 'life_older_adults')], 'disease': [('NC', 'pd_NC'), ('PD', 'pd_PD')]}
GN = {'children': 'Children', 'adolescents': 'Adolescents', 'young_adults': 'Young adults', 'older_adults': 'Older adults', 'NC': 'NC', 'PD': 'PD'}
WIN = ['Somatomotor', 'Dorsolateral PFC']
GPATH = lambda coh: os.path.join(ATL, f'{coh}_B', 'results', 'gmap_{h}.npy')

def jobs():
    for (study, gl) in GROUPS.items():
        for (g, coh) in gl:
            for kind in ('v5', 'v5m'):
                yield (study, g, coh, kind)

def render_job(a):
    (study, g, coh, kind, rng) = a
    os.environ['OMP_NUM_THREADS'] = '2'
    mica = micapipe()
    oursA = atlas_v5(f'{coh}_A', kind)
    oursB = atlas_v5(f'{coh}_B', kind)
    gm = gmap_cortex(GPATH(coh))
    tag = f'{study}_{g}_{kind}'
    only = os.environ.get('REFERENCE_ONLY', '')
    if not only:
        render_parcels_with_ref(os.path.join(RDR, f'parc_{tag}.png'), oursA, mica)
    cat = {h: border_categories(C.hemi_lab(oursA, h), C.hemi_lab(mica, h), C.edges(h), C.adjacency(h)) for h in C.HEMIS}
    L = lambda k: {h: cat[h][k] for h in C.HEMIS}
    allg = np.concatenate([gm[h][np.isfinite(gm[h])] for h in C.HEMIS])
    pct = {}
    for h in C.HEMIS:
        x = np.full(C.HN[h], np.nan)
        ok = np.isfinite(gm[h])
        x[ok] = np.searchsorted(np.sort(allg), gm[h][ok]) / len(allg) * 100
        pct[h] = x
    render_layers(os.path.join(RDR, f'grad_{tag}.png'), [(L('shared'), RGB['shared']), (L('only2'), RGB['mica']), (L('only1'), RGB['ours'])], background=pct, vmin=0, vmax=100)
    if only:
        log('RENDER', tag)
        return (tag, None, None)
    catAB = {h: border_categories(C.hemi_lab(oursA, h), C.hemi_lab(oursB, h), C.edges(h), C.adjacency(h)) for h in C.HEMIS}
    LA = lambda k: {h: catAB[h][k] for h in C.HEMIS}
    render_layers(os.path.join(RDR, f'halves_{tag}.png'), [(LA('shared'), (110, 110, 110)), (LA('only1'), RGB['A']), (LA('only2'), RGB['B'])], size=(2400, 1500))
    ab = {k: sum((int(catAB[h][k].sum()) for h in C.HEMIS)) for k in ('shared', 'only1', 'only2', 'B1', 'B2')}
    ab_m = {}
    for h in C.HEMIS:
        for k in ('shared', 'B1', 'B2'):
            ab_m[k] = ab_m.get(k, 0) + int(cat[h][k].sum())
    log('RENDER', tag)
    return (tag, ab, ab_m)

def stats_job(a):
    (study, g, coh, kind) = a
    os.environ['OMP_NUM_THREADS'] = '2'
    names = mmp_names()
    rois = roi_masks(names, WIN)
    rows = window_f1({'PFBN v5' if kind == 'v5' else 'PFBN v5m': atlas_v5(f'{coh}_A', kind), 'Micapipe': micapipe()}, GPATH(coh), rois)
    for r in rows:
        r.update(study=study, group=g, kind=kind)
    log('STATS', study, g, kind)
    return rows

def color_ranges():
    rng = {}
    for (study, gl) in GROUPS.items():
        v = np.concatenate([np.concatenate([x[np.isfinite(x)] for x in gmap_cortex(GPATH(coh)).values()]) for (_, coh) in gl])
        rng[study] = [float(np.percentile(v, 2)), float(np.percentile(v, 98))]
    json.dump(rng, open(os.path.join(RES, 'color_ranges.json'), 'w'), indent=1)
    return rng

def main_render():
    rng = color_ranges()
    with ProcessPoolExecutor(6) as ex:
        out = list(ex.map(render_job, [(s, g, c, k, rng[s]) for (s, g, c, k) in jobs()]))
    if os.environ.get('REFERENCE_ONLY'):
        log('RENDER_END')
        return
    rows = []
    for (tag, ab, abm) in out:
        f_ab = 2 * ab['shared'] / max(ab['B1'] + ab['B2'], 1)
        f_m = 2 * abm['shared'] / max(abm['B1'] + abm['B2'], 1)
        rows.append(dict(tag=tag, halfA_vs_halfB_border_agreement=f_ab, ours_vs_micapipe_border_agreement=f_m, **{f'AB_{k}': v for (k, v) in ab.items()}))
    pd.DataFrame(rows).to_csv(os.path.join(RES, 'border_agreement.csv'), index=False)
    log('RENDER_END')

def main_stats():
    with ProcessPoolExecutor(12) as ex:
        out = list(ex.map(stats_job, list(jobs())))
    pd.DataFrame([r for o in out for r in o]).to_csv(os.path.join(RES, 'window_f1.csv'), index=False)
    log('STATS_END')

def main_fig():
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.lines import Line2D
    from gradient_zoom_figs import crops, trim, HUES
    bx = crops()
    rng = json.load(open(os.path.join(RES, 'color_ranges.json')))
    f1 = pd.read_csv(os.path.join(RES, 'window_f1.csv'))
    agr = pd.read_csv(os.path.join(RES, 'border_agreement.csv')).set_index('tag')

    def trim_white(im, pad=6):
        m = (im[..., :3] < 0.97).any(2)
        r = np.where(m.any(1))[0]
        c = np.where(m.any(0))[0]
        return im[max(r[0] - pad, 0):r[-1] + pad, max(c[0] - pad, 0):c[-1] + pad]
    for (study, gl) in GROUPS.items():
        for kind in ('v5', 'v5m'):
            on = 'PFBN v5' if kind == 'v5' else 'PFBN v5m'
            n = len(gl)
            fig = plt.figure(figsize=(7.2, 1.55 * n + 0.75))
            gs = GridSpec(n + 1, 4, figure=fig, width_ratios=[2.1, 1, 1, 1.25], height_ratios=[1] * n + [0.32], hspace=0.12, wspace=0.06, left=0.07, right=0.995, top=0.95, bottom=0.03)
            for (i, (g, coh)) in enumerate(gl):
                tag = f'{study}_{g}_{kind}'
                ax = fig.add_subplot(gs[i, 0])
                ax.imshow(trim_white(plt.imread(os.path.join(RDR, f'parc_{tag}.png'))))
                ax.set_xticks([])
                ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(False)
                ax.set_ylabel(GN[g], fontsize=7.5)
                if i == 0:
                    ax.set_title(f'a  {on} parcels, Micapipe borders (black)', fontsize=7, loc='left')
                gim = plt.imread(os.path.join(RDR, f'grad_{tag}.png'))
                for (k, roi) in enumerate(WIN):
                    (r0, r1, c0, c1) = bx[roi]
                    ax = fig.add_subplot(gs[i, 1 + k])
                    ax.imshow(trim(gim[r0:r1, c0:c1], 2))
                    ax.set_xticks([])
                    ax.set_yticks([])
                    for s in ax.spines.values():
                        s.set_visible(True)
                        s.set_linewidth(1.0)
                        s.set_edgecolor(mpl.colors.hsv_to_rgb((HUES[roi], 0.9, 0.9)))
                    sub = f1[(f1.study == study) & (f1.group == g) & (f1.kind == kind) & (f1.roi == roi)].set_index('atlas')
                    d = sub.loc[f'{on} - Micapipe']
                    ax.text(0.02, 0.02, f"F1 {sub.loc[on, 'F1']:.2f} vs {sub.loc['Micapipe', 'F1']:.2f}\nspin p = {d.p_spin:.3f}", transform=ax.transAxes, fontsize=5.3, va='bottom', ha='left', bbox=dict(facecolor='white', alpha=0.75, lw=0, pad=1))
                    if i == 0:
                        ax.set_title(('b  ' if k == 0 else '') + roi, fontsize=7, loc='left')
                ax = fig.add_subplot(gs[i, 3])
                ax.imshow(trim_white(plt.imread(os.path.join(RDR, f'halves_{tag}.png'))))
                ax.set_xticks([])
                ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(False)
                ax.text(0.5, -0.02, f"half A vs B agreement {agr.loc[tag, 'halfA_vs_halfB_border_agreement']:.2f}", transform=ax.transAxes, fontsize=5.3, ha='center', va='top')
                if i == 0:
                    ax.set_title('c  Half A vs half B borders', fontsize=7, loc='left')
            cax = fig.add_axes([0.36, 0.035, 0.2, 0.012])
            cb = mpl.colorbar.ColorbarBase(cax, cmap=mpl.colormaps['YlOrRd'], norm=mpl.colors.Normalize(0, 100), orientation='horizontal')
            cb.ax.tick_params(labelsize=5, width=0.4, length=2)
            cb.outline.set_linewidth(0.4)
            cb.set_label('Held-out (half B) local RSFC gradient\n(percentile within group)', fontsize=5.8)
            lax = fig.add_axes([0.58, 0.0, 0.42, 0.06])
            lax.axis('off')
            c = lambda k: tuple(np.asarray(RGB[k]) / 255)
            lax.legend(handles=[Line2D([], [], color=c('shared'), lw=1.5), Line2D([], [], color=c('ours'), lw=1.5), Line2D([], [], color=c('mica'), lw=1.5), Line2D([], [], color=c('A'), lw=1.5), Line2D([], [], color=c('B'), lw=1.5)], labels=['shared border', f'{on} only', 'Micapipe only', 'half A only', 'half B only'], ncol=3, fontsize=5.3, frameon=False, loc='center', handlelength=1.4, columnspacing=1.0)
            P.save(fig, os.path.join(FIG, f'fig_reference_R1_{study}' + ('' if kind == 'v5' else '_v5m')))
            plt.close(fig)
    log('FIG_END')
if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if what in ('render', 'all'):
        main_render()
    if what in ('stats', 'all'):
        main_stats()
    if what in ('fig', 'all'):
        main_fig()
