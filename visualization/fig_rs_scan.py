import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
from pfbn import settings
import os, sys, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, settings.code('visualization'))
import eval_common as C
import plotstyle as P
from plotstyle import plt, matplotlib
from rs_scan import RS
import boundary_f1 as BF
R = os.path.join(C.FIGS, 'renders')
ORDER = list(RS)

def main():
    import wbrender as W
    z = np.load(os.path.join(C.SCR, 'kernel_psf.npz'))
    for k in ('fitted', 'r30'):
        out = os.path.join(R, f'psf_{k}.png')
        if not os.path.exists(out):
            v = np.nan_to_num(z[k])
            v[v < 0.05] = 0
            W.scalar_png(v, np.zeros(C.NV), out, palette='ROY-BIG-BL', pos_range=(0.05, 1.0), extra=['-disp-zero', 'false'])
    h = pd.concat([pd.read_csv(os.path.join(C.MV2, 'v5/results', f)) for f in ('homogeneity_long_v5.csv', 'homogeneity_long_v5b.csv')]).drop_duplicates(['subject', 'run', 'atlas'])
    h = h[h.run == 'rest'].pivot(index='subject', columns='atlas', values='weighted')
    d = pd.read_csv(os.path.join(C.RES, 'ds_rs_scan_dcbc.csv')).groupby(['subject', 'r_s']).DCBC.mean().unstack()
    com = h.index.intersection(d.index)
    assert len(com) == 32
    hom = {k: h.loc[com, v].values for (k, v) in RS.items()}
    dc = {k: d.loc[com, k].values for k in ORDER}
    sets = [(k, C.load_labels(os.path.join(C.MV2, 'v5/results', f'labels_{v}.npy')), os.path.join(C.MV2, 'results', 'gmap_rest_{h}.npy')) for (k, v) in RS.items()]
    (f, _, nulls) = BF.run('ds005498_rs', sets, 1000, 20)
    f = f.set_index('atlas')
    for k in ORDER:
        P.COL[k] = '#D1495B' if k == '30' else '#E3A1A9'
    fig = plt.figure(figsize=(10.5, 4.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.05])
    for (i, (k, t)) in enumerate((('fitted', 'Wave kernel, r_s = 8.5 mm (fitted)'), ('r30', 'Wave kernel, r_s = 30 mm (Pang 2023)'))):
        ax = fig.add_subplot(gs[0, i])
        P.image_panel(ax, os.path.join(R, f'psf_{k}.png'), t)
        P.panel_label(ax, 'ab'[i], -0.02, 1.0)
    ax = fig.add_subplot(gs[0, 2])
    P.image_panel(ax, os.path.join(R, 'ds005498_PFBN_v5.png'), 'PFBN v5 atlas (r_s = 30 mm)')
    P.panel_label(ax, 'c', -0.02, 1.0)
    cax = fig.add_axes([0.73, 0.62, 0.006, 0.22])
    roy = matplotlib.colors.LinearSegmentedColormap.from_list('roy_pos', ['#6E0000', '#FF0000', '#FF7800', '#FFFF00'])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=roy, norm=plt.Normalize(0.05, 1)), cax=cax)
    cb.set_label('Kernel / peak', fontsize=6)
    cb.ax.tick_params(labelsize=5)
    cb.outline.set_linewidth(0.4)
    st = {}
    ax = fig.add_subplot(gs[1, 0])
    P.violins(ax, hom, ORDER, 'Homogeneity (weighted r)', 'Held-out homogeneity', ref='30', stats_out=st, key='homogeneity')
    ax.set_xlabel('r_s (mm)')
    P.panel_label(ax, 'd')
    ax = fig.add_subplot(gs[1, 1])
    P.violins(ax, dc, ORDER, 'DCBC', 'DCBC (0-35 mm)', ref='30', stats_out=st, key='DCBC')
    ax.set_xlabel('r_s (mm)')
    P.panel_label(ax, 'e')
    ax = fig.add_subplot(gs[1, 2])
    P.null_violins(ax, {k: nulls[f'{k}__F1'] for k in ORDER}, {k: f.loc[k, 'F1'] for k in ORDER}, ORDER, 'Boundary F1', 'Boundary F1 vs gradient ridge', zs={k: f.loc[k, 'F1_z'] for k in ORDER})
    ax.set_xlabel('r_s (mm)')
    P.panel_label(ax, 'f')
    ax = fig.add_subplot(gs[1, 3])
    P.null_violins(ax, {k: nulls[f'{k}__dist'] for k in ORDER}, {k: f.loc[k, 'dist_mm'] for k in ORDER}, ORDER, 'Boundary-to-ridge distance (mm)', 'Boundary distance', zs={k: f.loc[k, 'dist_z'] for k in ORDER})
    ax.set_xlabel('r_s (mm)')
    P.panel_label(ax, 'g')
    fig.tight_layout()
    P.save(fig, os.path.join(C.FIGS, 'fig5_rs_scan'))
    rows = [dict(r_s=k, homogeneity_median=float(np.median(hom[k])), DCBC_median=float(np.median(dc[k])), F1=f.loc[k, 'F1'], F1_z=f.loc[k, 'F1_z'], dist_mm=f.loc[k, 'dist_mm'], dist_z=f.loc[k, 'dist_z']) for k in ORDER]
    pd.DataFrame(rows).to_csv(os.path.join(C.RES, 'rs_scan_summary.csv'), index=False)
    json.dump(st, open(os.path.join(C.RES, 'rs_scan_stats.json'), 'w'), indent=1)
    print(pd.DataFrame(rows).to_string(index=False))
    print(json.dumps(st, indent=1))
if __name__ == '__main__':
    main()
