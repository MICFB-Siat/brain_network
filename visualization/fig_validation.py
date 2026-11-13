import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
from pfbn import settings
import os, sys, json, glob, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_common as C
import plotstyle as P
from plotstyle import plt
from sklearn.metrics import adjusted_rand_score
R = os.path.join(C.FIGS, 'renders')
STATS = {}
DS_ORDER = ['Micapipe', 'PFBN v0', 'PFBN v5 w/o dynamics', 'PFBN v5']
DS_KEY = {'Micapipe': 'Schaefer200', 'PFBN v0': 'v0', 'PFBN v5 w/o dynamics': 'v4_stim_t03_c1', 'PFBN v5': 'v5_stim_t03_r30_clean'}
HCP_ORDER = ['Micapipe', 'PFBN v0', 'PFBN v5']
HCP_KEY = {'Micapipe': 'Micapipe', 'PFBN v0': 'PFBN v0', 'PFBN v5': 'PFBN v5|REST1', 'CATO (Lausanne-219)': 'CATO (Lausanne-219)'}
NEAR = slice(0, 5)

def ds_data():
    h = pd.read_csv(os.path.join(C.MV2, 'v5/results/homogeneity_long_v5.csv'))
    h = h[h.run == 'rest'].pivot(index='subject', columns='atlas', values='weighted')
    d = pd.read_csv(settings.out('dcbc_ds005498', 'results', 'dcbc_per_subject.csv')).groupby(['subject', 'atlas']).DCBC_weighted.mean().unstack()
    z = np.load(settings.out('dcbc_ds005498', 'results', 'dcbc_bins.npz'))
    subs = list(z['subjects'])
    near = {m: np.mean([(z[f'{DS_KEY[m]}__{hh}__within'] - z[f'{DS_KEY[m]}__{hh}__between'])[:, NEAR].mean(1) for hh in C.HEMIS], 0) for m in DS_ORDER}
    near = pd.DataFrame(near, index=subs)
    common = h.index.intersection(d.index).intersection(near.index)
    assert len(common) == 32, len(common)
    return {'hom': {m: h.loc[common, DS_KEY[m]].values for m in DS_ORDER}, 'dcbc': {m: d.loc[common, DS_KEY[m]].values for m in DS_ORDER}, 'near': {m: near.loc[common, m].values for m in DS_ORDER}, 'subjects': list(common)}

def hcp_data(order):
    files = sorted(glob.glob(os.path.join(C.SCR, 'subjects', '*.npz')))
    out = {k: {m: [] for m in order} for k in ('hom', 'dcbc', 'near', 'retest')}
    subs = []
    for f in files:
        z = np.load(f)
        subs.append(os.path.basename(f)[:-4])
        for m in order:
            k = HCP_KEY[m]
            (hom, size) = (z[f'hom|{k}|REST2'], z[f'size|{k}|REST2'])
            ok = ~np.isnan(hom)
            out['hom'][m].append(np.sum(hom[ok] * size[ok]) / np.sum(size[ok]))
            out['dcbc'][m].append(np.mean([z[f'dcbc|{k}|REST2|{h}'] for h in C.HEMIS]))
            out['near'][m].append(np.mean([(z[f'within|{k}|REST2|{h}'] - z[f'between|{k}|REST2|{h}'])[NEAR].mean() for h in C.HEMIS]))
            fk = HCP_KEY[m] if m != 'PFBN v5' else 'PFBN v5|REST1'
            pairs = [(a, b) for a in ('rfMRI_REST1_LR', 'rfMRI_REST1_RL') for b in ('rfMRI_REST2_LR', 'rfMRI_REST2_RL')]
            out['retest'][m].append(np.mean([np.arctanh(np.corrcoef(z[f'fc|{fk}|{a}'], z[f'fc|{fk}|{b}'])[0, 1]) for (a, b) in pairs]))
    return ({k: {m: np.array(v) for (m, v) in d.items()} for (k, d) in out.items()}, subs)

def parcel_dice(a, b):
    out = []
    for p in np.unique(a[a > 0]):
        m = a == p
        (q, c) = np.unique(b[m & (b > 0)], return_counts=True)
        if len(q) == 0:
            out.append(0)
            continue
        sizes = np.array([(b == x).sum() for x in q])
        out.append(np.max(2 * c / (m.sum() + sizes)))
    return np.array(out)

def n_components(lab):
    from scipy.sparse.csgraph import connected_components
    import scipy.sparse as sp
    tot = 0
    for hh in C.HEMIS:
        l = C.hemi_lab(lab, hh)
        e = C.edges(hh)
        k = (l[e[:, 0]] == l[e[:, 1]]) & (l[e[:, 0]] > 0)
        A = sp.coo_matrix((np.ones(k.sum()), (e[k, 0], e[k, 1])), shape=(len(l), len(l)))
        (nc, cl) = connected_components(A, directed=False)
        tot += len(np.unique(cl[l > 0]))
    return tot

def components_sets():
    from boundary_f1 import DS
    from hcp_retest import hcp_atlases
    out = [('ds005498', k, C.load_labels(DS[k])) for k in DS_ORDER]
    atl = hcp_atlases()
    out += [('HCP', m, atl[HCP_KEY[m]]) for m in HCP_ORDER + ['CATO (Lausanne-219)']]
    out += [('ds005498', 'PFBN v5 (before island clean-up)', C.load_labels(os.path.join(C.MV2, 'v5/results/labels_v5_stim_t03_r30.npy')))]
    return out

def f1_panels(ax_f1, ax_d, dataset, order, key):
    f = pd.read_csv(os.path.join(C.RES, 'boundary_f1_distance.csv'))
    f = f[f.dataset == dataset].set_index('atlas')
    nz = np.load(os.path.join(C.RES, 'boundary_f1_distance_null.npz'))
    t = lambda s: s.replace('|', '_').replace(' ', '_').replace('/', '_')
    nf = {m: nz[t(f'{dataset}__{key(m)}__F1')] for m in order}
    nd = {m: nz[t(f'{dataset}__{key(m)}__dist')] for m in order}
    P.null_violins(ax_f1, nf, {m: f.loc[key(m), 'F1'] for m in order}, order, 'Boundary F1 (2-vertex tolerance)', 'Boundary F1 vs gradient ridge', zs={m: f.loc[key(m), 'F1_z'] for m in order})
    P.null_violins(ax_d, nd, {m: f.loc[key(m), 'dist_mm'] for m in order}, order, 'Boundary-to-ridge distance (mm)', 'Boundary distance', zs={m: f.loc[key(m), 'dist_z'] for m in order})
    return f

def atlas_figure(prefix, order, key, title_set, out):
    n = len(order)
    (fig, axs) = plt.subplots(2, n, figsize=(1.9 * n, 2.6))
    for (i, m) in enumerate(order):
        tag = key(m)
        P.image_panel(axs[0, i], os.path.join(R, f'{prefix}_{tag}.png'), m)
        P.image_panel(axs[1, i], os.path.join(R, f'{prefix}_{tag}_on_gradient.png'))
    axs[0, 0].text(-0.08, 0.5, 'Atlas', transform=axs[0, 0].transAxes, rotation=90, va='center', ha='right', fontsize=7)
    axs[1, 0].text(-0.08, 0.5, 'Borders on\nheld-out gradient', transform=axs[1, 0].transAxes, rotation=90, va='center', ha='right', fontsize=7)
    vr = np.load(os.path.join(R, 'gradient_ranges.npy'))[0 if prefix == 'ds005498' else 1]
    cax = fig.add_axes([0.92, 0.12, 0.008, 0.3])
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(*vr))
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label('Gradient magnitude', fontsize=6)
    cb.ax.tick_params(labelsize=5)
    cb.outline.set_linewidth(0.4)
    fig.suptitle(title_set, fontsize=8, y=0.99)
    fig.subplots_adjust(wspace=0.02, hspace=0.05)
    P.save(fig, out)

def main():
    tagd = lambda m: m.replace(' ', '_').replace('/', '')
    tagh = lambda m: HCP_KEY[m].replace(' ', '_').replace('|', '_').replace('(', '').replace(')', '')
    atlas_figure('ds005498', DS_ORDER, tagd, 'ds005498 (n = 32): cohort atlases, 200 parcels', os.path.join(C.FIGS, 'fig1_ds005498_atlases'))
    atlas_figure('HCP', HCP_ORDER, tagh, 'HCP (n = 50): atlases built on REST1, gradient from REST2', os.path.join(C.FIGS, 'fig2_HCP_atlases'))
    d = ds_data()
    (fig, axs) = plt.subplots(1, 5, figsize=(11.5, 2.6))
    P.violins(axs[0], d['hom'], DS_ORDER, 'Homogeneity (weighted r)', 'Held-out homogeneity', stats_out=STATS, key='ds005498_homogeneity')
    P.violins(axs[1], d['dcbc'], DS_ORDER, 'DCBC', 'DCBC (0-35 mm)', stats_out=STATS, key='ds005498_DCBC')
    P.violins(axs[2], d['near'], DS_ORDER, 'Within - between r (0-5 mm)', 'Similarity drop across borders', stats_out=STATS, key='ds005498_near_boundary')
    fds = f1_panels(axs[3], axs[4], 'ds005498', DS_ORDER, lambda m: m)
    for (a, l) in zip(axs, 'abcde'):
        P.panel_label(a, l)
    fig.tight_layout()
    P.save(fig, os.path.join(C.FIGS, 'fig3_ds005498_metrics'))
    (h, subs) = hcp_data(HCP_ORDER)
    assert len(subs) == 50 or os.environ.get('VALIDATION_PARTIAL'), len(subs)
    (fig, axs) = plt.subplots(1, 7, figsize=(15, 2.6))
    P.violins(axs[0], h['hom'], HCP_ORDER, 'Homogeneity (weighted r)', 'Held-out homogeneity (REST2)', stats_out=STATS, key='HCP_homogeneity')
    P.violins(axs[1], h['dcbc'], HCP_ORDER, 'DCBC', 'DCBC (REST2)', stats_out=STATS, key='HCP_DCBC')
    P.violins(axs[2], h['near'], HCP_ORDER, 'Within - between r (0-5 mm)', 'Similarity drop across borders', stats_out=STATS, key='HCP_near_boundary')
    fh = f1_panels(axs[3], axs[4], 'HCP', HCP_ORDER, lambda m: HCP_KEY[m])
    a1 = C.load_labels(os.path.join(C.HREP, 'labels_v5_stim_t03_r30_REST1_clean.npy'))
    a2 = C.load_labels(os.path.join(C.HREP, 'labels_v5_stim_t03_r30_REST2_clean.npy'))
    dice = parcel_dice(a1, a2)
    ari = adjusted_rand_score(a1, a2)
    P.violins(axs[5], {'PFBN v5': dice}, ['PFBN v5'], 'Best-match Dice per parcel', f'REST1 vs REST2 atlas\n(ARI = {ari:.2f}, 200 parcels)', paired=False, point_size=3)
    P.violins(axs[6], h['retest'], HCP_ORDER, 'Edge-wise FC test-retest (Fisher z)', 'FC reliability REST1 vs REST2', stats_out=STATS, key='HCP_edge_retest')
    for (a, l) in zip(axs, 'abcdefg'):
        P.panel_label(a, l)
    fig.tight_layout()
    P.save(fig, os.path.join(C.FIGS, 'fig4_HCP_metrics'))
    (fig, axs) = plt.subplots(2, 3, figsize=(8.2, 4.9))
    P.violins(axs[0, 0], d['near'], DS_ORDER, 'Within - between r (0-5 mm)', 'ds005498: similarity drop across borders')
    f1_panels(axs[0, 1], axs[0, 2], 'ds005498', DS_ORDER, lambda m: m)
    P.violins(axs[1, 0], h['near'], HCP_ORDER, 'Within - between r (0-5 mm)', 'HCP REST2: similarity drop across borders')
    f1_panels(axs[1, 1], axs[1, 2], 'HCP', HCP_ORDER, lambda m: HCP_KEY[m])
    for (a, l) in zip(axs.ravel(), 'abcdef'):
        P.panel_label(a, l)
    fig.tight_layout()
    P.save(fig, os.path.join(C.FIGS, 'fig6_boundary_f1_similarity'))
    rows = []
    for (ds_, dd, order) in (('ds005498', d, DS_ORDER), ('HCP', h, HCP_ORDER)):
        for k in ('hom', 'dcbc', 'near') + (('retest',) if ds_ == 'HCP' else ()):
            for m in order:
                rows.append(dict(dataset=ds_, metric=k, method=m, n=len(dd[k][m]), median=float(np.median(dd[k][m])), mean=float(np.mean(dd[k][m]))))
    for (ds_, f, order, key) in (('ds005498', fds, DS_ORDER, lambda m: m), ('HCP', fh, HCP_ORDER, lambda m: HCP_KEY[m])):
        for m in order:
            r = f.loc[key(m)]
            rows.append(dict(dataset=ds_, metric='boundary_F1', method=m, n=1, median=r.F1, mean=r.F1_z))
            rows.append(dict(dataset=ds_, metric='boundary_distance_mm', method=m, n=1, median=r.dist_mm, mean=r.dist_z))
    rows.append(dict(dataset='HCP', metric='parcel_dice_REST1_REST2', method='PFBN v5', n=len(dice), median=float(np.median(dice)), mean=float(np.mean(dice))))
    rows += [dict(dataset=ds_, metric='n_connected_components', method=m, n=1, median=n_components(lab), mean=np.nan) for (ds_, m, lab) in components_sets()]
    rows.append(dict(dataset='HCP', metric='ARI_REST1_REST2', method='PFBN v5', n=1, median=ari, mean=ari))
    S = pd.DataFrame(rows)
    S.to_csv(os.path.join(C.RES, 'summary.csv'), index=False)
    json.dump(STATS, open(os.path.join(C.RES, 'stats.json'), 'w'), indent=1)
    print(S.to_string(index=False))
    print(json.dumps(STATS, indent=1))
if __name__ == '__main__':
    main()
