import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
from pfbn import settings
import os, sys, json, time
E = settings.out('ad_halfsplit')
RES = os.path.join(E, 'results')
FIG = os.path.join(E, 'figs')
sys.path.insert(0, settings.code('evaluation', 'boundary'))
WIN = ['Somatomotor', 'Dorsolateral PFC']

def main_dist():
    import numpy as np, pandas as pd
    import gradient_distance as M
    from concurrent.futures import ProcessPoolExecutor
    M.ATL = os.path.join(E, 'atlases')
    with ProcessPoolExecutor(2) as ex:
        res = list(ex.map(M.job, [('AD', 'ad_case', h) for h in M.C.HEMIS]))
    d = {h: (obs, null) for (_, h, obs, null) in res}
    (rows, nulls, vals) = ([], {}, {})
    for nm in d['lh'][0]:
        for r in d['lh'][0][nm]:
            o = M.pool(d['lh'][0][nm][r], d['rh'][0][nm][r])
            nl = [M.pool(a, b) for (a, b) in zip(d['lh'][1][nm][r], d['rh'][1][nm][r])]
            F = M.f1(o)[2]
            D_ = o['dsum'] / max(o['nB'], 1)
            nf = np.array([M.f1(x)[2] for x in nl])
            nd = np.array([x['dsum'] / max(x['nB'], 1) for x in nl])
            vals[nm, r] = (F, D_, nf, nd)
            rows.append(dict(group='AD', roi=r, comparison=nm, F1=F, F1_z=(F - nf.mean()) / nf.std(ddof=1), dist_mm=D_, dist_z=(D_ - nd.mean()) / nd.std(ddof=1), p_F1=(1 + (nf >= F).sum()) / (len(nf) + 1), p_dist=(1 + (nd <= D_).sum()) / (len(nd) + 1)))
            nulls[f'AD|{r}|{nm}|F1'] = nf
            nulls[f'AD|{r}|{nm}|dist'] = nd
    for ref in ('Micapipe', 'CATO'):
        for r in ['Whole cortex'] + WIN:
            (F5, D5, nf5, nd5) = vals['PFBN v5', r]
            (Fr, Dr, nfr, ndr) = vals[ref, r]
            (dF, dD, nF, nD) = (F5 - Fr, D5 - Dr, nf5 - nfr, nd5 - ndr)
            rows.append(dict(group='AD', roi=r, comparison=f'PFBN v5 - {ref}', F1=dF, dist_mm=dD, F1_z=(dF - nF.mean()) / nF.std(ddof=1), dist_z=(dD - nD.mean()) / nD.std(ddof=1), p_F1=(1 + (nF >= dF).sum()) / (len(nF) + 1), p_dist=(1 + (nD <= dD).sum()) / (len(nD) + 1), p_F1_two_sided_centred=(1 + (np.abs(nF - nF.mean()) >= abs(dF - nF.mean())).sum()) / (1 + len(nF)), p_dist_two_sided_centred=(1 + (np.abs(nD - nD.mean()) >= abs(dD - nD.mean())).sum()) / (1 + len(nD))))
            nulls[f'AD|{r}|PFBN v5 - {ref}|F1'] = nF
            nulls[f'AD|{r}|PFBN v5 - {ref}|dist'] = nD
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, 'dist_f1_AD.csv'), index=False)
    np.savez(os.path.join(RES, 'dist_f1_nulls_AD.npz'), **nulls)
    pd.set_option('display.width', 250)
    print(df.round(4).to_string(index=False), flush=True)

def main_dcbc():
    sys.path.insert(0, settings.code('evaluation', 'dcbc'))
    sys.path.insert(0, settings.code('pfbn', 'pipeline'))
    sys.path.insert(0, settings.code('evaluation', 'dcbc'))
    import numpy as np, pandas as pd, torch
    from scipy.stats import wilcoxon
    import cohort_lib as R, common as PC, dcbc as D
    SIX = settings.path('cato_atlases')
    ATL = {'CATO': np.asarray(np.load(os.path.join(SIX, 'assets', 'lausanne250_labels.npz'))['labels']).astype(np.int64), 'Micapipe': R.micapipe().astype(np.int64), 'NC atlas (PFBN v5)': np.load(os.path.join(R.ATL, 'pd_NC_A', 'results', 'labels_v5_stim_t03_r30_clean.npy')).astype(np.int64), 'AD atlas (PFBN v5)': np.load(os.path.join(E, 'atlases', 'ad_case_A', 'results', 'labels_v5_stim_t03_r30_clean.npy')).astype(np.int64)}
    pairs = {h: D.HemiPairs(h) for h in ('lh', 'rh')}
    cj = json.load(open(os.path.join(E, 'cohorts', 'adsplit_ad_case_B.json')))
    PC.DENOISE = cj.get('denoise', 'none')
    PC.DROP = int(cj.get('drop', 5))
    rows = []
    t0 = time.time()
    for s in cj['subjects']:
        X = np.concatenate([(lambda x: x - x.mean(0))(PC.load_run(r)[0]) for r in s['runs']], 0)
        Xt = torch.from_numpy(X).to(D.dev)
        res = {}
        for (h, sl) in (('lh', slice(0, R.NL)), ('rh', slice(R.NL, R.NCORT))):
            P_ = pairs[h]
            P_.set_subject(Xt[:, sl])
            for (k, lab) in ATL.items():
                res.setdefault(k, []).append(P_.evaluate(lab[sl])['DCBC_weighted'])
        for (k, v) in res.items():
            rows.append(dict(subject=s['sid'], group='AD', atlas=k, DCBC=float(np.mean(v)), frames=int(X.shape[0])))
        del Xt
        torch.cuda.empty_cache()
    dc = pd.DataFrame(rows)
    dc.to_csv(os.path.join(RES, 'dcbc_AD.csv'), index=False)
    t = dc.pivot(index='subject', columns='atlas', values='DCBC')
    own = 'AD atlas (PFBN v5)'
    st = {'n': len(t), 'seconds': time.time() - t0, 'median': {k: float(t[k].median()) for k in ATL}}
    for ref in ('CATO', 'Micapipe', 'NC atlas (PFBN v5)'):
        st[f'{own} vs {ref}'] = dict(n_greater=int((t[own] > t[ref]).sum()), p=float(wilcoxon(t[own], t[ref]).pvalue))
    st['Micapipe vs CATO'] = dict(n_greater=int((t['Micapipe'] > t['CATO']).sum()), p=float(wilcoxon(t['Micapipe'], t['CATO']).pvalue))
    json.dump(st, open(os.path.join(RES, 'dcbc_AD_stats.json'), 'w'), indent=1)
    print(json.dumps(st, indent=1), flush=True)

def main_fig():
    import numpy as np, pandas as pd, matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.stats import wilcoxon
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 6.5, 'axes.linewidth': 0.8, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    stars = lambda p: '****' if p < 0.0001 else '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
    COL = {'CATO': '#555555', 'Micapipe': '#D9A5B3', 'NC atlas (PFBN v5)': '#A0A0A0', 'AD atlas (PFBN v5)': '#5DA9D6'}
    dc = pd.read_csv(os.path.join(RES, 'dcbc_AD.csv'))
    t = dc.pivot(index='subject', columns='atlas', values='DCBC')
    names = list(COL)
    f = plt.figure(figsize=(6.6, 2.6))
    ax = f.add_axes([0.07, 0.2, 0.3, 0.62])
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
    (lo, hi) = (min(map(np.min, data)), max(map(np.max, data)))
    rg = hi - lo
    for (k, i) in enumerate((2, 1, 0)):
        p = wilcoxon(data[3], data[i]).pvalue
        y = hi + rg * (0.07 + 0.14 * k)
        ax.plot([i, i, 3, 3], [y - rg * 0.025, y, y, y - rg * 0.025], color='k', lw=0.7)
        ax.text((i + 3) / 2, y, stars(p), ha='center', va='bottom', fontsize=6.5)
    ax.set_ylim(lo - rg * 0.06, hi + rg * 0.55)
    ax.set_xticks(range(4))
    ax.set_xticklabels(['CATO', 'Micapipe', 'NC\natlas', 'AD\natlas'], fontsize=6)
    ax.set_ylabel('DCBC (held-out half B)')
    ax.set_title(f'AD patients (n = {len(t)})', fontsize=7)
    ax.spines[['top', 'right']].set_visible(False)
    sp = pd.read_csv(os.path.join(RES, 'dist_f1_AD.csv'))
    nl = np.load(os.path.join(RES, 'dist_f1_nulls_AD.npz'))
    for (k, roi) in enumerate(['Whole cortex'] + WIN):
        for (j, ref) in enumerate(('Micapipe', 'CATO')):
            ax = f.add_axes([0.45 + k * 0.185 + j * 0.085, 0.2, 0.065, 0.62])
            r = sp[(sp.roi == roi) & (sp.comparison == f'PFBN v5 - {ref}')].iloc[0]
            nd = nl[f'AD|{roi}|PFBN v5 - {ref}|dist']
            b = ax.violinplot(nd, positions=[0], widths=0.8, showextrema=False)['bodies'][0]
            b.set_facecolor('#E3E3E3')
            b.set_edgecolor('#9E9E9E')
            b.set_alpha(1)
            ax.scatter([0], [r.dist_mm], marker='D', s=20, color='#5DA9D6', edgecolor='black', lw=0.6, zorder=3)
            ax.axhline(0, color='0.5', lw=0.5, ls='--')
            ax.set_xticks([0])
            ax.set_xticklabels([f'vs\n{ref}'], fontsize=5.5)
            ax.spines[['top', 'right']].set_visible(False)
            ax.tick_params(labelsize=5)
            ax.set_title(f'{r.dist_mm:+.2f} mm\nP = {r.p_dist:.3f}', fontsize=5.3, pad=2)
            if j == 0:
                ax.text(1.15, 1.25, roi, transform=ax.transAxes, ha='center', fontsize=6.3)
            if k == 0 and j == 0:
                ax.set_ylabel('Δ border-to-ridge distance (mm)\nPFBN v5 − reference', fontsize=5.8)
    for e in ('png', 'pdf', 'svg'):
        f.savefig(os.path.join(FIG, f'fig_AD.{e}'), dpi=600, bbox_inches='tight')
if __name__ == '__main__':
    {'dist': main_dist, 'dcbc': main_dcbc, 'fig': main_fig}[sys.argv[1]]()
