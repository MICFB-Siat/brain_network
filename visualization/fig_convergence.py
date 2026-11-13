import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
from pfbn import settings
import os, sys, json
os.environ['WBR_TMP'] = settings.tmp('reproducibility', 'wbr')
os.environ['MPLCONFIGDIR'] = settings.tmp('reproducibility', 'mpl')
RDR = settings.tmp('reproducibility', 'renders')
for d in (os.environ['WBR_TMP'], os.environ['MPLCONFIGDIR'], RDR):
    os.makedirs(d, exist_ok=True)
ROOT = settings.out()
sys.path[0:0] = [settings.code('evaluation', 'boundary'), settings.code('visualization')]
from boundary_tools import C, render_layers, thin_border
import numpy as np, pandas as pd
from sklearn.metrics import adjusted_rand_score as ari
from scipy.optimize import linear_sum_assignment
E = f'{ROOT}/reproducibility'
ABLATION = f'{ROOT}/dynamics_ablation/atlases'
M = [('Without dynamics', 'v4_stim_t03_c1'), ('With dynamics', 'v5_stim_t03_r30')]
NS = (5, 10, 20)
EXC = 'life_young_adults'

def ref(c, m):
    return np.load(f'{ABLATION}/{c}_A/results/labels_{m}_clean.npy').astype(int)

def test(c, m, n, d):
    return np.load(f'{E}/atlases/{c}_n{n}_d{d}_A/results/labels_{m}_clean.npy').astype(int)

def mismatch(a, r):
    k = (a > 0) & (r > 0)
    (ka, kr) = (np.unique(a[k]), np.unique(r[k]))
    X = np.zeros((len(ka), len(kr)))
    np.add.at(X, (np.searchsorted(ka, a[k]), np.searchsorted(kr, r[k])), 1)
    (i, j) = linear_sum_assignment(-X)
    mp = dict(zip(ka[i], kr[j]))
    am = np.array([mp.get(x, -1) if x > 0 else 0 for x in a])
    return k & (am != r)

def stats():
    rows = []
    for c in ['life_children', 'life_adolescents', 'life_young_adults', 'life_older_adults']:
        for (nm, m) in M:
            R_ = ref(c, m)
            for n in NS:
                for d in range(3):
                    a = test(c, m, n, d)
                    rows.append(dict(cohort=c, method=nm, n=n, draw=d, ARI_to_full=ari(a, R_), mismatch=float(mismatch(a, R_).mean())))
    df = pd.DataFrame(rows)
    df.to_csv(f'{E}/results/convergence_to_full.csv', index=False)
    from scipy.stats import wilcoxon
    x = df.pivot_table(index=['cohort', 'n', 'draw'], columns='method', values='ARI_to_full')
    dd = x['With dynamics'] - x['Without dynamics']
    print('With closer', (dd > 0).sum(), '/', len(dd), 'P = %.2g' % wilcoxon(dd).pvalue)
    y = x.loc[EXC].xs(5, level='n') if False else x.loc[EXC, 5]
    md = y['With dynamics'] - y['Without dynamics']
    dsel = int((md - md.median()).abs().idxmin())
    json.dump(dict(draw=dsel, n_closer=int((dd > 0).sum()), n_total=int(len(dd)), p=float(wilcoxon(dd).pvalue)), open(f'{E}/results/convergence_meta.json', 'w'))
    print(df.groupby(['cohort', 'n', 'method']).ARI_to_full.mean().unstack('method').round(3))

def render():
    meta = json.load(open(f'{E}/results/convergence_meta.json'))
    d = meta['draw']
    out = {}
    for (nm, m) in M:
        R_ = ref(EXC, m)
        for n in NS:
            a = test(EXC, m, n, d)
            mm = mismatch(a, R_)
            out[f'{m}_{n}'] = float(mm.mean())
            lay = {h: C.hemi_lab(mm.astype(int), h) > 0 for h in C.HEMIS}
            bord = {h: thin_border(C.hemi_lab(R_, h), C.edges(h)) for h in C.HEMIS}
            render_layers(f'{RDR}/conv_{m}_{n}.png', [(lay, (192, 57, 43)), (bord, (120, 120, 120))], background=None, base_rgb=(235, 235, 235))
    meta['mismatch'] = out
    json.dump(meta, open(f'{E}/results/convergence_meta.json', 'w'), indent=1)
    print(out)

def fig():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'Liberation Sans', 'font.weight': 'bold', 'font.size': 8, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    meta = json.load(open(f'{E}/results/convergence_meta.json'))
    (fig, axs) = plt.subplots(2, 3, figsize=(7.2, 4.2), gridspec_kw=dict(wspace=0.04, hspace=0.18, left=0.13, right=0.99, top=0.86, bottom=0.03))
    for (i, (nm, m)) in enumerate(M):
        for (j, n) in enumerate(NS):
            im = plt.imread(f'{RDR}/conv_{m}_{n}.png')[..., :3]
            (H, W) = im.shape[:2]
            im = im[:, :W // 2]
            mk = (im < 0.97).any(2)
            (r_, c_) = (np.where(mk.any(1))[0], np.where(mk.any(0))[0])
            im = im[r_[0]:r_[-1] + 1, c_[0]:c_[-1] + 1]
            ax = axs[i, j]
            ax.imshow(im)
            ax.set_xticks([])
            ax.set_yticks([])
            [s.set_visible(False) for s in ax.spines.values()]
            ax.text(0.5, -0.02, f"{100 * meta['mismatch'][f'{m}_{n}']:.0f}% differ from full atlas", transform=ax.transAxes, ha='center', va='top', fontsize=7.5, color='#C0392B' if i == 0 else '#1F5F86')
            if i == 0:
                ax.set_title(f'{n} subjects', fontsize=10)
        axs[i, 0].text(-0.08, 0.5, nm.replace(' ', '\n'), transform=axs[i, 0].transAxes, ha='right', va='center', fontsize=10, color='#555555' if i == 0 else '#1F5F86')
    fig.suptitle('Young adults: small-sample atlas vs 50-subject atlas (red = parcel differs; grey lines = 50-subject borders)', fontsize=8.5, x=0.13, ha='left')
    for e in ('png', 'pdf', 'svg'):
        fig.savefig(f'{E}/figs/fig_convergence_brain.{e}', dpi=300)
if __name__ == '__main__':
    {'stats': stats, 'render': render, 'fig': fig}[sys.argv[1]]()
