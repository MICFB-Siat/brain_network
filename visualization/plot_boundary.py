import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
from pfbn import settings
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, settings.code('visualization'))
import numpy as np, pandas as pd
import cohort_lib as R
import wbrender as W
sys.path.insert(0, settings.code('pfbn', 'pipeline'))
import common as PC
ap = argparse.ArgumentParser()
ap.add_argument('--exp_dir', required=True)
ap.add_argument('--ref', required=True)
ap.add_argument('--order', required=True)
ap.add_argument('--gcolors', default='')
ap.add_argument('--no_brain', action='store_true')
a = ap.parse_args()
RES = os.path.join(a.exp_dir, 'results')
FIG = os.path.join(a.exp_dir, 'figs')
os.makedirs(FIG, exist_ok=True)
order = a.order.split(',')
plt = R.mpl()
st = json.load(open(os.path.join(RES, 'stats.json')))
GC = dict(zip(order, a.gcolors.split(','))) if a.gcolors else dict(zip(order, ['#4C9F70', '#E0A030', '#D1495B', '#5B6FB0', '#8E5BA8'][:len(order)]))
nice = lambda g: g.replace('_', ' ')
dc = pd.read_csv(os.path.join(RES, 'dcbc_per_subject.csv'))
(fig, axs) = plt.subplots(1, len(order), figsize=(1.55 * len(order) + 0.3, 2.4), sharey=False)
for (ax, g) in zip(np.atleast_1d(axs), order):
    d = dc[dc.group == g]
    own = d[d.atlas == g].set_index('subject').DCBC
    oth = d[~d.own & (d.atlas != 'Micapipe')].groupby('subject').DCBC.mean().loc[own.index]
    mic = d[d.atlas == 'Micapipe'].set_index('subject').DCBC.loc[own.index]
    R.violin_box(ax, [own.values, oth.values, mic.values], ['Own-group\nPFBN v5', 'Other-group\nPFBN v5', 'Micapipe'], colors=[GC[g], '#BBBBBB', R.COLORS['Micapipe']], paired=True, ylabel='DCBC (held-out half)' if g == order[0] else '')
    s = st['dcbc'][g]
    top = max(own.max(), oth.max(), mic.max())
    span = top - min(own.min(), oth.min(), mic.min())
    R.bracket(ax, 0, 1, top + 0.04 * span, f"{R.pstar(s['own_vs_others_p'])}")
    R.bracket(ax, 0, 2, top + 0.14 * span, f"{R.pstar(s['own_vs_micapipe_p'])}")
    ax.set_title(f"{nice(g)} (n={s['n']})", fontsize=7)
fig.tight_layout()
R.save(fig, os.path.join(FIG, 'fig_dcbc'))
dp = pd.read_csv(os.path.join(RES, 'displacement_parcels.csv'))
others = [g for g in order if g != a.ref]
short = {'young_adults': 'YA'}.get(a.ref, a.ref)
(fig, axs) = plt.subplots(1, len(others), figsize=(1.8 * len(others) + 0.6, 2.5))
for (ax, g) in zip(np.atleast_1d(axs), others):
    tr = dp[dp.comparison == f'{g} test-retest'].set_index('mica_parcel').displacement_mm
    ra = dp[dp.comparison == f'{g} vs {a.ref} (A)'].set_index('mica_parcel').displacement_mm
    rb = dp[dp.comparison == f'{g} vs {a.ref} (B)'].set_index('mica_parcel').displacement_mm
    ok = tr.notna() & ra.notna() & rb.notna()
    R.violin_box(ax, [tr[ok].values, ra[ok].values, rb[ok].values], ['Test-retest\n(A vs B)', f'vs {short}\n(A)', f'vs {short}\n(B)'], colors=['#BBBBBB', GC[g], GC[g]], paired=True, ylabel='Boundary displacement (mm, per parcel)' if g == others[0] else '')
    s = st['displacement'][g]
    top = max(tr[ok].max(), ra[ok].max(), rb[ok].max())
    R.bracket(ax, 0, 1, top * 1.04, R.pstar(s['vsrefA_vs_testretest_p']))
    ax.set_title(f"{nice(g)} vs {short}\npattern r(A,B)={s['pattern_reliability_spearman_r']:.2f}", fontsize=7)
    ax.tick_params(axis='x', labelrotation=0)
    plt.setp(ax.get_xticklabels(), ha='center')
fig.tight_layout()
R.save(fig, os.path.join(FIG, 'fig_displacement'))
if not a.no_brain:
    os.environ.setdefault('WBR_TMP', settings.tmp())
    mp = np.load(os.path.join(RES, 'maps.npz'))
    keys = mp['mica_keys']
    mica = R.micapipe()
    full = lambda x: (PC.cortex_to_full(x, fill=0)[:32492], PC.cortex_to_full(x, fill=0)[32492:])
    for g in order:
        lab = mp[f'atlas_{g}_A']
        (l, r) = full(lab)
        W.labels_png(l.astype(int), r.astype(int), os.path.join(FIG, f'brain_atlas_{g}.png'))
    vmax = float(np.nanpercentile(np.concatenate([mp[f'{g}_vs_{a.ref}_(A)|parcel'.replace(' ', '_')] for g in others]), 95))
    for g in others:
        per = mp[f'{g}_vs_{a.ref}_(A)|parcel'.replace(' ', '_')]
        v = np.zeros(R.NCORT)
        for (k, p) in zip(keys, per):
            v[mica == k] = p if np.isfinite(p) else 0
        (l, r) = full(v)
        W.scalar_png(l, r, os.path.join(FIG, f'brain_disp_{g}.png'), palette='magma', pos_range=(0, vmax))
        ref = mp[f'atlas_{a.ref}_A']
        gg = mp[f'atlas_{g}_A']
        lab = np.ones(R.NCORT, int)
        for (h, sl) in (('L', slice(0, R.NL)), ('R', slice(R.NL, R.NCORT))):
            pass
        bl = []
        for lb in (ref, gg):
            (ll, rr) = full(lb)
            bl.append((W.border_mask(ll.astype(int), 'L'), W.border_mask(rr.astype(int), 'R')))
        out = []
        for i in range(2):
            x = np.where(np.concatenate([[1]])[0] == 1, 1, 1) * np.ones(32492, int)
            x[bl[0][i]] = 2
            x[bl[1][i]] = 3
            x[bl[0][i] & bl[1][i]] = 4
            base = full(ref)[i] > 0
            x[~base] = 0
            out.append(x)
        W.labels_png(out[0], out[1], os.path.join(FIG, f'brain_overlay_{g}.png'), borders=False, colors={1: (0.92, 0.92, 0.92), 2: (0, 0, 0), 3: tuple((int(GC[g][j:j + 2], 16) / 255 for j in (1, 3, 5))), 4: (0.45, 0.1, 0.55)})
    import matplotlib as mpl
    (fig, ax) = plt.subplots(figsize=(1.6, 0.35))
    cb = mpl.colorbar.ColorbarBase(ax, cmap=mpl.cm.magma, norm=mpl.colors.Normalize(0, vmax), orientation='horizontal')
    cb.set_label(f'Boundary displacement vs {nice(a.ref)} (mm)', fontsize=6)
    cb.ax.tick_params(labelsize=6)
    R.save(fig, os.path.join(FIG, 'brain_disp_colorbar'))
    json.dump({'displacement_colour_range_mm': [0, vmax], 'palette': 'magma'}, open(os.path.join(FIG, 'brain_render_info.json'), 'w'))
print('saved', FIG)
