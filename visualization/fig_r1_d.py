import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
from pfbn import settings
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib as mpl
R = settings.out()
MECHANISM = os.path.join(R, 'dynamics_mechanism')
OUT = os.path.join(R, 'prism_export', 'figs')
plt.rcParams.update({'font.family': 'Liberation Sans', 'font.weight': 'bold', 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
info = pd.read_csv(os.path.join(MECHANISM, 'results', 'prior_affinity_info.csv'))

def lh(path):
    im = plt.imread(path)[..., :3]
    (h, w) = im.shape[:2]
    im = im[:, :w // 2]
    m = (im < 0.98).any(2)
    r = np.where(m.any(1))[0]
    c = np.where(m.any(0))[0]
    return im[max(r[0] - 10, 0):r[-1] + 10, max(c[0] - 10, 0):c[-1] + 10]
fig = plt.figure(figsize=(5.6, 3.6))
for (k, (nm, col)) in enumerate((('Without dynamics', '#9E9E9E'), ('With dynamics', '#8CC3DE'))):
    ax = fig.add_axes([0.04 + k * 0.43, 0.1, 0.4, 0.74])
    ax.imshow(lh(os.path.join(MECHANISM, 'figs', 'renders', f"prior_affinity_{nm.replace(' ', '_')}.png")))
    ax.axis('off')
    r = info[info.prior == nm].half_max_geodesic_mm
    ax.set_title(nm, fontsize=11, weight='bold', pad=4, color='black')
    ax.text(0.5, -0.04, f"Half-max radius {r.median():.0f} mm\n(seeds: {', '.join((f'{v:.0f}' for v in r))} mm)", transform=ax.transAxes, ha='center', va='top', fontsize=8)
    for s in ('left', 'right', 'top', 'bottom'):
        pass
cax = fig.add_axes([0.905, 0.25, 0.018, 0.45])
cb = mpl.colorbar.ColorbarBase(cax, cmap=mpl.colormaps['magma'], norm=mpl.colors.Normalize(0, 1))
cb.set_label('Prior coupling to seed', fontsize=8, weight='bold')
cb.ax.tick_params(labelsize=7)
fig.text(0.01, 0.97, 'd', fontsize=14, weight='bold', va='top')
fig.text(0.5, 0.97, 'Spatial prior from three seeds (area 4, area 46, V1)', ha='center', va='top', fontsize=9, weight='bold')
for e in ('png', 'pdf', 'svg'):
    fig.savefig(os.path.join(OUT, f'fig_R1_d_dynamics_prior.{e}'), dpi=600)
print('ok')
