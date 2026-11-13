import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
from pfbn import settings
import os, sys, json, time, hashlib
import numpy as np
RUNNER = settings.out()
ROOT = settings.out()
NASR = settings.scratch()
ATL = f'{ROOT}/atlases'
COLORS = {'Micapipe': '#8C8C8C', 'PFBN v0': '#5B8FD0', 'PFBN v5': '#D1495B', 'PFBN v5 w/o dynamics': '#E8A33D'}
FILES = {'PFBN v0': 'v0/results/labels_v0.npy', 'PFBN v5': 'results/labels_v5_stim_t03_r30_clean.npy', 'PFBN v5 w/o dynamics': 'results/labels_v4_stim_t03_c1_clean.npy'}
(NL, NCORT) = (29696, 59412)

def log(*a):
    print(time.strftime('[%Y-%m-%d %H:%M:%S] ') + ' '.join((str(x) for x in a)), flush=True)

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda : f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

def micapipe():
    z = np.load(settings.path('micapipe_labels'))
    return np.asarray(z['labels']).astype(np.int64)

def atlas(name, cohort):
    if name == 'Micapipe':
        return micapipe()
    return np.load(os.path.join(ATL, cohort, FILES[name])).astype(np.int64)

def atlas_path(name, cohort):
    return 'Micapipe200_labels.npz' if name == 'Micapipe' else os.path.join(ATL, cohort, FILES[name])

def onehot(lab):
    keys = np.unique(lab[lab > 0])
    M = np.zeros((len(lab), len(keys)), np.float32)
    idx = np.searchsorted(keys, lab[lab > 0])
    M[np.where(lab > 0)[0], idx] = 1
    return (M / M.sum(0, keepdims=True), keys)

def fc_z(ts_list):
    X = np.concatenate([t - t.mean(0) for t in ts_list], 0)
    X = X / np.where(X.std(0) == 0, 1, X.std(0))
    with np.errstate(invalid='ignore', divide='ignore'):
        r = np.corrcoef(X.T)
    r = np.nan_to_num(r, nan=0.0)
    iu = np.triu_indices(r.shape[0], 1)
    return np.arctanh(np.clip(r[iu], -0.999999, 0.999999))

def mpl():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none', 'legend.frameon': False})
    return plt

def violin_box(ax, data, names, colors=None, paired=False, ylabel='', points=True, jitter=0.08, seed=0, width=0.7, violin=True):
    rng = np.random.default_rng(seed)
    colors = colors or [COLORS.get(n, '#777777') for n in names]
    pos = np.arange(len(data))
    if violin:
        vp = ax.violinplot([np.asarray(d, float) for d in data], positions=pos, widths=width, showextrema=False)
        for (b, c) in zip(vp['bodies'], colors):
            b.set_facecolor(c)
            b.set_edgecolor('none')
            b.set_alpha(0.28)
    bp = ax.boxplot([np.asarray(d, float) for d in data], positions=pos, widths=0.18 if violin else 0.5, patch_artist=True, showfliers=False, medianprops=dict(color='black', linewidth=0.9), whiskerprops=dict(linewidth=0.6), capprops=dict(linewidth=0.6), boxprops=dict(linewidth=0.6))
    for (b, c) in zip(bp['boxes'], colors):
        b.set_facecolor(c if not violin else 'white')
        b.set_alpha(1.0)
    xs = []
    for (i, d) in enumerate(data):
        x = i + rng.uniform(-jitter, jitter, len(d))
        xs.append(x)
        if points:
            ax.scatter(x, d, s=4, color=colors[i], edgecolor='none', alpha=0.8, zorder=3)
    if paired and points:
        for i in range(len(data) - 1):
            for (a, b2, ya, yb) in zip(xs[i], xs[i + 1], data[i], data[i + 1]):
                ax.plot([a, b2], [ya, yb], color='#BBBBBB', linewidth=0.3, alpha=0.5, zorder=1)
    ax.set_xticks(pos)
    ax.set_xticklabels(names, rotation=30, ha='right')
    ax.set_ylabel(ylabel)
    return pos

def pstar(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'

def bracket(ax, x1, x2, y, text, h=None):
    h = h if h is not None else (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color='black', linewidth=0.5)
    ax.text((x1 + x2) / 2, y + h, text, ha='center', va='bottom', fontsize=6)

def save(fig, path_noext):
    os.makedirs(os.path.dirname(path_noext), exist_ok=True)
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(f'{path_noext}.{ext}', dpi=300, bbox_inches='tight')
