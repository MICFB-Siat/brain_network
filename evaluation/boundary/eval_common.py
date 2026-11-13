import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
from pfbn import settings
import os, sys, json, time, hashlib
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_v, '12')
import numpy as np, nibabel as nib
import scipy.sparse as sp
from scipy.io import loadmat
ROOT = settings.out()
EXP = os.path.join(ROOT, 'validation')
RES = os.path.join(EXP, 'results')
FIGS = os.path.join(EXP, 'figs')
SCR = settings.scratch('validation')
SWEEP = settings.path('cato_sweep')
SIX = settings.path('cato_atlases')
PRIOR = settings.path('hcp_subjects')
S4TMP = os.path.join(settings.path('templates'), 'tmp')
HREP = settings.out('hcp_repeat')
MV2 = settings.out('ds005498')
V0 = settings.out('ds005498_v0')
DCBC_SCR = settings.scratch('dcbc_ds005498')
MICAPIPE = settings.path('micapipe_labels')
SPINS = settings.path('spin_rotations')
CONTE69 = os.path.join(settings.path('conte69'), 'Conte69.{H}.midthickness.32k_fs_LR.surf.gii')
(NL, NR, NCORT, NV) = (29696, 29716, 59412, 32492)
HEMIS = ('lh', 'rh')
HL = {'lh': 'L', 'rh': 'R'}
HOFF = {'lh': 0, 'rh': NV}
HN = {'lh': NL, 'rh': NR}
(MAX_DIST, N_BINS) = (35.0, 35)
for _d in (RES, FIGS, SCR):
    os.makedirs(_d, exist_ok=True)
COL = {'Micapipe': '#8C8C8C', 'PFBN v0': '#5B8FD0', 'PFBN v5': '#D1495B', 'PFBN v5 w/o dynamics': '#E8A33D', 'CATO (Lausanne-219)': '#6A8D73'}

def log(*a):
    print(time.strftime('[%Y-%m-%d %H:%M:%S] ') + ' '.join((str(x) for x in a)), flush=True)

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda : f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()
_MED = None

def medial():
    global _MED
    if _MED is None:
        _MED = loadmat(os.path.join(S4TMP, 'fs_LR_32k_medial_mask.mat'))['medial_mask'][:, 0].astype(int)
    return _MED

def cortex_idx(h):
    m = medial()[HOFF[h]:HOFF[h] + NV]
    idx = np.where(m == 1)[0]
    assert len(idx) == HN[h]
    return idx

def hemi_lab(lab, h):
    return lab[:NL] if h == 'lh' else lab[NL:]

def edges(h):
    return np.load(os.path.join(S4TMP, f'{h}_edges.npy')).astype(np.int64)

def adjacency(h):
    e = edges(h)
    n = HN[h]
    A = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    return (A + A.T).tocsr()

def mesh_graph(h):
    xyz = nib.load(CONTE69.format(H=HL[h])).darrays[0].data.astype(np.float64)
    cidx = cortex_idx(h)
    e = edges(h)
    n = HN[h]
    w = np.linalg.norm(xyz[cidx[e[:, 0]]] - xyz[cidx[e[:, 1]]], axis=1)
    G = sp.coo_matrix((w, (e[:, 0], e[:, 1])), shape=(n, n))
    return (G + G.T).tocsr()

def read_dlabel_cortex(path):
    d = nib.load(path).get_fdata()[0]
    assert d.shape[0] in (64984, NCORT), d.shape
    return (d[medial() == 1] if d.shape[0] == 64984 else d).round().astype(np.int64)

def load_labels(path, key='labels'):
    if path.endswith('.dlabel.nii'):
        lab = read_dlabel_cortex(path)
    elif path.endswith('.npz'):
        lab = np.load(path)[key].astype(np.int64)
    else:
        lab = np.load(path).astype(np.int64)
    assert lab.shape == (NCORT,), (path, lab.shape)
    return lab

def boundary_mask(lab_h, e):
    (la, lb) = (lab_h[e[:, 0]], lab_h[e[:, 1]])
    d = (la != lb) & (la > 0) & (lb > 0)
    m = np.zeros(len(lab_h), bool)
    m[e[d, 0]] = True
    m[e[d, 1]] = True
    return m

def dilate(mask, A, k):
    x = mask.astype(np.float32)
    for _ in range(k):
        x = x + A @ x
    return x > 0

def cortex_to_hemi_full(lab, h, fill=0):
    out = np.full(NV, fill, dtype=np.asarray(lab).dtype)
    out[cortex_idx(h)] = hemi_lab(np.asarray(lab), h)
    return out
