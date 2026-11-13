import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
from pfbn import settings
import os, sys, json, time, hashlib
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '6')
os.environ.setdefault('GY_GPU', os.environ['CUDA_VISIBLE_DEVICES'])
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_v, '8')
import numpy as np
import nibabel as nib
from scipy.io import loadmat
EXP = settings.out('ds005498')
NAS = settings.path('ds005498', 'ds005498_minimal_v2_20261024')
RESULTS = os.path.join(EXP, 'results')
LOGS = os.path.join(EXP, 'logs')
EXP1 = settings.out('ds005498_v0')
NAS1 = settings.path('ds005498', 'ds005498_TMS_homogeneity_20261023')
PFBN_COPY1 = settings.code('pfbn', 'models')
S4 = settings.path('templates')
S4TMP = os.path.join(S4, 'tmp')
MODES = settings.out('eigenmodes')
BA = settings.out('boundary_agreement')
ANALYSIS = settings.code('pfbn', 'eigenmodes')
MICAPIPE = settings.path('micapipe_labels')
WB = settings.get('wb_command')
TMPFS = os.environ.get('MV2_TMPFS', settings.tmp('ds005498'))
FIT_KEY = 'stimMinus38x22x48@2'
REST_KEY = 'resting@1'
(NL, NR, NCORT, NV) = (29696, 29716, 59412, 32492)
HEMIS = ('lh', 'rh')
HEMI_LETTER = {'lh': 'L', 'rh': 'R'}
HEMI_OFF = {'lh': 0, 'rh': NV}
HEMI_N = {'lh': NL, 'rh': NR}
(KAPPA, CW, K_GRAD, MAX_IT, STOP) = (10.0, 200.0, 15.0, 30, 0.002)
for d in (RESULTS, LOGS, NAS):
    os.makedirs(d, exist_ok=True)

def log(*a):
    s = time.strftime('[%Y-%m-%d %H:%M:%S] ') + ' '.join((str(x) for x in a))
    print(s, flush=True)

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda : f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

class Timings:
    PATH = os.path.join(RESULTS, 'timings.json')

    @classmethod
    def load(cls):
        return json.load(open(cls.PATH)) if os.path.exists(cls.PATH) else {}

    @classmethod
    def put(cls, stage, key, seconds, extra=None):
        t = cls.load()
        t.setdefault(stage, {})
        t[stage][key] = {'seconds': round(float(seconds), 2), 'finished': time.strftime('%Y-%m-%d %H:%M:%S')}
        if extra:
            t[stage][key].update(extra)
        json.dump(t, open(cls.PATH, 'w'), indent=1)

def medial():
    return loadmat(os.path.join(S4TMP, 'fs_LR_32k_medial_mask.mat'))['medial_mask'][:, 0].astype(int)

def cortex_idx(hemi):
    m = medial()[HEMI_OFF[hemi]:HEMI_OFF[hemi] + NV]
    idx = np.where(m == 1)[0]
    assert len(idx) == HEMI_N[hemi]
    return idx

def cortex_to_full(values_cortex, fill=np.nan):
    m = medial()
    out = np.full(64984, fill, dtype=np.result_type(np.asarray(values_cortex).dtype, type(fill)) if fill is not None else np.asarray(values_cortex).dtype)
    out[m == 1] = values_cortex
    return out

def full_to_cortex(values_full):
    return np.asarray(values_full)[medial() == 1]

def sphere(hemi):
    g = nib.load(os.path.join(S4TMP, f'S1200.{HEMI_LETTER[hemi]}.sphere.32k_fs_LR.surf.gii'))
    xyz = g.darrays[0].data.astype(np.float64)
    F = g.darrays[1].data.astype(np.int64)
    return (xyz, F)

def hemi_edges_cortex(hemi):
    e = np.load(os.path.join(S4TMP, f'{hemi}_edges.npy')).astype(np.int64)
    return e

def fit_subjects():
    runs = json.load(open(os.path.join(EXP1, 'nts_runs.json')))
    subs = sorted((s for (s, v) in runs.items() if FIT_KEY in v))
    return (subs, runs)

def load_cortex_bold(path):
    img = nib.load(path)
    X = img.get_fdata(dtype=np.float32)[:, :NCORT]
    return X

def demean_unit(X, torch_mod=None):
    if torch_mod is not None:
        X = X - X.mean(0, keepdim=True)
        n = X.norm(dim=0, keepdim=True)
        n[n == 0] = 1.0
        return X / n
    X = X - X.mean(0, keepdims=True)
    n = np.linalg.norm(X, axis=0, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def cortex_labels_to_dlabel(labels_cortex, out_path):
    if PFBN_COPY1 not in sys.path:
        sys.path.insert(0, PFBN_COPY1)
    from generate_dlabel import generate_dlabel_from_labels
    lab = np.asarray(labels_cortex).astype(np.int64).copy()
    lab[NL:] = np.where(lab[NL:] > 0, lab[NL:] - 100, 0)
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        generate_dlabel_from_labels(lab, out_path, S4)
    return out_path

def read_dlabel_cortex(path):
    d = nib.load(path).get_fdata()[0]
    assert d.shape[0] in (64984, NCORT), d.shape
    return (d[medial() == 1] if d.shape[0] == 64984 else d).round().astype(np.int64)
