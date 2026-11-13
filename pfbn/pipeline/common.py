import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
from pfbn import settings
import os, sys, json, time, hashlib
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_v, '4')
import numpy as np
import nibabel as nib
from scipy.io import loadmat
CODE = os.path.dirname(os.path.abspath(__file__))
NAS_ROOT = settings.scratch('pipeline')
COHORT_JSON = os.environ.get('PFBN_COHORT', '')
COHORT = json.load(open(COHORT_JSON)) if COHORT_JSON else {'name': 'none', 'out': settings.tmp('pfbn_none'), 'subjects': []}
NAME = COHORT['name']
EXP = COHORT['out']
NAS = COHORT.get('nas', os.path.join(NAS_ROOT, NAME))
DENOISE = COHORT.get('denoise', 'none')
DROP = int(COHORT.get('drop', 5))
RESULTS = os.path.join(EXP, 'results')
LOGS = os.path.join(EXP, 'logs')
TMP_ROOT = os.path.join(os.environ.get('PFBN_STACKS_ROOT', settings.tmp('pfbn_stacks')), NAME)
STACKS = os.path.join(TMP_ROOT, 'stacks')
GRAD = os.path.join(NAS, 'gradients')
MODELS = os.path.join(TMP_ROOT, 'pfbn_models')
WORK = os.path.join(EXP, 'pfbn_work')
TMPFS = os.path.join(os.environ.get('PFBN_TMPFS', settings.tmp('pfbn_cohort')), NAME)
TEMPLATES = settings.path('templates')
S4TMP = os.path.join(TEMPLATES, 'tmp')
PFBN_COPY1 = settings.code('pfbn', 'models')
CHECKPOINTS = settings.path('checkpoints')
MODES = settings.out('eigenmodes')
MICAPIPE = settings.path('micapipe_labels')
WB = settings.get('wb_command')
N_PC = 150
(NL, NR, NCORT, NV) = (29696, 29716, 59412, 32492)
HEMIS = ('lh', 'rh')
HEMI_LETTER = {'lh': 'L', 'rh': 'R'}
HEMI_OFF = {'lh': 0, 'rh': NV}
HEMI_N = {'lh': NL, 'rh': NR}
HEMI_SL = {'lh': slice(0, NL), 'rh': slice(NL, NCORT)}
K_GRAD = 15.0
if COHORT_JSON:
    for _d in (RESULTS, LOGS, STACKS, MODELS, WORK):
        os.makedirs(_d, exist_ok=True)

def log(*a):
    print(time.strftime('[%Y-%m-%d %H:%M:%S] ') + f'[{NAME}] ' + ' '.join((str(x) for x in a)), flush=True)

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
_MED = None

def medial():
    global _MED
    if _MED is None:
        _MED = loadmat(os.path.join(S4TMP, 'fs_LR_32k_medial_mask.mat'))['medial_mask'][:, 0].astype(int)
    return _MED

def cortex_idx(hemi):
    m = medial()[HEMI_OFF[hemi]:HEMI_OFF[hemi] + NV]
    idx = np.where(m == 1)[0]
    assert len(idx) == HEMI_N[hemi]
    return idx

def cortex_to_full(values_cortex, fill=np.nan):
    v = np.asarray(values_cortex)
    m = medial()
    out = np.full(64984, fill, dtype=np.result_type(v.dtype, np.asarray(fill).dtype))
    out[m == 1] = v
    return out

def sphere(hemi):
    g = nib.load(os.path.join(S4TMP, f'S1200.{HEMI_LETTER[hemi]}.sphere.32k_fs_LR.surf.gii'))
    return (g.darrays[0].data.astype(np.float64), g.darrays[1].data.astype(np.int64))

def hemi_edges_cortex(hemi):
    return np.load(os.path.join(S4TMP, f'{hemi}_edges.npy')).astype(np.int64)

def subjects():
    return [s['sid'] for s in COHORT['subjects']]

def runs_of(sid):
    for s in COHORT['subjects']:
        if s['sid'] == sid:
            return [r if isinstance(r, dict) else {'path': r} for r in s['runs']]
    raise KeyError(sid)

def read_cortex(path):
    img = nib.load(path)
    ax = img.header.get_axis(1)
    d = np.asarray(img.dataobj, dtype=np.float32)
    cols = []
    for (name, hemi) in (('CIFTI_STRUCTURE_CORTEX_LEFT', 'lh'), ('CIFTI_STRUCTURE_CORTEX_RIGHT', 'rh')):
        for (nm, sl, bm) in ax.iter_structures():
            if nm == name:
                assert np.array_equal(np.asarray(bm.vertex), cortex_idx(hemi)), (path, name)
                cols.append(d[:, sl])
                break
        else:
            raise ValueError(f'{path}: no {name}')
    return np.concatenate(cols, 1)

def load_run(run):
    raw = read_cortex(run['path'])
    if DENOISE == 'none':
        n = np.linalg.norm(raw, axis=0, keepdims=True)
        n[n == 0] = 1
        x = (raw / n)[DROP:]
        x = x[run.get('start', 0):run.get('stop', None)]
        return (np.ascontiguousarray(x, dtype=np.float32), x, {'frames': int(x.shape[0])})
    x = raw[DROP:][run.get('start', 0):run.get('stop', None)].astype(np.float64)
    nr = np.linalg.norm(x, axis=0, keepdims=True)
    nr[nr == 0] = 1
    raw_l2 = (x / nr).astype(np.float32)
    T = x.shape[0]
    t = np.linspace(-1, 1, T)
    D = np.stack([np.ones(T), t], 1)
    x = x - D @ np.linalg.lstsq(D, x, rcond=None)[0]
    if DENOISE == 'detrend_gsr':
        g = x.mean(1, keepdims=True)
        g = g - g.mean()
        x = x - g @ (g.T @ x / float(g.T @ g))
    n = np.linalg.norm(x, axis=0, keepdims=True)
    n[n == 0] = 1
    return ((x / n).astype(np.float32), raw_l2, {'frames': int(T)})

def unit_cols(X, torch_mod=None):
    if torch_mod is not None:
        n = X.norm(dim=0, keepdim=True)
        n[n == 0] = 1.0
        return X / n
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
        generate_dlabel_from_labels(lab, out_path, TEMPLATES)
    return out_path

def micapipe():
    z = np.load(MICAPIPE)
    k = [k for k in z.files][0]
    return np.asarray(z[k]).astype(np.int64).ravel()
