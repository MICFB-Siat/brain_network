import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
from pfbn import settings
import os
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '4')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', os.environ.get('GY_GPU', '1'))
import json
import subprocess
import time
import numpy as np
import nibabel as nib
ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = settings.path('gradient_cohort', 'analysis', 'cache')
TMP_DIR = settings.tmp('analysis')
for _d in (CACHE_DIR, TMP_DIR):
    os.makedirs(_d, exist_ok=True)
_FW = settings.path('framework')
_EXP = settings.path('lifespan_reference')
PY = _EXP + '/env/bin/python'
WB = settings.get('wb_command')
CONTE69_DIR = settings.path('conte69')
FSLR_DIR = settings.path('fs_lr')
MICAPIPE_DIR = _FW + '/Micapipe'
HCP_ROOT = settings.path('hcp_3t')
GRADIENT_DIR = _EXP + '/results/gradient'
GROUP_MAP = GRADIENT_DIR + '/group_summaries/young_adults/group_boundary_density.dscalar.nii'
HCP_RUNS = ('rfMRI_REST1_LR', 'rfMRI_REST1_RL', 'rfMRI_REST2_LR', 'rfMRI_REST2_RL')
N_VERT = 32492
HEMIS = ('L', 'R')
HEMI_OFFSET = {'L': 0, 'R': N_VERT}

def log(*a):
    print(time.strftime('[%H:%M:%S]'), *a, flush=True)

def surf_path(hemi, kind='midthickness'):
    if kind == 'sphere':
        return f'{FSLR_DIR}/fs_LR.32k.{hemi}.sphere.surf.gii'
    return f'{CONTE69_DIR}/Conte69.{hemi}.{kind}.32k_fs_LR.surf.gii'

def load_surf(hemi, kind='midthickness'):
    g = nib.load(surf_path(hemi, kind))
    coords = np.asarray(g.darrays[0].data, dtype=np.float64)
    faces = np.asarray(g.darrays[1].data, dtype=np.int64)
    assert coords.shape[0] == N_VERT, coords.shape
    return (coords, faces)
_EDGE_CACHE = {}

def edges(hemi):
    if hemi not in _EDGE_CACHE:
        (_, f) = load_surf(hemi, 'midthickness')
        e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [0, 2]]])
        e = np.sort(e, axis=1)
        e = np.unique(e, axis=0)
        _EDGE_CACHE[hemi] = e
    return _EDGE_CACHE[hemi]

def adjacency(hemi, mask=None):
    import scipy.sparse as sp
    e = edges(hemi)
    if mask is not None:
        e = e[mask[e[:, 0]] & mask[e[:, 1]]]
    n = N_VERT
    a = sp.coo_matrix((np.ones(len(e), bool), (e[:, 0], e[:, 1])), shape=(n, n))
    a = (a + a.T).tocsr()
    return a

def neighbor_lists(hemi, mask=None):
    a = adjacency(hemi, mask)
    return [a.indices[a.indptr[i]:a.indptr[i + 1]] for i in range(N_VERT)]
_AXIS_CACHE = {}

def cortex_axis(cifti_path=GROUP_MAP):
    if cifti_path in _AXIS_CACHE:
        return _AXIS_CACHE[cifti_path]
    img = nib.load(cifti_path)
    ax = img.header.get_axis(img.ndim - 1)
    out = {}
    for (name, sl, bm) in ax.iter_structures():
        if name == 'CIFTI_STRUCTURE_CORTEX_LEFT':
            out['L'] = np.asarray(bm.vertex, dtype=np.int64)
            out['sl_L'] = sl
        elif name == 'CIFTI_STRUCTURE_CORTEX_RIGHT':
            out['R'] = np.asarray(bm.vertex, dtype=np.int64)
            out['sl_R'] = sl
    assert out['sl_L'].start == 0 and out['sl_R'].start == len(out['L']), 'unexpected axis layout'
    (out['n_L'], out['n_R']) = (len(out['L']), len(out['R']))
    out['n_cortex'] = out['n_L'] + out['n_R']
    c2f = np.concatenate([out['L'], out['R'] + N_VERT])
    f2c = np.full(2 * N_VERT, -1, np.int64)
    f2c[c2f] = np.arange(len(c2f))
    out['cortex2full'] = c2f
    out['full2cortex'] = f2c
    out['cortex_mask_full'] = f2c >= 0
    out['bm'] = ax[:out['n_cortex']]
    _AXIS_CACHE[cifti_path] = out
    return out

def cortex_to_full(values, fill=np.nan, axis_info=None):
    ai = axis_info or cortex_axis()
    values = np.asarray(values)
    out = np.full(values.shape[:-1] + (2 * N_VERT,), fill, dtype=np.result_type(values.dtype, type(fill)) if fill is not None else values.dtype)
    out[..., ai['cortex2full']] = values
    return out

def full_to_cortex(values_full, axis_info=None):
    ai = axis_info or cortex_axis()
    return np.asarray(values_full)[..., ai['cortex2full']]

def load_cifti_row(path, row=0):
    img = nib.load(path)
    ai = cortex_axis(path)
    d = np.asarray(img.dataobj[row, :ai['n_cortex']], dtype=np.float64)
    return cortex_to_full(d, np.nan, ai)

def load_parcellation(name):
    if name == 'schaefer200' or name == 'schaefer400':
        k = name[-3:]
        lab = []
        for h in HEMIS:
            g = nib.load(f'{FSLR_DIR}/Schaefer_2018/Schaefer2018_17Networks_{k}.32k.{h}.label.gii')
            lab.append(np.asarray(g.darrays[0].data, dtype=np.int64))
        return np.concatenate(lab)
    if name == 'gordon333':
        p = f'{FSLR_DIR}/Gordon333_FreesurferSubcortical.32k_fs_LR.dlabel.nii'
        img = nib.load(p)
        ai = cortex_axis(p)
        d = np.asarray(img.dataobj[0, :ai['n_cortex']]).round().astype(np.int64)
        return cortex_to_full(d, 0, ai).astype(np.int64)
    if name == 'schaefer200_micapipe':
        return np.loadtxt(f'{MICAPIPE_DIR}/schaefer-200_conte69.csv', delimiter=',').round().astype(np.int64)
    if name.endswith('.npz'):
        return np.load(name)['labels_full'].astype(np.int64)
    if name.endswith('.dlabel.nii'):
        img = nib.load(name)
        ai = cortex_axis(name)
        d = np.asarray(img.dataobj[0, :ai['n_cortex']]).round().astype(np.int64)
        return cortex_to_full(d, 0, ai).astype(np.int64)
    raise ValueError(name)

def relabel_consecutive(labels):
    labels = np.asarray(labels)
    u = np.unique(labels[labels > 0])
    lut = np.zeros(int(labels.max()) + 1 if labels.size else 1, np.int64)
    lut[u] = np.arange(1, len(u) + 1)
    return lut[labels]

def write_dscalar(values_cortex, out_path, names=None, template=GROUP_MAP):
    values_cortex = np.atleast_2d(np.asarray(values_cortex, dtype=np.float32))
    ai = cortex_axis(template)
    names = names or [f'map{i}' for i in range(values_cortex.shape[0])]
    sax = nib.cifti2.ScalarAxis(names)
    img = nib.cifti2.Cifti2Image(values_cortex, header=(sax, ai['bm']))
    img.to_filename(out_path)
    return out_path

def write_dlabel(labels_full, out_path, names=None, seed=0):
    labels_full = np.asarray(labels_full).astype(np.int64)
    lab_c = full_to_cortex(labels_full)
    tmp_ds = out_path + '.tmp.dscalar.nii'
    write_dscalar(lab_c.astype(np.float32), tmp_ds, ['labels'])
    rng = np.random.RandomState(seed)
    keys = np.unique(lab_c[lab_c > 0])
    lst = out_path + '.labellist.txt'
    with open(lst, 'w') as f:
        for k in keys:
            nm = names[k] if names and k in names else f'parcel_{k}'
            (r, g, b) = rng.randint(30, 230, 3)
            f.write(f'{nm}\n{k} {r} {g} {b} 255\n')
    subprocess.run([WB, '-cifti-label-import', tmp_ds, lst, out_path, '-drop-unused-labels'], check=True)
    os.remove(tmp_ds)
    return out_path

def torch_device():
    import torch
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '4')))
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    log('WARNING: CUDA not available, falling back to CPU')
    return torch.device('cpu')

def save_json(obj, path):

    def _conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    with open(path, 'w') as f:
        json.dump(obj, f, indent=1, default=_conv)
