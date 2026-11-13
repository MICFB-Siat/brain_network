import os
import numpy as np
from PIL import Image
from mrf_prepare import compute_fmri_average, current_dir_path as _mrf_dir

def compute_avgconnectivity_fixed(dataset_name, dataset_dir, work_dir):
    if dataset_name.endswith('_lh'):
        V = 29696
    elif dataset_name.endswith('_rh'):
        V = 29716
    else:
        raise ValueError(f'dataset name must end with _lh/_rh: {dataset_name}')
    work_dir = os.path.abspath(work_dir)
    if os.path.abspath(_mrf_dir) != work_dir:
        raise RuntimeError(f'work_dir ({work_dir}) must be the directory containing mrf_prepare.py ({_mrf_dir})')
    out_dir = os.path.join(work_dir, 'tmp', dataset_name, 'FeatExt_Prepare')
    tiff_path = os.path.join(out_dir, 'avgconnectivity.tiff')
    index_path = os.path.join(out_dir, 'index.npy')
    if os.path.exists(tiff_path) and os.path.exists(index_path):
        print(f'  avgconnectivity.tiff and index.npy already exist for {dataset_name}; skipping.')
        return True
    os.makedirs(out_dir, exist_ok=True)
    (fmri_avg, n_files, T, skipped) = compute_fmri_average(dataset_name, dataset_dir)
    fmri_avg = np.asarray(fmri_avg, dtype=np.float64)
    assert fmri_avg.shape[1] == V, (fmri_avg.shape, V)
    print(f'  fmri_avg: shape {fmri_avg.shape}, from {n_files} files, skipped {len(skipped)}')
    index = np.linspace(0, V - 1, 1024).astype(np.int64)
    sub = fmri_avg[:, index]
    r = np.corrcoef(sub.T)
    nbad = int(np.isnan(r).sum())
    if nbad:
        print(f'  WARNING: {nbad} NaN entries in the 1024x1024 correlation (constant vertices); set to 0')
        r = np.nan_to_num(r, nan=0.0)
    r = np.clip(r, -1.0, 1.0)
    img = ((r + 1.0) / 2.0 * 255.0).astype(np.uint8)
    rgb = np.stack([img, img, img], axis=-1)
    Image.fromarray(rgb, mode='RGB').save(tiff_path, compression=None)
    np.save(index_path, index)
    print(f'  saved {tiff_path} (min {img.min()}, max {img.max()}, mean {img.mean():.2f}) and {index_path}')
    return True
