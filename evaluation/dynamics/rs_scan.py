import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
from pfbn import settings
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_common as C
from eval_common import log
import numpy as np, pandas as pd
sys.path.insert(1, settings.code('evaluation', 'dcbc'))
import ds005498_common as C2
from hcp_retest import Pairs
from concurrent.futures import ProcessPoolExecutor
RS = {'8.5 (fitted)': 'v5_stim_t03_clean', '15': 'v5_stim_t03_r15_clean', '20': 'v5_stim_t03_r20_clean', '30': 'v5_stim_t03_r30_clean', '45': 'v5_stim_t03_r45_clean'}
MODES = settings.out('eigenmodes')

def atlases():
    return {k: C.load_labels(os.path.join(C.MV2, 'v5', 'results', f'labels_{v}.npy')) for (k, v) in RS.items()}

def all_subjects(subs):
    import torch
    (_, runs) = C2.fit_subjects()
    atl = atlases()
    pairs = {h: Pairs(h) for h in C.HEMIS}
    rows = []
    for (i, s) in enumerate(subs):
        X = torch.from_numpy(C2.load_cortex_bold(runs[s][C2.REST_KEY])).cuda()
        for h in C.HEMIS:
            P = pairs[h]
            P.set_data(X[:, :C.NL] if h == 'lh' else X[:, C.NL:])
            for (k, lab) in atl.items():
                rows.append(dict(subject=s, hemi=h, r_s=k, DCBC=P.evaluate(C.hemi_lab(lab, h))[0]))
        log(f'{i + 1}/{len(subs)} {s}')
    return rows

def psf():
    z = np.load(os.path.join(MODES, 'modes_L.npz'))
    psi = z['psi'].astype(np.float64)
    lam = z['evals'].astype(np.float64)
    idx = z['idx']
    c = 8494
    out = {}
    for (name, r) in (('fitted', 8.5), ('r30', 30.0)):
        g = 1 / (1 + r ** 2 * lam) ** 2
        k = psi @ (g * psi[c])
        full = np.full(C.NV, np.nan)
        full[idx] = k / k.max()
        out[name] = full
    out['centre_fsLR'] = idx[c]
    return out

def main():
    t0 = time.time()
    (subs, _) = C2.fit_subjects()
    assert len(subs) == 32
    rows = all_subjects(subs)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(C.RES, 'ds_rs_scan_dcbc.csv'), index=False)
    print(df.groupby(['r_s', 'subject']).DCBC.mean().groupby('r_s').median())
    np.savez(os.path.join(C.SCR, 'kernel_psf.npz'), **psf())
    log(f'RS_END {time.time() - t0:.0f}s')
if __name__ == '__main__':
    main()
