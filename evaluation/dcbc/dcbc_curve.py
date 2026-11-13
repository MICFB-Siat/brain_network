import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
from pfbn import settings
import os, sys, json, time, types
(GROUP, COH) = (sys.argv[1], sys.argv[2])
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[k] = '4'
EXP = settings.out('dcbc_curve')
RES = os.path.join(EXP, 'results')
sys.path.insert(0, settings.code('evaluation', 'dcbc'))
sys.path.insert(0, settings.code('pfbn', 'pipeline'))
sys.path.insert(0, settings.code('evaluation', 'dcbc'))
import numpy as np, pandas as pd, torch
import cohort_lib as R
import common as PC
import dcbc as D
pkg = types.ModuleType('DCBC')
pkg.__path__ = [settings.code('third_party', 'DCBC')]
sys.modules['DCBC'] = pkg
from DCBC.dcbc import compute_DCBC
torch.set_default_device('cuda')
SIX = settings.path('cato_atlases')
CATO = np.asarray(np.load(os.path.join(SIX, 'assets', 'lausanne250_labels.npz'))['labels']).astype(np.int64)
MICA = R.micapipe()
OURS = np.load(os.path.join(R.ATL, COH + '_A', R.FILES['PFBN v5'])).astype(np.int64)
ATL = {'CATO': CATO, 'Micapipe': MICA, 'Our Method': OURS}
for (k, v) in ATL.items():
    assert v.shape == (R.NCORT,), (k, v.shape)
pairs = {h: D.HemiPairs(h) for h in ('lh', 'rh')}
dist = {}
for h in ('lh', 'rh'):
    z = np.load(D.build_geodesic(h))
    d = z['dist']
    keep = (d > 0) & (d <= D.MAX_DIST)
    assert np.array_equal(z['i'][keep].astype(np.int64), pairs[h].I_np)
    dist[h] = torch.from_numpy(d[keep].astype(np.float32)).to(D.dev)
cb = f'{COH}_B'
cj = json.load(open(os.path.join(R.RUNNER, 'cohorts', f'{cb}.json')))
os.environ['PFBN_COHORT'] = os.path.join(R.RUNNER, 'cohorts', f'{cb}.json')
PC.DENOISE = cj.get('denoise', 'none')
PC.DROP = int(cj.get('drop', 5))
rows = []
bins = {}
t0 = time.time()
for (si, s) in enumerate(cj['subjects']):
    runs = [r if isinstance(r, dict) else {'path': r} for r in s['runs']]
    X = np.concatenate([(lambda x: x - x.mean(0))(PC.load_run(r)[0]) for r in runs], 0)
    Xt = torch.from_numpy(X).to(D.dev)
    res_iv = {}
    res_au = {}
    for (h, sl) in (('lh', slice(0, R.NL)), ('rh', slice(R.NL, R.NCORT))):
        P = pairs[h]
        P.set_subject(Xt[:, sl])
        Zh = Xt[:, sl] - Xt[:, sl].mean(0, keepdim=True)
        nrm = Zh.norm(dim=0)
        valid = nrm > 0
        func = (Zh / torch.where(valid, nrm, torch.ones_like(nrm))).T.contiguous()
        for (k, lab) in ATL.items():
            r = P.evaluate(lab[sl])
            res_iv.setdefault(k, []).append(r['DCBC_weighted'])
            for name in ('within', 'between', 'n_within', 'n_between', 'weights'):
                bins.setdefault(f'{k}|{h}|{name}', []).append(r[name])
            lt = torch.from_numpy(lab[sl]).to(D.dev)
            ok = (lt[P.I] > 0) & (lt[P.J] > 0) & P.pair_ok
            sp = torch.sparse_coo_tensor(torch.stack([P.I[ok], P.J[ok]]), dist[h][ok], (P.n, P.n))
            A = compute_DCBC(maxDist=35, binWidth=1, parcellation=lt.float(), func=func, dist=sp, weighting=True, backend='torch')
            res_au.setdefault(k, []).append(float(A['DCBC']))
            bins.setdefault(f'author|{k}|{h}|corr_within', []).append(torch.stack(A['corr_within']).double().cpu().numpy())
            bins.setdefault(f'author|{k}|{h}|corr_between', []).append(torch.stack(A['corr_between']).double().cpu().numpy())
            bins.setdefault(f'author|{k}|{h}|num_within', []).append(torch.stack(A['num_within']).double().cpu().numpy())
            bins.setdefault(f'author|{k}|{h}|num_between', []).append(torch.stack(A['num_between']).double().cpu().numpy())
            bins.setdefault(f'author|{k}|{h}|DCBC', []).append(float(A['DCBC']))
            del sp, A
        del Zh, func
    for k in ATL:
        rows.append(dict(subject=s['sid'], group=GROUP, atlas=k, DCBC_IV=float(np.mean(res_iv[k])), DCBC_author=float(np.mean(res_au[k])), frames=int(X.shape[0])))
    del Xt
    torch.cuda.empty_cache()
    R.log(f"{GROUP} {si + 1}/{len(cj['subjects'])} {s['sid']}: " + ', '.join((f'{k} {np.mean(res_iv[k]):.4f}/{np.mean(res_au[k]):.4f}' for k in ATL)))
pd.DataFrame(rows).to_csv(os.path.join(RES, f'dcbc_{GROUP}.csv'), index=False)
np.savez_compressed(os.path.join(RES, f'bins_{GROUP}.npz'), subjects=np.array([s['sid'] for s in cj['subjects']]), bin_edges=np.arange(0, 35.5, 1.0), **{k: np.stack(v) for (k, v) in bins.items()})
R.log(f"DCBC_CURVE_END {GROUP} {len(cj['subjects'])} subjects {time.time() - t0:.0f}s")
