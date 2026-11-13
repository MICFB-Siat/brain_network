import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
from pfbn import settings
import os, sys, json, time
(GROUP, COH) = (sys.argv[1], sys.argv[2])
sys.path.insert(0, settings.code('evaluation', 'dcbc'))
sys.path.insert(0, settings.code('pfbn', 'pipeline'))
sys.path.insert(0, settings.code('evaluation', 'dcbc'))
import numpy as np, pandas as pd, torch
import cohort_lib as R
import common as PC
import dcbc as D
EXP = settings.out('boundary_distance')
RES = os.path.join(EXP, 'results')
SIX = settings.path('cato_atlases')
CATO = np.asarray(np.load(os.path.join(SIX, 'assets', 'lausanne250_labels.npz'))['labels']).astype(np.int64)
MICA = R.micapipe()
assert CATO.shape == MICA.shape == (R.NCORT,), (CATO.shape, MICA.shape)
ATL = {'CATO': CATO, 'Micapipe': MICA}
pairs = {h: D.HemiPairs(h) for h in ('lh', 'rh')}
cb = f'{COH}_B'
cj = json.load(open(os.path.join(R.RUNNER, 'cohorts', f'{cb}.json')))
os.environ['PFBN_COHORT'] = os.path.join(R.RUNNER, 'cohorts', f'{cb}.json')
PC.DENOISE = cj.get('denoise', 'none')
PC.DROP = int(cj.get('drop', 5))
rows = []
t0 = time.time()
for s in cj['subjects']:
    runs = [r if isinstance(r, dict) else {'path': r} for r in s['runs']]
    X = np.concatenate([(lambda x: x - x.mean(0))(PC.load_run(r)[0]) for r in runs], 0)
    Xt = torch.from_numpy(X).to(D.dev)
    res = {}
    for (h, sl) in (('lh', slice(0, R.NL)), ('rh', slice(R.NL, R.NCORT))):
        P = pairs[h]
        P.set_subject(Xt[:, sl])
        for (k, lab) in ATL.items():
            res.setdefault(k, []).append(P.evaluate(lab[sl])['DCBC_weighted'])
    for (k, v) in res.items():
        rows.append(dict(subject=s['sid'], group=GROUP, atlas=k, DCBC=float(np.mean(v)), frames=int(X.shape[0])))
    del Xt
    torch.cuda.empty_cache()
pd.DataFrame(rows).to_csv(os.path.join(RES, f'dcbc_cato_{GROUP}.csv'), index=False)
print(f"DCBC_CATO_END {GROUP} {len(cj['subjects'])} subjects {time.time() - t0:.0f}s", flush=True)
