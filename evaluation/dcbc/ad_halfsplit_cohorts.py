import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
from pfbn import settings
import json, os
import nibabel as nib
RU = settings.out()
E = settings.out('ad_halfsplit')
DROP = 5
j = json.load(open(os.path.join(RU, 'cohorts', 'ad_all.json')))
assert int(j.get('drop', 5)) == DROP
subs = [s for s in j['subjects'] if s['group'] == 'case']
T = {}
(A, Bh) = ([], [])
for s in subs:
    (ra, rb) = ([], [])
    for p in s['runs']:
        p = p['path'] if isinstance(p, dict) else p
        T[p] = nib.load(p).shape[0]
        n = (T[p] - DROP) // 2
        ra.append({'path': p, 'start': 0, 'stop': n})
        rb.append({'path': p, 'start': n, 'stop': 2 * n})
    A.append(dict(s, runs=ra))
    Bh.append(dict(s, runs=rb))
os.makedirs(os.path.join(E, 'cohorts'), exist_ok=True)
for (h, ss) in (('A', A), ('B', Bh)):
    name = f'adsplit_ad_case_{h}'
    json.dump({'name': name, 'out': os.path.join(E, 'atlases', f'ad_case_{h}'), 'denoise': j.get('denoise', 'detrend_gsr'), 'drop': DROP, 'nas': os.path.join(settings.tmp('adsplit_nas'), name), 'adsplit_source': os.path.join(RU, 'cohorts', 'ad_all.json'), 'subjects': ss}, open(os.path.join(E, 'cohorts', f'{name}.json'), 'w'), indent=1)
json.dump(T, open(os.path.join(E, 'cohorts', 'frames.json'), 'w'), indent=1)
print(len(subs), 'patients; frames', sorted(set(T.values())), 'denoise', j.get('denoise'))
