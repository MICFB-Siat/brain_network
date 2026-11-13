import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
from pfbn import settings
import json, os, sys
CJ = settings.out('cohorts')
os.makedirs(CJ, exist_ok=True)
PRIOR = settings.path('hcp_subjects')
P = os.path.join(settings.path('hcp_3t'), '{s}_3T_rfMRI_REST_fix/{s}/MNINonLinear/Results/{r}/{r}_Atlas_MSMAll_hp2000_clean.dtseries.nii')
subs = json.load(open(os.path.join(PRIOR, 'selected_subjects.json')))
sess = sys.argv[1] if len(sys.argv) > 1 else 'REST1'
runs = [f'rfMRI_{sess}_LR', f'rfMRI_{sess}_RL']
out = {'name': f'hcp_{sess.lower()}_val', 'out': os.path.join(settings.out('validation_hcp'), f'hcp_{sess.lower()}'), 'denoise': 'none', 'drop': 5, 'subjects': [{'sid': s, 'group': 'young_adults', 'runs': [P.format(s=s, r=r) for r in runs]} for s in subs]}
miss = [p for x in out['subjects'] for p in x['runs'] if not os.path.exists(p)]
print('missing', len(miss), miss[:3])
json.dump(out, open(f'{CJ}/hcp_{sess.lower()}_val.json', 'w'), indent=1)
