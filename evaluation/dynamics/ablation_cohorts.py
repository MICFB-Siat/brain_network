import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
from pfbn import settings
import json, os
RU = settings.out()
E = settings.out('dynamics_ablation')
COH = ['life_children', 'life_adolescents', 'life_young_adults', 'life_older_adults', 'pd_NC', 'pd_PD']
os.makedirs(os.path.join(E, 'cohorts'), exist_ok=True)
for c in COH:
    j = json.load(open(os.path.join(RU, 'cohorts', f'{c}_A.json')))
    assert j['name'] == f'{c}_A', j['name']
    j['name'] = f'ablation_{c}_A'
    j['out'] = os.path.join(E, 'atlases', f'{c}_A')
    j['nas'] = os.path.join(settings.tmp('ablation_nas'), f'ablation_{c}_A')
    j['ablation_source'] = os.path.join(RU, 'cohorts', f'{c}_A.json')
    json.dump(j, open(os.path.join(E, 'cohorts', f'ablation_{c}_A.json'), 'w'), indent=1)
    print(c, len(j['subjects']))
