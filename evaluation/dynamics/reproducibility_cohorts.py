import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
from pfbn import settings
import json, os, numpy as np
E = settings.out('reproducibility')
SRC = settings.out('cohorts')
NAS = settings.tmp('reproducibility', 'nas')
COH = ['life_children', 'life_adolescents', 'life_young_adults', 'life_older_adults', 'pd_NC', 'pd_PD']

def write(name, base, subs, half):
    d = dict(base)
    d['name'] = f'repro_{name}'
    d['out'] = os.path.join(settings.tmp('reproducibility', 'atlases'), name)
    d['repro_final'] = f'{E}/atlases/{name}'
    d['nas'] = f'{NAS}/repro_{name}'
    d['subjects'] = subs
    d['repro_source'] = f'{SRC}/{half}'
    json.dump(d, open(f'{E}/cohorts/repro_{name}.json', 'w'), indent=1)
partA = []
for c in COH + ['gsp']:
    b = json.load(open(f'{SRC}/{c}_B.json'))
    write(f'{c}_B', b, b['subjects'], f'{c}_B.json')
    partA.append(f'{c}_B')
partB = []
for c in ['life_children', 'life_adolescents', 'life_young_adults', 'life_older_adults']:
    a = json.load(open(f'{SRC}/{c}_A.json'))
    b = json.load(open(f'{SRC}/{c}_B.json'))
    sa = [s['sid'] for s in a['subjects']]
    assert sa == [s['sid'] for s in b['subjects']]
    for n in (5, 10, 20):
        for draw in range(3):
            idx = sorted(np.random.default_rng(93 + 100 * n + draw).choice(len(sa), n, replace=False).tolist())
            for (half, src) in (('A', a), ('B', b)):
                nm = f'{c}_n{n}_d{draw}_{half}'
                write(nm, src, [src['subjects'][i] for i in idx], f'{c}_{half}.json')
                partB.append(nm)
open(f'{E}/cohorts/partA.txt', 'w').write('\n'.join(partA) + '\n')
open(f'{E}/cohorts/partB.txt', 'w').write('\n'.join(partB) + '\n')
print(len(partA), len(partB))
