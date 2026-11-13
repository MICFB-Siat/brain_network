import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
from pfbn import settings
import os, numpy as np, pandas as pd
R = settings.out()
S = os.path.join(R, 'boundary_distance', 'results')
OUT = os.path.join(R, 'prism_export', 'results')
os.makedirs(OUT, exist_ok=True)
G = [('children', 'Children'), ('adolescents', 'Adolescents'), ('young_adults', 'Young adults'), ('older_adults', 'Older adults'), ('NC', 'NC'), ('PD', 'PD')]
d = pd.read_csv(os.path.join(S, 'dcbc_groups_with_cato.csv'))
A = [('CATO', 'CATO'), ('Micapipe', 'Micapipe'), ('Other-group PFBN v5 (mean)', 'Ours (other group)'), ('Own PFBN v5', 'Ours (own group)')]
cols = {}
for (g, gn) in G:
    x = d[d.group == g].pivot(index='subject', columns='atlas', values='DCBC')
    for (a, an) in A:
        cols[f'{gn} | {an}'] = x[a].values
n = max((len(v) for v in cols.values()))
pd.DataFrame({k: np.r_[v, [np.nan] * (n - len(v))] for (k, v) in cols.items()}).to_csv(os.path.join(OUT, 'R1_DCBC_prism.csv'), index=False, float_format='%.6f')
t = pd.read_csv(os.path.join(S, 'dist_f1_groups.csv'))
t = t[t.roi == 'Whole cortex']
nl = np.load(os.path.join(S, 'dist_f1_nulls.npz'))
print('comparisons:', sorted(t.comparison.unique()))
B = [('CATO', 'CATO'), ('Micapipe', 'Micapipe'), ('PFBN v5', 'Ours')]
cols = {}
for (g, gn) in G:
    for (a, an) in B:
        obs = float(t[(t.group == g) & (t.comparison == a)].dist_mm.iloc[0])
        cols[f'{gn} | {an} observed'] = np.r_[obs, [np.nan] * 999]
        cols[f'{gn} | {an} spin null'] = nl[f'{g}|Whole cortex|{a}|dist']
pd.DataFrame(cols).to_csv(os.path.join(OUT, 'R1_boundary_distance_prism.csv'), index=False, float_format='%.6f')
print('ok')
