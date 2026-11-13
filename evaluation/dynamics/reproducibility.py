import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
from pfbn import settings
import os, sys, glob, json
sys.path.insert(0, settings.code('pfbn', 'pipeline'))
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
import common as C
E = settings.out('reproducibility')
B = settings.out()
METH = {'With dynamics': 'v5_stim_t03_r30', 'Without dynamics': 'v4_stim_t03_c1'}
ED = {h: C.hemi_edges_cortex(h) for h in C.HEMIS}

def adj(h):
    e = ED[h]
    n = C.HEMI_N[h]
    return sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
A1 = {h: (adj(h) + adj(h).T).tocsr() for h in C.HEMIS}

def boundary(lab):
    out = []
    for h in C.HEMIS:
        l = lab[C.HEMI_SL[h]]
        e = ED[h]
        d = l[e[:, 0]] != l[e[:, 1]]
        b = np.zeros(len(l), bool)
        b[e[d, 0]] = True
        b[e[d, 1]] = True
        out.append(b)
    return np.concatenate(out)

def dil(b):
    out = []
    for h in C.HEMIS:
        x = b[C.HEMI_SL[h]].astype(float)
        out.append(x + A1[h] @ x > 0)
    return np.concatenate(out)

def bdice(a, b):
    (ba, bb) = (boundary(a), boundary(b))
    return float(((ba & dil(bb)).sum() + (bb & dil(ba)).sum()) / (ba.sum() + bb.sum()))

def comps(lab):
    n = 0
    isl = 0
    for h in C.HEMIS:
        l = lab[C.HEMI_SL[h]]
        e = ED[h]
        same = l[e[:, 0]] == l[e[:, 1]]
        g = sp.coo_matrix((np.ones(same.sum()), (e[same, 0], e[same, 1])), shape=(len(l), len(l)))
        (k, cc) = connected_components(g, directed=False)
        n += k
        df = pd.DataFrame({'l': l, 'c': cc})
        s = df.groupby(['l', 'c']).size().reset_index(name='n')
        big = s.groupby('l').n.transform('max')
        isl += int(s.n[s.n < big].sum())
    return (n, isl)

def load(d, m, clean=True):
    p = f"{d}/labels_{m}{('_clean' if clean else '')}.npy"
    return np.load(p).astype(np.int64) if os.path.exists(p) else None

def pair_dirs():
    rows = []
    for c in ['life_children', 'life_adolescents', 'life_young_adults', 'life_older_adults', 'pd_NC', 'pd_PD', 'gsp']:
        a = f'{B}/dynamics_ablation/atlases/{c}_A/results' if c != 'gsp' else f'{B}/atlases/gsp_A/results'
        rows.append(dict(part='A', cohort=c, n=-1, draw=-1, dA=a, dB=f'{E}/atlases/{c}_B/results'))
    for f in sorted(glob.glob(f'{E}/atlases/*_n*_d*_A/results')):
        nm = f.split('/')[-2][:-2]
        (c, n, d) = nm.rsplit('_', 2)
        rows.append(dict(part='B', cohort=c, n=int(n[1:]), draw=int(d[1:]), dA=f, dB=f.replace('_A/results', '_B/results')))
    return rows
rows = []
for r in pair_dirs():
    if not (os.path.exists(r['dB'] + '/DONE') and (r['part'] == 'A' or os.path.exists(r['dA'] + '/DONE'))):
        continue
    for (meth, m) in METH.items():
        (a, b) = (load(r['dA'], m), load(r['dB'], m))
        if a is None or b is None:
            continue
        (ca, ia) = comps(load(r['dA'], m, False))
        (cb, ib) = comps(load(r['dB'], m, False))
        rows.append(dict(part=r['part'], cohort=r['cohort'], n=r['n'], draw=r['draw'], method=meth, ARI=adjusted_rand_score(a, b), AMI=adjusted_mutual_info_score(a, b), boundary_dice=bdice(a, b), comps_raw=(ca + cb) / 2, island_vertices_raw=(ia + ib) / 2))
df = pd.DataFrame(rows)
df.to_csv(f'{E}/results/reliability.csv', index=False)
chk = []
for c in ['life_children', 'life_adolescents', 'life_young_adults', 'life_older_adults', 'pd_NC', 'pd_PD', 'gsp']:
    p = f'{E}/atlases/{c}_B/results/labels_v5_stim_t03_r30_clean.npy'
    q = f'{B}/atlases/{c}_B/results/labels_v5_stim_t03_r30_clean.npy'
    if os.path.exists(p) and os.path.exists(q):
        chk.append(dict(cohort=c, ARI_vs_frozen=adjusted_rand_score(np.load(p), np.load(q))))
pd.DataFrame(chk).to_csv(f'{E}/results/rebuild_check.csv', index=False)
pd.set_option('display.width', 200)
print(pd.DataFrame(chk))
print(df.pivot_table(index=['part', 'cohort', 'n'], columns='method', values=['ARI', 'boundary_dice', 'comps_raw'], aggfunc='mean').round(3))
