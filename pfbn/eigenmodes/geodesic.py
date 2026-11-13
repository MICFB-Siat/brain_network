import argparse
import os
import time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
import common as C

def geodesic_cache_path(hemi, limit=35.0):
    return os.path.join(C.CACHE_DIR, f'geodesic_{hemi}_{limit:g}mm.npz')

def build_geodesic(hemi, limit=35.0, chunk=1500, force=False, verbose=True):
    out = geodesic_cache_path(hemi, limit)
    if os.path.exists(out) and (not force):
        return out
    t0 = time.time()
    (coords, _) = C.load_surf(hemi, 'midthickness')
    ai = C.cortex_axis()
    cortex = ai[hemi]
    mask = np.zeros(C.N_VERT, bool)
    mask[cortex] = True
    e = C.edges(hemi)
    e = e[mask[e[:, 0]] & mask[e[:, 1]]]
    w = np.linalg.norm(coords[e[:, 0]] - coords[e[:, 1]], axis=1)
    G = sp.coo_matrix((w, (e[:, 0], e[:, 1])), shape=(C.N_VERT, C.N_VERT))
    G = (G + G.T).tocsr()
    (ii, jj, dd) = ([], [], [])
    for s in range(0, len(cortex), chunk):
        src = cortex[s:s + chunk]
        D = dijkstra(G, directed=False, indices=src, limit=limit)
        (r, c) = np.nonzero(np.isfinite(D) & (D > 0))
        keep = src[r] < c
        ii.append(src[r[keep]].astype(np.int32))
        jj.append(c[keep].astype(np.int32))
        dd.append(D[r[keep], c[keep]].astype(np.float32))
        if verbose and s // chunk % 5 == 0:
            C.log(f'geodesic {hemi}: {s + len(src)}/{len(cortex)} sources, {sum((len(x) for x in ii)):,} pairs, {time.time() - t0:.0f}s')
    i = np.concatenate(ii)
    j = np.concatenate(jj)
    d = np.concatenate(dd)
    o = np.lexsort((j, i))
    (i, j, d) = (i[o], j[o], d[o])
    el = time.time() - t0
    np.savez(out, i=i, j=j, dist=d, hemi=hemi, limit=limit, n_cortex=len(cortex), seconds=el, surface=C.surf_path(hemi, 'midthickness'), note='graph Dijkstra on mesh edges, medial wall excluded, i<j within hemisphere')
    if verbose:
        C.log(f'geodesic {hemi}: {len(i):,} pairs (<= {limit} mm) in {el:.1f}s -> {out}  ({os.path.getsize(out) / 1000000.0:.0f} MB)')
    return out

def load_geodesic(hemi, limit=35.0):
    z = np.load(build_geodesic(hemi, limit, verbose=True))
    return (z['i'], z['j'], z['dist'])

def load_geodesic_cortex(limit=35.0):
    ai = C.cortex_axis()
    (I, J, D, H) = ([], [], [], [])
    for (k, h) in enumerate(C.HEMIS):
        (i, j, d) = load_geodesic(h, limit)
        f2c = ai['full2cortex']
        off = C.HEMI_OFFSET[h]
        (ic, jc) = (f2c[i.astype(np.int64) + off], f2c[j.astype(np.int64) + off])
        assert (ic >= 0).all() and (jc >= 0).all()
        I.append(ic.astype(np.int32))
        J.append(jc.astype(np.int32))
        D.append(d)
        H.append(np.full(len(d), k, np.int8))
    return (np.concatenate(I), np.concatenate(J), np.concatenate(D), np.concatenate(H))
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=float, default=35.0)
    ap.add_argument('--chunk', type=int, default=1500)
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    for h in C.HEMIS:
        p = build_geodesic(h, a.limit, a.chunk, a.force)
        z = np.load(p)
        print(f"{h}: {len(z['i']):,} pairs, n_cortex={int(z['n_cortex'])}, built in {float(z['seconds']):.1f}s, mean pairs/vertex={2 * len(z['i']) / int(z['n_cortex']):.0f}, dist range {z['dist'].min():.2f}-{z['dist'].max():.2f}")
