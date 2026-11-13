import argparse
import time
import numpy as np
import common as C
from dcbc import SpinRotations
_EDGES_FULL = None

def _edges_full():
    global _EDGES_FULL
    if _EDGES_FULL is None:
        ai = C.cortex_axis()
        es = []
        for h in C.HEMIS:
            e = C.edges(h) + C.HEMI_OFFSET[h]
            m = ai['cortex_mask_full']
            es.append(e[m[e[:, 0]] & m[e[:, 1]]])
        _EDGES_FULL = np.concatenate(es)
    return _EDGES_FULL

def border_interior(labels_full, unassigned_as_border=True):
    lab = np.asarray(labels_full)
    e = _edges_full()
    (la, lb) = (lab[e[:, 0]], lab[e[:, 1]])
    diff = la != lb
    if not unassigned_as_border:
        diff &= (la > 0) & (lb > 0)
    border = np.zeros(lab.shape, bool)
    border[e[diff, 0]] = True
    border[e[diff, 1]] = True
    border &= lab > 0
    interior = (lab > 0) & ~border & C.cortex_axis()['cortex_mask_full']
    return (border, interior)

def border_minus_interior(map_full, labels_full, unassigned_as_border=True):
    (b, i) = border_interior(labels_full, unassigned_as_border)
    m = np.asarray(map_full, dtype=np.float64)
    ok = np.isfinite(m)
    return (float(np.mean(m[b & ok]) - np.mean(m[i & ok])), int((b & ok).sum()), int((i & ok).sum()))

def alignment(map_full, labels_full, spins=None, n_rot=1000, unassigned_as_border=True, verbose=False):
    spins = spins or SpinRotations(max(n_rot, 1), 0)
    n_rot = min(n_rot, spins.n_rot)
    t0 = time.time()
    (raw, nb, ni) = border_minus_interior(map_full, labels_full, unassigned_as_border)
    null = np.empty(n_rot)
    for k in range(n_rot):
        null[k] = border_minus_interior(map_full, spins.rotate(labels_full, k), unassigned_as_border)[0]
    (mu, sd) = (float(null.mean()), float(null.std(ddof=1)))
    z = (raw - mu) / sd if sd > 0 else np.nan
    p_one = (1 + np.sum(null >= raw)) / (n_rot + 1)
    p_two = (1 + np.sum(np.abs(null - mu) >= abs(raw - mu))) / (n_rot + 1)
    res = dict(raw=raw, null_mean=mu, null_sd=sd, z=float(z), p_one_sided=float(p_one), p_two_sided=float(p_two), n_border=nb, n_interior=ni, n_rot=n_rot, null=null)
    if verbose:
        C.log(f'alignment: raw={raw:.5f} null={mu:.5f}+-{sd:.5f} z={z:.2f} p={p_one:.4f} (border {nb}, interior {ni}) {time.time() - t0:.1f}s')
    return res

def _rotated_border_interior_gpu(labels_full, spins, ks, unassigned_as_border, device):
    import torch
    ai = C.cortex_axis()
    e = torch.from_numpy(_edges_full()).to(device)
    lab = torch.from_numpy(np.asarray(labels_full).astype(np.int64)).to(device)
    cortex = torch.from_numpy(ai['cortex_mask_full']).to(device)
    idx = np.empty((len(ks), 2 * C.N_VERT), np.int64)
    for h in C.HEMIS:
        o = C.HEMI_OFFSET[h]
        idx[:, o:o + C.N_VERT] = spins.idx[h][ks].astype(np.int64) + o
    lr = lab[torch.from_numpy(idx).to(device)]
    lr[:, ~cortex] = 0
    (la, lb) = (lr[:, e[:, 0]], lr[:, e[:, 1]])
    diff = la != lb
    if not unassigned_as_border:
        diff &= (la > 0) & (lb > 0)
    del la, lb
    cnt = torch.zeros(lr.shape, dtype=torch.int32, device=device)
    d32 = diff.to(torch.int32)
    cnt.index_add_(1, e[:, 0], d32)
    cnt.index_add_(1, e[:, 1], d32)
    labelled = lr > 0
    border = (cnt > 0) & labelled
    interior = labelled & ~border & cortex[None, :]
    return (border.to(torch.float32), interior.to(torch.float32))

def alignment_many(maps_full, labels_full, spins=None, n_rot=1000, unassigned_as_border=True, device=None, batch=250, backend='cpu'):
    import torch
    if backend not in ('cpu', 'gpu'):
        raise ValueError(backend)
    spins = spins or SpinRotations(max(n_rot, 1), 0)
    n_rot = min(n_rot, spins.n_rot)
    device = device or C.torch_device()
    M = np.atleast_2d(np.asarray(maps_full, dtype=np.float32))
    F = np.isfinite(M)
    M0 = torch.from_numpy(np.where(F, M, 0.0)).to(device)
    Ft = torch.from_numpy(F.astype(np.float32)).to(device)

    def _means(masks):
        B = torch.from_numpy(masks.astype(np.float32)).to(device) if isinstance(masks, np.ndarray) else masks
        return M0 @ B.T / (Ft @ B.T).clamp(min=1)
    (b, i) = border_interior(labels_full, unassigned_as_border)
    raw = (_means(b[None]) - _means(i[None]))[:, 0].cpu().numpy()
    null = np.empty((M.shape[0], n_rot), np.float32)
    for a in range(0, n_rot, batch):
        ks = range(a, min(a + batch, n_rot))
        if backend == 'gpu':
            (Bt, It) = _rotated_border_interior_gpu(labels_full, spins, np.arange(ks.start, ks.stop), unassigned_as_border, device)
            null[:, a:a + len(ks)] = (_means(Bt) - _means(It)).cpu().numpy()
            del Bt, It
            continue
        (Bs, Is) = ([], [])
        for k in ks:
            (bk, ik) = border_interior(spins.rotate(labels_full, k), unassigned_as_border)
            Bs.append(bk)
            Is.append(ik)
        null[:, a:a + len(ks)] = (_means(np.stack(Bs)) - _means(np.stack(Is))).cpu().numpy()
    (mu, sd) = (null.mean(1), null.std(1, ddof=1))
    z = (raw - mu) / sd
    p_one = (1 + (null >= raw[:, None]).sum(1)) / (n_rot + 1)
    p_two = (1 + (np.abs(null - mu[:, None]) >= np.abs(raw - mu)[:, None]).sum(1)) / (n_rot + 1)
    return dict(raw=raw, null_mean=mu, null_sd=sd, z=z, p_one_sided=p_one, p_two_sided=p_two, n_border=(F & b).sum(1), n_interior=(F & i).sum(1), null=null, n_rot=n_rot)
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--map', default=C.GROUP_MAP)
    ap.add_argument('--row', type=int, default=0)
    ap.add_argument('--parcellation', nargs='+', default=['schaefer200'])
    ap.add_argument('--n-rot', type=int, default=1000)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    m = C.load_cifti_row(a.map, a.row)
    spins = SpinRotations(a.n_rot, 0)
    out = {}
    for name in a.parcellation:
        lab = C.load_parcellation(name)
        r = alignment(m, lab, spins, a.n_rot)
        out[name] = {k: v for (k, v) in r.items() if k != 'null'}
        print(f"{name}: border-interior = {r['raw']:.5f}, null {r['null_mean']:.5f} +- {r['null_sd']:.5f}, z = {r['z']:.2f}, p(one-sided) = {r['p_one_sided']:.4f}, n_border={r['n_border']}, n_interior={r['n_interior']}")
    if a.out:
        C.save_json(out, a.out)
