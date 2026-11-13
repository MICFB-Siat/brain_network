import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
from pfbn import settings
from pathlib import Path
import json, sys, time, csv, os, hashlib, warnings
warnings.filterwarnings('ignore', 'All-NaN slice')
import numpy as np, nibabel as nib, torch, gco
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.sparse.csgraph import connected_components
from scipy.optimize import nnls, least_squares
sys.path.insert(0, settings.code('pfbn', 'eigenmodes'))
import paths as P, common as C, alignment as A
from dcbc import SpinRotations
OUT = Path(settings.out('eigenmodes'))
OUT.mkdir(parents=True, exist_ok=True)
GROUP = sys.argv[1] if len(sys.argv) > 1 else 'young_adults'
NSUB = int(sys.argv[2]) if len(sys.argv) > 2 else 10
N_IT = 6
K_GRAD = 15.0
KAPPA = 10.0
CW = 200.0
K_MAX = int(os.environ.get('EIG_KMAX', '800'))
R_FIX = 30.0
R_GRID = np.arange(2.0, 80.01, 0.5)
GRID = os.environ.get('EIG_VARIANTS', 'raw:100,mode50:100,mode100:100,mode200:100,mode400:100,mode800:100,wave200:100,wave400:100,wavefix200:100,wavefix400:100,raw:30,mode200:30,wave200:30,wave400:30')
TAG = os.environ.get('EIG_TAG', '')
VARIANTS = [(p.split(':')[0], float(p.split(':')[1])) for p in GRID.split(',')]
dev = 'cuda'
spins = SpinRotations(1000, seed=0)
man = P.manifest()
TIMES = {}

def say(*a):
    s = f"[{time.strftime('%H:%M:%S')}] " + ' '.join((str(x) for x in a))
    print(s, flush=True)
    open(OUT / f'log_{GROUP}{TAG}.txt', 'a').write(s + '\n')
geo = {}
for (h, off) in (('L', 0), ('R', 32492)):
    g = nib.load(str(P.W / f'spheres/S1200.{h}.sphere.32k_fs_LR.surf.gii'))
    xyz = g.darrays[0].data.astype(np.float32)
    F = g.darrays[1].data
    e = np.unique(np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [0, 2]]]), 1), axis=0)
    (mid_xyz, mid_F) = C.load_surf(h, 'midthickness')
    geo[h] = dict(off=off, s=xyz / np.linalg.norm(xyz, axis=1, keepdims=True), edges=e, mid_xyz=mid_xyz, mid_F=mid_F)

def own_map(sid):
    img = nib.load(str(P.EXP / 'results/gradient' / sid / 'local_boundary_maps.dscalar.nii'))
    d = np.asarray(img.get_fdata(), np.float32)
    names = [m.map_name for m in img.header.get_index_map(0)]
    v = d[names.index('mean_smoothed_local_gradient')]
    full = np.full(64984, np.nan, np.float32)
    for (name, sl, bma) in img.header.get_axis(1).iter_structures():
        full[(0 if 'LEFT' in str(name) else 32492) + bma.vertex] = v[sl]
    return (full, img)

def bold_full(sid, run=0):
    srcs = json.load(open(P.SUBJECT_SOURCES))[sid]
    if run >= len(srcs):
        return None
    img = nib.load(srcs[run])
    X = np.asarray(img.get_fdata(), np.float32)
    full = np.full((X.shape[0], 64984), np.nan, np.float32)
    for (name, sl, bma) in img.header.get_axis(1).iter_structures():
        if 'CORTEX' in str(name):
            full[:, (0 if 'LEFT' in str(name) else 32492) + bma.vertex] = X[:, sl]
    return full

def valid_mask(sid, lab_full, gmap, X):
    out = {}
    for (h, G) in geo.items():
        sl = slice(G['off'], G['off'] + 32492)
        out[h] = (lab_full[sl] > 0) & ~np.isnan(X[0, sl]) & ~np.isnan(gmap[sl])
    return out

def cot_laplacian(xyz, F):
    (i, j, k) = (F[:, 0], F[:, 1], F[:, 2])

    def cot(a, b, c):
        u = xyz[b] - xyz[a]
        v = xyz[c] - xyz[a]
        return (u * v).sum(1) / np.linalg.norm(np.cross(u, v), axis=1)
    (ci, cj, ck) = (cot(i, j, k), cot(j, k, i), cot(k, i, j))
    rows = np.concatenate([j, k, i])
    cols = np.concatenate([k, i, j])
    w = 0.5 * np.concatenate([ci, cj, ck])
    n = len(xyz)
    W = sp.coo_matrix((w, (rows, cols)), shape=(n, n))
    W = (W + W.T).tocsr()
    L = sp.diags(np.asarray(W.sum(1)).ravel()) - W
    area = 0.5 * np.linalg.norm(np.cross(xyz[j] - xyz[i], xyz[k] - xyz[i]), axis=1)
    m = np.zeros(n)
    np.add.at(m, i, area / 3)
    np.add.at(m, j, area / 3)
    np.add.at(m, k, area / 3)
    return (L.tocsc(), m)

def eigenmodes(h, valid, k_max):
    f = OUT / f'modes_{h}.npz'
    if f.exists():
        d = np.load(f)
        say(f'{h}: loaded cached modes', d['psi'].shape)
        return dict(idx=d['idx'], evals=d['evals'], psi=d['psi'], mass=d['mass'])
    t0 = time.time()
    G = geo[h]
    F = G['mid_F']
    F = F[valid[F].all(1)]
    used = np.zeros(32492, bool)
    used[F.ravel()] = True
    pos = -np.ones(32492, int)
    idx = np.where(used)[0]
    pos[idx] = np.arange(len(idx))
    Fl = pos[F]
    xyz = G['mid_xyz'][idx]
    (L, m) = cot_laplacian(xyz, Fl)
    adj = sp.coo_matrix((np.ones(len(Fl) * 3), (np.concatenate([Fl[:, 0], Fl[:, 1], Fl[:, 2]]), np.concatenate([Fl[:, 1], Fl[:, 2], Fl[:, 0]]))), shape=(len(idx),) * 2)
    (ncc, cc) = connected_components(adj, directed=False)
    big = np.bincount(cc).argmax()
    keep = cc == big
    say(f'{h}: valid {valid.sum()} vertices, {len(idx)} in faces, {ncc} components, keeping largest {keep.sum()} (dropped {int((~keep).sum())}; {int(valid.sum() - len(idx))} face-less)')
    if not keep.all():
        L = L[keep][:, keep].tocsc()
        m = m[keep]
        idx = idx[keep]
    M = sp.diags(m).tocsc()
    say(f'{h}: eigsh k={k_max} on {L.shape[0]} vertices ...')
    (evals, psi) = eigsh(L, k=k_max, M=M, sigma=-1e-06, which='LM')
    o = np.argsort(evals)
    (evals, psi) = (evals[o], psi[:, o])
    evals[evals < 0] = 0.0
    psi = psi / np.sqrt((psi * m[:, None] * psi).sum(0, keepdims=True))
    orth = np.abs(psi.T @ (m[:, None] * psi) - np.eye(k_max)).max()
    dt = time.time() - t0
    TIMES[f'eigenmodes_{h}_sec'] = dt
    say(f'{h}: eigsh done in {dt:.0f}s; max |Psi^T M Psi - I| = {orth:.2e}; total area {m.sum():.0f} mm^2')
    say(f'{h}: wavelength 2*pi/sqrt(lambda_K) (mm): ' + ', '.join((f'K={K}: {2 * np.pi / np.sqrt(evals[K - 1]):.1f}' for K in (50, 100, 200, 400, 800) if K <= k_max)))
    np.savez(f, idx=idx, evals=evals, psi=psi.astype(np.float32), mass=m.astype(np.float32), n_valid=int(valid.sum()))
    json.dump(dict(hemi=h, n_vertices=int(L.shape[0]), k=k_max, wall_sec=dt, orth_err=float(orth), wavelength_mm={K: float(2 * np.pi / np.sqrt(evals[K - 1])) for K in (50, 100, 200, 400, 800) if K <= k_max}, evals_head=evals[:5].tolist(), surface=C.surf_path(h, 'midthickness')), open(OUT / f'eigenmodes_{h}_info.json', 'w'), indent=1)
    return dict(idx=idx, evals=evals, psi=psi.astype(np.float32), mass=m.astype(np.float32))

def fit_wave(p, lam, r_grid):
    lp = np.log(p)
    best = None
    for r in r_grid:
        f = 1.0 / (1.0 + r * r * lam) ** 2
        (s0, n0) = nnls(np.stack([f, np.ones_like(f)], 1), p)[0]
        x0 = np.log(np.maximum([s0, n0], 1e-12 * p.max()))
        res = least_squares(lambda x: np.log(np.exp(np.minimum(x[0], 300)) * f + np.exp(np.minimum(x[1], 300))) - lp, x0, method='lm')
        sse = float((res.fun ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, r, np.exp(res.x[0]), np.exp(res.x[1]))
    (sse, r, s, n) = best
    r2 = 1 - sse / float(((lp - lp.mean()) ** 2).sum())
    return dict(r_s=float(r), s=float(s), n=float(n), r2_log=float(r2))

def solve(Xn, S, lab0, edges, w_edge, kappa, tau, n_it):
    lab = lab0.copy()
    L = lab0.max() + 1
    V = len(lab)
    Xt = Xn.T
    for it in range(n_it):
        M = torch.zeros(V, L, device=dev)
        M[torch.arange(V), torch.from_numpy(lab).to(dev)] = 1.0
        mu = Xn @ M
        mu = mu / (mu.norm(dim=0, keepdim=True) + 1e-06)
        cc = S.T @ M
        cc = cc / (cc.norm(dim=0, keepdim=True) + 1e-06)
        U = -(kappa * (Xt @ mu) + tau * (S @ cc))
        U = U - U.min()
        pairwise = 1 - np.eye(L)
        new = gco.cut_general_graph(edges, w_edge.astype(np.float64), U.cpu().numpy().astype(np.float64), pairwise, n_iter=-1, algorithm='swap')
        new = np.asarray(new).astype(np.int64)
        changed = (new != lab).mean()
        lab = new
        if changed < 0.002:
            break
    return lab

def n_components(lab, edges):
    tot = 0
    for l in np.unique(lab):
        idx = np.where(lab == l)[0]
        pos = -np.ones(len(lab), int)
        pos[idx] = np.arange(len(idx))
        m = (lab[edges[:, 0]] == l) & (lab[edges[:, 1]] == l)
        e = edges[m]
        (n, _) = connected_components(sp.coo_matrix((np.ones(len(e)), (pos[e[:, 0]], pos[e[:, 1]])), shape=(len(idx), len(idx))), directed=False)
        tot += n
    return tot

def components_full(lab):
    comp = 0
    for G in geo.values():
        sl = slice(G['off'], G['off'] + 32492)
        l = lab[sl]
        valid = l > 0
        idx = np.where(valid)[0]
        pos = -np.ones(32492, int)
        pos[idx] = np.arange(len(idx))
        e = G['edges']
        e = e[valid[e[:, 0]] & valid[e[:, 1]]]
        comp += n_components(l[idx], np.stack([pos[e[:, 0]], pos[e[:, 1]]], 1))
    return comp

def homogeneity(Z, lab):
    T = Z.shape[0]
    (u, inv) = np.unique(lab, return_inverse=True)
    keep = u > 0
    Mo = torch.zeros(len(lab), len(u), device=dev)
    Mo[torch.arange(len(lab)), torch.from_numpy(inv).to(dev)] = 1.0
    S = Z @ Mo
    n = Mo.sum(0)
    ss = (S * S).sum(0)
    h = (ss - n * T) / T / (n * (n - 1)).clamp(min=1)
    ok = torch.from_numpy(keep).to(dev) & (n >= 2)
    return float((h[ok] * n[ok]).sum() / n[ok].sum())
rows = []
fit_rows = []
sids = [s for s in sorted(man) if man[s]['group'] == GROUP and P.ours_dlabel(s, GROUP) is not None and (P.EXP / 'results/gradient' / s / 'local_boundary_maps.dscalar.nii').exists()][:NSUB]
say(GROUP, 'subjects', sids, 'K_MAX', K_MAX, 'variants', GRID)
schaefer = C.cortex_to_full(np.load(P.ASSETS / 'Micapipe200_labels.npz')['labels'], 0).astype(int)
modes = None
eig_mask = {}
for (si, sid) in enumerate(sids):
    t0 = time.time()
    lab_full = np.asarray(C.load_parcellation(str(P.ours_dlabel(sid, GROUP)))).astype(int)
    (gmap, _) = own_map(sid)
    ref_group = np.load(P.CACHE / 'native_group_map' / f'{sid}.npy')
    X = bold_full(sid, 0)
    t_load1 = time.time() - t0
    vm = valid_mask(sid, lab_full, gmap, X)
    if modes is None:
        modes = {h: eigenmodes(h, vm[h], K_MAX) for h in geo}
        for h in geo:
            eig_mask[h] = np.zeros(32492, bool)
            eig_mask[h][modes[h]['idx']] = True
    for h in geo:
        d = int((vm[h] & ~eig_mask[h]).sum())
        d2 = int((eig_mask[h] & ~vm[h]).sum())
        if d or d2:
            say(f'{sid} {h}: valid mask differs from eigen mask: {d} valid-not-in-modes, {d2} modes-not-valid (intersection used)')
    t1 = time.time()
    rep = {}
    info = {}
    for (h, G) in geo.items():
        sl = slice(G['off'], G['off'] + 32492)
        midx = modes[h]['idx']
        Xm = X[:, sl][:, midx].copy()
        Xm[np.isnan(Xm)] = 0.0
        Xm = torch.from_numpy(Xm).to(dev)
        Xm = Xm - Xm.mean(0, keepdim=True)
        Psi = torch.from_numpy(modes[h]['psi']).to(dev)
        m = torch.from_numpy(modes[h]['mass']).to(dev)
        lam = modes[h]['evals']
        Acoef = Psi.T @ (m[:, None] * Xm.T)
        p = (Acoef ** 2).mean(1).cpu().numpy().astype(np.float64)
        tot = float((Xm ** 2).sum())
        fw = fit_wave(p, lam, R_GRID)
        ff = fit_wave(p, lam, np.array([R_FIX]))
        fit_rows.append(dict(subject=sid, hemi=h, r_s_mm=fw['r_s'], s=fw['s'], n=fw['n'], r2_log=fw['r2_log'], s_fix30=ff['s'], n_fix30=ff['n'], r2_log_fix30=ff['r2_log'], p0=p[0], p_med=float(np.median(p)), frac_var_Kmax=float(p.sum() * Xm.shape[0] / tot)))
        rep[h] = {'raw': (Xm, 1.0, float('nan'))}
        for (name, _) in VARIANTS:
            if name == 'raw' or name in rep[h]:
                continue
            K = int(''.join((ch for ch in name if ch.isdigit())))
            kind = name.rstrip('0123456789')
            if kind == 'mode':
                w = np.ones(K)
            else:
                ft = fw if kind == 'wave' else ff
                Sj = ft['s'] / (1 + ft['r_s'] ** 2 * lam[:K]) ** 2
                w = Sj / (Sj + ft['n'])
            Ak = Acoef[:K] * torch.from_numpy(w.astype(np.float32)).to(dev)[:, None]
            Xr = (Psi[:, :K] @ Ak).T
            rep[h][name] = (Xr, float((Xr ** 2).sum()) / tot, float(w.sum()))
        info[h] = dict(fit=fw, fit_fix=ff)
    t_proj = time.time() - t1
    TIMES.setdefault('proj_fit_sec', []).append(t_proj)
    say(sid, f"load {t_load1:.0f}s, proj+fit {t_proj:.0f}s; r_s L {info['L']['fit']['r_s']:.1f} R {info['R']['fit']['r_s']:.1f} mm (R2 {info['L']['fit']['r2_log']:.2f}/{info['R']['fit']['r2_log']:.2f})")
    results = {'Ours_init': (lab_full.copy(), 0.0, {}, {})}
    for (name, tau) in VARIANTS:
        vname = f'{name}_t{tau:g}'
        out = np.zeros(64984, int)
        tm = time.time()
        fv = {}
        em = {}
        for (h, G) in geo.items():
            sl = slice(G['off'], G['off'] + 32492)
            l0 = lab_full[sl]
            midx = modes[h]['idx']
            valid = vm[h] & eig_mask[h]
            idx = np.where(valid)[0]
            pos = -np.ones(32492, int)
            pos[idx] = np.arange(len(idx))
            e = G['edges']
            e = e[valid[e[:, 0]] & valid[e[:, 1]]]
            e2 = np.stack([pos[e[:, 0]], pos[e[:, 1]]], 1).astype(np.int32)
            g = gmap[sl]
            g_edge = (g[e[:, 0]] + g[e[:, 1]]) / 2
            g_edge = np.clip(g_edge / np.nanpercentile(g[valid], 99), 0, 1)
            w_edge = CW * np.exp(-K_GRAD * g_edge)
            (Xr, fv[h], em[h]) = rep[h][name]
            mpos = -np.ones(32492, int)
            mpos[midx] = np.arange(len(midx))
            Xh = Xr[:, mpos[idx]]
            Xh = Xh - Xh.mean(0, keepdim=True)
            Xh = Xh / (Xh.norm(dim=0, keepdim=True) + 1e-06)
            S = torch.from_numpy(G['s'][idx]).to(dev)
            (uniq, l0c) = np.unique(l0[idx], return_inverse=True)
            lab = solve(Xh, S, l0c, e2, w_edge, KAPPA, tau, N_IT)
            out[G['off'] + idx] = uniq[lab]
        results[vname] = (out, time.time() - tm, fv, em)
    del rep
    torch.cuda.empty_cache()
    t2 = time.time()
    X2 = bold_full(sid, 1)
    Z2 = None
    if X2 is not None:
        X2 = torch.from_numpy(X2).to(dev)
        X2 = X2 - X2.mean(0, keepdim=True)
        sd = X2.std(0, keepdim=True)
        Z2 = X2 / sd
        okv = torch.isfinite(Z2).all(0) & (sd[0] > 0)
        Z2[:, ~okv] = 0.0
        okv = okv.cpu().numpy()
    t_load2 = time.time() - t2

    def homog(lab):
        if Z2 is None:
            return float('nan')
        l = lab.copy()
        l[~okv] = 0
        return homogeneity(Z2, l)
    results['Schaefer200_micapipe'] = (schaefer.copy(), 0.0, {}, {})
    t3 = time.time()
    for (vname, (lab, tm, fv, em)) in results.items():
        te = time.time()
        r = A.alignment_many(np.stack([gmap, ref_group]), lab, spins, n_rot=1000, unassigned_as_border=False, backend='gpu')
        comp = components_full(lab)
        hom = homog(lab)
        t_eval = time.time() - te
        rows.append(dict(subject=sid, group=GROUP, variant=vname, n_parcels=int(len(np.unique(lab[lab > 0]))), n_components=comp, z_own=float(r['z'][0]), raw_own=float(r['raw'][0]), z_group=float(r['z'][1]), raw_group=float(r['raw'][1]), homogeneity_run2=hom, frac_var_L=fv.get('L', np.nan), frac_var_R=fv.get('R', np.nan), eff_modes_L=em.get('L', np.nan), eff_modes_R=em.get('R', np.nan), mrf_sec=round(tm, 1), eval_sec=round(t_eval, 1)))
        if vname not in ('Schaefer200_micapipe',):
            np.save(OUT / f'labels_{sid}_{vname}.npy', lab.astype(np.int16))
    del X2, Z2
    torch.cuda.empty_cache()
    TIMES.setdefault('load_run1_sec', []).append(t_load1)
    TIMES.setdefault('load_run2_sec', []).append(t_load2)
    TIMES.setdefault('eval_all_variants_sec', []).append(time.time() - t3)
    say(sid, f'{time.time() - t0:.0f}s total (load2 {t_load2:.0f}s, eval {time.time() - t3:.0f}s)', ' | '.join((f"{r['variant']}: own {r['z_own']:.1f} grp {r['z_group']:.1f} C{r['n_components']} H{r['homogeneity_run2']:.3f} mrf{r['mrf_sec']:.0f}s" for r in rows if r['subject'] == sid)))
    with open(OUT / f'eigen_ablation_{GROUP}{TAG}.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(OUT / f'modes_fit{TAG}.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(fit_rows[0]))
        w.writeheader()
        w.writerows(fit_rows)
    json.dump(TIMES, open(OUT / f'timings{TAG}.json', 'w'), indent=1)
say('SUMMARY (median over subjects)')
say(f"{'variant':24s} {'z_own':>6s} {'z_grp':>6s} {'parc':>5s} {'comp':>5s} {'homog2':>7s} {'fvarL':>6s} {'effL':>6s} {'mrf_s':>6s}")
for name in ['Ours_init', 'Schaefer200_micapipe'] + [f'{n}_t{t:g}' for (n, t) in VARIANTS]:
    v = [r for r in rows if r['variant'] == name]
    md = lambda k: np.nanmedian([r[k] for r in v])
    say(f"{name:24s} {md('z_own'):6.2f} {md('z_group'):6.2f} {md('n_parcels'):5.0f} {md('n_components'):5.0f} {md('homogeneity_run2'):7.4f} {md('frac_var_L'):6.3f} {md('eff_modes_L'):6.1f} {md('mrf_sec'):6.1f}")
say('TIMINGS (sec): ' + json.dumps({k: round(float(np.median(v)), 1) if isinstance(v, list) else round(v, 1) for (k, v) in TIMES.items()}))
say('r_s (mm) per subject/hemi: ' + ', '.join((f"{r['subject']}{r['hemi']} {r['r_s_mm']:.1f}" for r in fit_rows)))
say('DONE')
