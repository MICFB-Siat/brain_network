import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from common import log
import numpy as np, torch, gco
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
dev = 'cuda'
(MAX_IT, STOP, NO_INJECT_LAST, BLOCK, L) = (60, 0.002, 10, 40, 100)
METHODS = {'v4_filt_t03_c1': dict(X='filt', prior='sphere', kappa=1.0, tau=0.3, c_mult=1, init='v3', gmap='own_session', r_s=None, frozen_from='minimal_v2_ds005498_20261024/v4/run_mrf_v4.py VARIANTS[v4_filt_t03_c1]'), 'v4_stim_t03_c1': dict(X='raw', prior='sphere', kappa=1.0, tau=0.3, c_mult=1, init='v3', gmap='own', r_s=None, frozen_from='minimal_v2_ds005498_20261024/v4/run_mrf_v4.py VARIANTS[v4_stim_t03_c1]'), 'v5_stim_t03_r30': dict(X='raw', prior='wave', kappa=1.0, tau=0.3, c_mult=1, init='v3', gmap='own_session', r_s=30.0, frozen_from='minimal_v2_ds005498_20261024/v5/run_mrf_v5.py VARIANTS[v5_stim_t03_r30]')}

def edge_base(hemi):
    e = C.hemi_edges_cortex(hemi)
    cidx = C.cortex_idx(hemi)
    g = np.nan_to_num(np.load(os.path.join(C.RESULTS, f'gmap_{hemi}.npy')).astype(np.float64)[cidx], nan=0.0)
    p99 = np.nanpercentile(g, 99)
    g_edge = np.clip((g[e[:, 0]] + g[e[:, 1]]) / 2 / p99, 0, 1)
    base = np.exp(-C.K_GRAD * g_edge)
    return (e.astype(np.int32), base, {'p99': float(p99), 'n_edges': int(len(e)), 'edge_base_median': float(np.median(base)), 'edge_base_mean': float(base.mean()), 'gmap': f'gmap_{hemi}.npy'})

def reinject(lab, S_np, it):
    inj = []
    for l in np.where(np.bincount(lab, minlength=L) == 0)[0]:
        sizes = np.bincount(lab, minlength=L)
        big = int(np.argmax(sizes))
        idx = np.where(lab == big)[0]
        cen = S_np[idx].mean(0)
        cen /= np.linalg.norm(cen)
        far = idx[np.argmin(S_np[idx] @ cen)]
        near = idx[np.argsort(-(S_np[idx] @ S_np[far]))[:BLOCK + 1]]
        lab[near] = l
        inj.append({'it': it, 'label': int(l), 'from_parcel': big, 'from_size': int(sizes[big]), 'n_vertices': int(len(near)), 'far_vertex': int(far)})
    return (lab, inj)

def load_modes(hemi, r_s):
    z = np.load(os.path.join(C.MODES, f'modes_{C.HEMI_LETTER[hemi]}.npz'))
    idx = z['idx']
    assert np.array_equal(idx, C.cortex_idx(hemi)), 'modes idx != cortex vertices'
    lam = z['evals'].astype(np.float64)
    g = 1.0 / (1.0 + r_s ** 2 * lam) ** 2
    Psi = torch.from_numpy(z['psi'].astype(np.float32)).to(dev)
    mass = torch.from_numpy(z['mass'].astype(np.float32)).to(dev)
    gt = torch.from_numpy(g.astype(np.float32)).to(dev)
    PsiG = Psi * gt[None, :]
    MPsi = Psi * mass[:, None]
    info = {'r_s_mm': r_s, 'n_modes': int(Psi.shape[1]), 'g_head': g[:5].tolist(), 'g_tail': g[-3:].tolist(), 'sum_g': float(g.sum()), 'lambda_max': float(lam.max()), 'wavelength_mm_at_g_half': float(2 * np.pi / np.sqrt(lam[np.argmin(np.abs(g - 0.5))])) if (g < 0.5).any() else None, 'modes_file': os.path.join(C.MODES, f'modes_{C.HEMI_LETTER[hemi]}.npz')}
    del Psi
    return (PsiG, MPsi, mass, info)

def wave_term(M, PsiG, MPsi, mass):
    Pbar = MPsi.T @ M
    ml = mass @ M
    Pbar = Pbar / (ml[None, :] + 1e-12)
    return PsiG @ Pbar

def solve_v4(Xn, S, lab0, edges, ebase, kappa, tau, c_mult, n_it=MAX_IT, stop=STOP, fixed=None, hist=None, inject_log=None):
    lab = lab0.copy()
    V = len(lab)
    Xt = Xn.T
    S_np = S.cpu().numpy().astype(np.float64)
    scal = None
    w = None
    for it in range(n_it):
        t0 = time.time()
        M = torch.zeros(V, L, device=dev)
        M[torch.arange(V), torch.from_numpy(lab).to(dev)] = 1.0
        mu = Xn @ M
        mu = mu / (mu.norm(dim=0, keepdim=True) + 1e-06)
        cc = S.T @ M
        cc = cc / (cc.norm(dim=0, keepdim=True) + 1e-06)
        A = Xt @ mu
        B = S @ cc
        if scal is None:
            (sA0, sB0) = (float(A.std()), float(B.std()))
            sig_v = (A / sA0).std(dim=1)
            (sigma_med, sigma_mean) = (float(sig_v.median()), float(sig_v.mean()))
            c_base0 = 0.5 * sigma_med / float(np.median(ebase))
            if fixed is not None:
                (sA, sB, c_base) = (float(fixed['sA']), float(fixed['sB']), float(fixed['c_base']))
            else:
                (sA, sB, c_base) = (sA0, sB0, c_base0)
            w = (c_base * c_mult * ebase).astype(np.float64)
            scal = {'sA': sA, 'sB': sB, 'c_base': c_base, 'c_mult': c_mult, 'c': c_base * c_mult, 'fixed_from_base_run': fixed is not None, 'it0_own_sA': sA0, 'it0_own_sB': sB0, 'it0_own_c_base': c_base0, 'it0_sigma_vertex_median': sigma_med, 'it0_sigma_vertex_mean': sigma_mean, 'edge_base_median': float(np.median(ebase)), 'w_median': float(np.median(w)), 'w_mean': float(w.mean()), 'w_max': float(w.max()), 'it0_A_mean': float(A.mean()), 'it0_B_mean': float(B.mean()), 'it0_A_range': [float(A.min()), float(A.max())], 'it0_B_range': [float(B.min()), float(B.max())]}
        U = -(kappa * A / sA + tau * B / sB)
        U = U - U.min()
        new = gco.cut_general_graph(edges, w, U.cpu().numpy().astype(np.float64), 1 - np.eye(L), n_iter=-1, algorithm='swap')
        new = np.asarray(new).astype(np.int64)
        changed = float((new != lab).mean())
        n_alive_cut = int(len(np.unique(new)))
        inj = []
        if it + 1 <= n_it - NO_INJECT_LAST:
            (new, inj) = reinject(new, S_np, it + 1)
        lab = new
        if inject_log is not None:
            inject_log.extend(inj)
        if hist is not None:
            hist.append({'it': it + 1, 'frac_changed': changed, 'n_labels_after_cut': n_alive_cut, 'n_labels': int(len(np.unique(lab))), 'reinjected': len(inj), 'seconds': round(time.time() - t0, 2)})
        if changed < stop and len(inj) == 0:
            break
    return (lab, scal)

def solve_v5(Xn, S, PsiG, MPsi, mass, lab0, edges, ebase, kappa, tau, c_mult, n_it=MAX_IT, stop=STOP, fixed=None, hist=None, inject_log=None):
    lab = lab0.copy()
    V = len(lab)
    Xt = Xn.T
    S_np = S.cpu().numpy().astype(np.float64)
    scal = None
    w = None
    for it in range(n_it):
        t0 = time.time()
        M = torch.zeros(V, L, device=dev)
        M[torch.arange(V), torch.from_numpy(lab).to(dev)] = 1.0
        mu = Xn @ M
        mu = mu / (mu.norm(dim=0, keepdim=True) + 1e-06)
        A = Xt @ mu
        B = wave_term(M, PsiG, MPsi, mass)
        if scal is None:
            (sA0, sB0) = (float(A.std()), float(B.std()))
            sig_v = (A / sA0).std(dim=1)
            (sigma_med, sigma_mean) = (float(sig_v.median()), float(sig_v.mean()))
            c_base0 = 0.5 * sigma_med / float(np.median(ebase))
            if fixed is not None:
                (sA, sB, c_base) = (float(fixed['sA']), float(fixed['sB']), float(fixed['c_base']))
            else:
                (sA, sB, c_base) = (sA0, sB0, c_base0)
            w = (c_base * c_mult * ebase).astype(np.float64)
            Bs = B / sB0
            sigB_v = Bs.std(dim=1)
            top2 = torch.topk(Bs, 2, dim=1).values
            gapB = top2[:, 0] - top2[:, 1]
            As = A / sA0
            top2a = torch.topk(As, 2, dim=1).values
            gapA = top2a[:, 0] - top2a[:, 1]
            scal = {'sA': sA, 'sB': sB, 'c_base': c_base, 'c_mult': c_mult, 'c': c_base * c_mult, 'fixed_from_base_run': fixed is not None, 'it0_own_sA': sA0, 'it0_own_sB': sB0, 'it0_own_c_base': c_base0, 'it0_sigma_vertex_median': sigma_med, 'it0_sigma_vertex_mean': sigma_mean, 'it0_sigma_vertex_B_median': float(sigB_v.median()), 'it0_sigma_vertex_B_mean': float(sigB_v.mean()), 'it0_B_top1_top2_gap_median': float(gapB.median()), 'it0_A_top1_top2_gap_median': float(gapA.median()), 'it0_B_frac_entries_below_1pct_of_max': float((B < 0.01 * B.max()).float().mean()), 'edge_base_median': float(np.median(ebase)), 'w_median': float(np.median(w)), 'w_mean': float(w.mean()), 'w_max': float(w.max()), 'it0_A_mean': float(A.mean()), 'it0_B_mean': float(B.mean()), 'it0_A_range': [float(A.min()), float(A.max())], 'it0_B_range': [float(B.min()), float(B.max())], 'nan_in_B_it0': bool(torch.isnan(B).any())}
        U = -(kappa * A / sA + tau * B / sB)
        U = U - U.min()
        new = gco.cut_general_graph(edges, w, U.cpu().numpy().astype(np.float64), 1 - np.eye(L), n_iter=-1, algorithm='swap')
        new = np.asarray(new).astype(np.int64)
        changed = float((new != lab).mean())
        n_alive_cut = int(len(np.unique(new)))
        inj = []
        if it + 1 <= n_it - NO_INJECT_LAST:
            (new, inj) = reinject(new, S_np, it + 1)
        lab = new
        if inject_log is not None:
            inject_log.extend(inj)
        if hist is not None:
            hist.append({'it': it + 1, 'frac_changed': changed, 'n_labels_after_cut': n_alive_cut, 'n_labels': int(len(np.unique(lab))), 'reinjected': len(inj), 'seconds': round(time.time() - t0, 2)})
        if changed < stop and len(inj) == 0:
            break
    return (lab, scal)

def n_components(lab):
    off = 0
    tot = 0
    for hemi in C.HEMIS:
        n = len(C.cortex_idx(hemi))
        e = C.hemi_edges_cortex(hemi)
        Lh = lab[off:off + n]
        same = Lh[e[:, 0]] == Lh[e[:, 1]]
        A = coo_matrix((np.ones(same.sum()), (e[same, 0], e[same, 1])), shape=(n, n))
        tot += connected_components(A, directed=False)[0]
        off += n
    return tot

def clean(lab):
    out = lab.copy()
    report = []
    off = 0
    for hemi in C.HEMIS:
        n = len(C.cortex_idx(hemi))
        e = C.hemi_edges_cortex(hemi)
        Lh = lab[off:off + n]
        same = Lh[e[:, 0]] == Lh[e[:, 1]]
        A = coo_matrix((np.ones(same.sum()), (e[same, 0], e[same, 1])), shape=(n, n))
        (nc, comp) = connected_components(A, directed=False)
        for lb in np.unique(Lh):
            (cs, cnt) = np.unique(comp[Lh == lb], return_counts=True)
            if len(cs) > 1:
                keep = cs[np.argmax(cnt)]
                for (c, k) in zip(cs, cnt):
                    if c == keep:
                        continue
                    idx = np.where(comp == c)[0]
                    m = np.isin(e[:, 0], idx) ^ np.isin(e[:, 1], idx)
                    nb = np.concatenate([Lh[e[m, 0]], Lh[e[m, 1]]])
                    nb = nb[nb != lb]
                    (vals, cn) = np.unique(nb, return_counts=True)
                    new = vals[np.argmax(cn)]
                    out[off + idx] = new
                    report.append(dict(hemi=hemi, label=int(lb), island_size=int(k), main_size=int(cnt.max()), reassigned_to=int(new)))
        off += n
    return (out, report)

def run_method(name):
    cfg = METHODS[name]
    tag = name
    out_cortex = np.zeros(C.NCORT, np.int64)
    rep = {'method': name, 'config': cfg, 'max_it': MAX_IT, 'stop': STOP, 'no_inject_last': NO_INJECT_LAST, 'block': BLOCK}
    t_all = time.time()
    for hemi in C.HEMIS:
        t0 = time.time()
        cidx = C.cortex_idx(hemi)
        xp = os.path.join(C.STACKS, f"X_{cfg['X']}_{hemi}.npy")
        X = np.load(xp, mmap_mode='r')
        Xn = C.unit_cols(torch.from_numpy(np.asarray(X)).to(dev), torch)
        (xyz, _) = C.sphere(hemi)
        S = torch.from_numpy((xyz[cidx] / np.linalg.norm(xyz[cidx], axis=1, keepdims=True)).astype(np.float32)).to(dev)
        lab0 = np.load(os.path.join(C.RESULTS, f'init_{hemi}.npy')).astype(np.int64)
        iinfo = {'n_labels_init': int(len(np.unique(lab0))), 'file': f'init_{hemi}.npy'}
        (e, ebase, einfo) = edge_base(hemi)
        minfo = None
        if cfg['prior'] == 'wave':
            (PsiG, MPsi, mass, minfo) = load_modes(hemi, float(cfg['r_s']))
        t_load = time.time() - t0
        (hist, inj) = ([], [])
        t1 = time.time()
        if cfg['prior'] == 'wave':
            (lab, scal) = solve_v5(Xn, S, PsiG, MPsi, mass, lab0, e, ebase, cfg['kappa'], cfg['tau'], cfg['c_mult'], hist=hist, inject_log=inj)
            del PsiG, MPsi, mass
        else:
            (lab, scal) = solve_v4(Xn, S, lab0, e, ebase, cfg['kappa'], cfg['tau'], cfg['c_mult'], hist=hist, inject_log=inj)
        t_mrf = time.time() - t1
        off = 0 if hemi == 'lh' else C.NL
        out_cortex[off:off + C.HEMI_N[hemi]] = lab + 1 + (0 if hemi == 'lh' else 100)
        dead = sorted(set(range(L)) - set(np.unique(lab).tolist()))
        rep[hemi] = {'X_file': xp, 'T_rows': int(Xn.shape[0]), 'load_seconds': round(t_load, 1), 'mrf_seconds': round(t_mrf, 1), 'iterations': len(hist), 'converged': bool(hist[-1]['frac_changed'] < STOP and hist[-1]['reinjected'] == 0), 'final_frac_changed': hist[-1]['frac_changed'], 'n_labels_final': int(len(np.unique(lab))), 'dead_labels_final': dead, 'n_injections': len(inj), 'init': iinfo, 'edge': einfo, 'modes': minfo, 'scalars': scal, 'history': hist, 'injections': inj}
        log(f"{tag} {hemi}: {len(hist)} it, {t_mrf:.0f}s MRF (+{t_load:.0f}s load), labels {rep[hemi]['n_labels_final']}, last change {hist[-1]['frac_changed']:.4f}, injections {len(inj)}, converged {rep[hemi]['converged']}, sA {scal['sA']:.4f} sB {scal['sB']:.3g} c {scal['c']:.4f}")
        del Xn, S
        torch.cuda.empty_cache()
    rep['total_seconds'] = round(time.time() - t_all, 1)
    rep['mrf_seconds'] = round(rep['lh']['mrf_seconds'] + rep['rh']['mrf_seconds'], 1)
    np.save(os.path.join(C.RESULTS, f'labels_{tag}.npy'), out_cortex.astype(np.int16))
    json.dump(rep, open(os.path.join(C.RESULTS, f'mrf_{tag}.json'), 'w'), indent=1)
    C.Timings.put('mrf', tag, rep['mrf_seconds'], {'total_seconds': rep['total_seconds'], 'iterations_lh': rep['lh']['iterations'], 'iterations_rh': rep['rh']['iterations'], 'injections_lh': rep['lh']['n_injections'], 'injections_rh': rep['rh']['n_injections'], 'converged_lh': rep['lh']['converged'], 'converged_rh': rep['rh']['converged']})
    t0 = time.time()
    C.cortex_labels_to_dlabel(out_cortex, os.path.join(C.RESULTS, f'{tag}.dlabel.nii'))
    C.Timings.put('dlabel', tag, time.time() - t0)
    return (out_cortex, rep)

def islands(tag):
    p_out = os.path.join(C.RESULTS, 'islands.json')
    rep = json.load(open(p_out)) if os.path.exists(p_out) else {}
    t0 = time.time()
    lab = np.load(os.path.join(C.RESULTS, f'labels_{tag}.npy')).astype(np.int64)
    (out, isl) = clean(lab)
    r = {'islands': isl, 'n_islands': len(isl), 'changed_vertices': int((out != lab).sum()), 'components_before': n_components(lab), 'components_after': n_components(out), 'labels_before': int(len(np.unique(lab))), 'labels_after': int(len(np.unique(out))), 'island_sizes': sorted((int(i['island_size']) for i in isl)), 'seconds': time.time() - t0}
    rep[tag] = r
    np.save(os.path.join(C.RESULTS, f'labels_{tag}_clean.npy'), out.astype(np.int16))
    json.dump(rep, open(p_out, 'w'), indent=1)
    C.cortex_labels_to_dlabel(out, os.path.join(C.RESULTS, f'{tag}_clean.dlabel.nii'))
    C.Timings.put('islands', tag, r['seconds'], {'n_islands': len(isl), 'changed_vertices': r['changed_vertices']})
    log(f"islands {tag}: {len(isl)} islands ({r['changed_vertices']} vertices), components {r['components_before']} -> {r['components_after']}, labels {r['labels_before']} -> {r['labels_after']}")
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', default='v5_stim_t03_r30')
    a = ap.parse_args()
    for m in [v for v in a.method.split(',') if v]:
        if os.path.exists(os.path.join(C.RESULTS, f'labels_{m}.npy')):
            log(f'{m}: labels exist, skipping')
        else:
            run_method(m)
        if not os.path.exists(os.path.join(C.RESULTS, f'labels_{m}_clean.npy')):
            islands(m)
    log('MRF_END')
