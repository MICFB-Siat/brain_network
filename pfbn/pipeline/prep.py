import os, sys, json, time, argparse, subprocess, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from common import log
import numpy as np, nibabel as nib, torch
from scipy.optimize import nnls, least_squares
dev = 'cuda'
os.makedirs(C.TMPFS, exist_ok=True)

def stage_reduce():
    outs = [os.path.join(C.STACKS, f'X_raw_{h}.npy') for h in C.HEMIS]
    pmeta = os.path.join(C.RESULTS, 'reduce_meta.json')
    if all((os.path.exists(p) for p in outs)) and os.path.exists(pmeta):
        log('reduce: exists')
        return
    t0 = time.time()
    subs = C.subjects()
    blocks = {h: [] for h in C.HEMIS}
    fsum = np.zeros(C.NCORT, np.float64)
    nrun = 0
    meta = {'cohort': C.COHORT_JSON, 'denoise': C.DENOISE, 'drop': C.DROP, 'subjects': subs, 'rows': [], 'per_subject': [], 'input_sha256': {}}
    from concurrent.futures import ThreadPoolExecutor

    def _load(sid):
        out = [C.load_run(r) for r in C.runs_of(sid)]
        return [(x, raw_l2.mean(0, dtype=np.float64), inf['frames']) for (x, raw_l2, inf) in out]
    pool = ThreadPoolExecutor(int(os.environ.get('PFBN_LOAD_THREADS', '3')))
    futs = [pool.submit(_load, sid) for sid in subs]
    for (i, sid) in enumerate(subs):
        t1 = time.time()
        X = []
        frames = []
        for (x, rm, fr) in futs[i].result():
            fsum += rm
            nrun += 1
            X.append(x)
            frames.append(fr)
        futs[i] = None
        t_load = time.time() - t1
        Y = torch.cat([(lambda t: t - t.mean(0, keepdim=True))(torch.from_numpy(x).to(dev)) for x in X], 0)
        del X
        assert torch.isfinite(Y).all(), sid
        k = int(min(C.N_PC, Y.shape[0] - 1))
        rec = {'subject': sid, 'frames': frames, 'n_pc': k, 'load_seconds': round(t_load, 2)}
        for h in C.HEMIS:
            Yh = C.unit_cols(Y[:, C.HEMI_SL[h]], torch)
            G = (Yh @ Yh.T).double()
            (ev, U) = torch.linalg.eigh(G)
            ev = ev.flip(0).clamp(min=0)
            U = U.flip(1)
            R = U[:, :k].float().T @ Yh
            tot = float(ev.sum())
            rec[h] = {'var_frac_pc': float(ev[:k].sum()) / tot, 'n_zero_var_vertices': int(((R ** 2).sum(0) == 0).sum())}
            blocks[h].append(R.cpu().numpy().astype(np.float32))
            del Yh, G, U, R
        meta['rows'].append(k)
        meta['per_subject'].append(rec)
        del Y
        torch.cuda.empty_cache()
        if (i + 1) % 10 == 0 or i == 0:
            log(f'reduce: {i + 1}/{len(subs)} subjects, {time.time() - t0:.0f}s (load {t_load:.1f}s, frames {frames})')
    for h in C.HEMIS:
        Xc = np.concatenate(blocks[h], 0).astype(np.float32)
        p = os.path.join(C.STACKS, f'X_raw_{h}.npy')
        np.save(p, Xc)
        log(f'reduce: saved {p} {Xc.shape} ({Xc.nbytes / 1000000000.0:.2f} GB)')
    fav = (fsum / nrun).astype(np.float32)
    for h in C.HEMIS:
        np.save(os.path.join(C.RESULTS, f'fmri_avg_tmean_{h}.npy'), fav[C.HEMI_SL[h]][None, :])
    meta['seconds'] = time.time() - t0
    meta['n_runs'] = nrun
    json.dump(meta, open(pmeta, 'w'), indent=1)
    C.Timings.put('prep', 'reduce', time.time() - t0, {'n_subjects': len(subs), 'n_runs': nrun})

def seed_mean_from_gradient_matrix(path, hemi, chunk=1024):
    t0 = time.time()
    img = nib.load(path)
    assert img.shape == (C.NV, C.NV), img.shape
    cidx = C.cortex_idx(hemi)
    is_c = np.zeros(C.NV, bool)
    is_c[cidx] = True
    sum_axis1 = np.zeros(C.NV, np.float64)
    sum_axis0 = np.zeros(C.NV, np.float64)
    absmed_cols = 0.0
    absmed_rows = 0.0
    for a in range(0, C.NV, chunk):
        G = np.asarray(img.dataobj[:, a:a + chunk], dtype=np.float64)
        G[~np.isfinite(G)] = 0.0
        cols_c = is_c[a:a + chunk]
        sum_axis1 += G[:, cols_c].sum(1)
        sum_axis0[a:a + chunk] = G[is_c].sum(0)
        absmed_cols += np.abs(G[:, ~cols_c]).sum()
        absmed_rows += np.abs(G[~is_c]).sum()
    nmed = int((~is_c).sum())
    absmed_cols /= max(nmed * C.NV, 1)
    absmed_rows /= max(nmed * C.NV, 1)
    seeds_are_cols = absmed_cols <= absmed_rows
    m = (sum_axis1 if seeds_are_cols else sum_axis0) / len(cidx)
    m[~is_c] = np.nan
    info = {'path': path, 'seeds_axis': 'columns' if seeds_are_cols else 'rows', 'map_cortex_mean': float(np.nanmean(m)), 'map_cortex_p99': float(np.nanpercentile(m, 99)), 'seconds': time.time() - t0}
    return (m.astype(np.float32), info)

def group_fc_gradient(X_path, hemi, n_sub):
    out = os.path.join(C.GRAD, f'{hemi}_gradient_matrix.dscalar.nii')
    local = os.path.join(C.TMPFS, os.path.basename(out))
    if os.path.exists(out):
        log(f'gradient {hemi}: exists')
        return (out, None)
    t0 = time.time()
    X = torch.from_numpy(np.load(X_path, mmap_mode='r')[:]).to(dev)
    fc = (X.T @ X / n_sub).cpu().numpy().astype(np.float32)
    del X
    torch.cuda.empty_cache()
    t_fc = time.time() - t0
    fc_diag_mean = float(np.mean(np.diag(fc)))
    cidx = C.cortex_idx(hemi)
    full = np.zeros((C.NV, C.NV), np.float32)
    full[np.ix_(cidx, cidx)] = fc
    del fc
    tmpl = nib.load(os.path.join(C.S4TMP, f'{hemi}_rsfmri.dtseries.nii'))
    bax = tmpl.header.get_axis(1)
    assert bax.size == C.NV, bax.size
    hdr = tmpl.header.from_axes((bax, bax))
    tmp_fc = os.path.join(C.TMPFS, f'{hemi}_fc.dconn.nii')
    nib.cifti2.Cifti2Image(dataobj=full, header=hdr).to_filename(tmp_fc)
    del full
    surf = os.path.join(C.S4TMP, f'S1200.{C.HEMI_LETTER[hemi]}.sphere.32k_fs_LR.surf.gii')
    cmd = [C.WB, '-cifti-gradient', tmp_fc, 'ROW', local, '-left-surface' if hemi == 'lh' else '-right-surface', surf]
    t1 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, OMP_NUM_THREADS=os.environ.get('PFBN_WB_THREADS', '16')))
    if r.returncode != 0:
        raise RuntimeError(f'wb_command failed: {r.stderr[-2000:]}')
    t_wb = time.time() - t1
    os.remove(tmp_fc)
    info = {'fc_seconds': t_fc, 'wb_seconds': t_wb, 'total_seconds': time.time() - t0, 'n_sub': n_sub, 'fc_diag_mean': fc_diag_mean, 'nas_path': out}
    log(f'gradient {hemi}: fc {t_fc:.0f}s wb_command {t_wb:.0f}s (fc diag mean {fc_diag_mean:.3f})')
    C.Timings.put('prep', f'gradient_{hemi}', info['total_seconds'], {'wb_seconds': round(t_wb, 1)})
    return (local, info)

def stage_grad(keep_gradient=False):
    n_sub = len(C.subjects())
    infos = {}
    for h in C.HEMIS:
        p = os.path.join(C.RESULTS, f'gmap_{h}.npy')
        if os.path.exists(p):
            log(f'gmap {h}: exists')
            continue
        (gpath, ginfo) = group_fc_gradient(os.path.join(C.STACKS, f'X_raw_{h}.npy'), h, n_sub)
        (m, info) = seed_mean_from_gradient_matrix(gpath, h)
        np.save(p, m)
        info['gradient'] = ginfo
        infos[h] = info
        if ginfo is not None:
            if keep_gradient:
                shutil.move(gpath, ginfo['nas_path'])
                info['path'] = ginfo['nas_path']
            else:
                os.remove(gpath)
                info['path'] = 'deleted (4.2 GB; only the seed-mean map is used downstream)'
        log(f'gmap {h}: done')
    if infos:
        json.dump(infos, open(os.path.join(C.RESULTS, 'gmap_info.json'), 'w'), indent=1)
R_GRID = np.arange(2.0, 80.01, 0.5)

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

def stage_filter():
    pj = os.path.join(C.RESULTS, 'dynamics_fit.json')
    if all((os.path.exists(os.path.join(C.STACKS, f'X_filt_{h}.npy')) for h in C.HEMIS)) and os.path.exists(pj):
        log('filter: exists')
        return
    t0 = time.time()
    meta = json.load(open(os.path.join(C.RESULTS, 'reduce_meta.json')))
    rows = meta['rows']
    n = len(rows)
    off = np.concatenate([[0], np.cumsum(rows)])
    fit_out = {}
    for h in C.HEMIS:
        z = np.load(os.path.join(C.MODES, f'modes_{C.HEMI_LETTER[h]}.npz'))
        assert np.array_equal(z['idx'], C.cortex_idx(h))
        Psi = torch.from_numpy(z['psi']).to(dev)
        mass = torch.from_numpy(z['mass']).to(dev)
        lam = z['evals'].astype(np.float64)
        K = Psi.shape[1]
        X = np.load(os.path.join(C.STACKS, f'X_raw_{h}.npy'), mmap_mode='r')
        assert X.shape[0] == off[-1], X.shape
        A = []
        P = np.zeros(K)
        for i in range(n):
            Xs = torch.from_numpy(np.asarray(X[off[i]:off[i + 1]])).to(dev)
            Ai = Psi.T @ (mass[:, None] * Xs.T)
            P += ((Ai ** 2).sum(1) / C.N_PC).cpu().numpy()
            A.append(Ai)
        P /= n
        fw = fit_wave(P, lam, R_GRID)
        Sj = fw['s'] / (1 + fw['r_s'] ** 2 * lam) ** 2
        w = Sj / (Sj + fw['n'])
        wt = torch.from_numpy(w.astype(np.float32)).to(dev)
        blocks = [C.unit_cols((Psi @ (A[i] * wt[:, None])).T, torch).cpu().numpy() for i in range(n)]
        Xf = np.concatenate(blocks, 0).astype(np.float32)
        np.save(os.path.join(C.STACKS, f'X_filt_{h}.npy'), Xf)
        fit_out[h] = dict(r_s_mm=fw['r_s'], s=fw['s'], n=fw['n'], r2_log=fw['r2_log'], sum_w=float(w.sum()))
        log(f"filter {h}: r_s {fw['r_s']:.1f} mm, R2(log) {fw['r2_log']:.3f}")
        del A, Psi, blocks
        torch.cuda.empty_cache()
    json.dump({'fits': fit_out}, open(pj, 'w'), indent=1)
    C.Timings.put('prep', 'filter', time.time() - t0)
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stages', default='reduce,filter,grad')
    ap.add_argument('--keep_gradient', action='store_true')
    a = ap.parse_args()
    for st in a.stages.split(','):
        log(f'=== {st}')
        t0 = time.time()
        if st == 'reduce':
            stage_reduce()
        elif st == 'filter':
            stage_filter()
        elif st == 'grad':
            stage_grad(a.keep_gradient)
        else:
            raise SystemExit(st)
        log(f'=== {st} done {time.time() - t0:.0f}s')
    log('PREP_END')
