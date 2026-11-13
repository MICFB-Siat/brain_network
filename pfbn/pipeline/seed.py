import os, sys, json, time, shutil, subprocess, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from common import log
import numpy as np, torch
from PIL import Image
WORK = C.WORK
dev = 'cuda'
PY = sys.executable

def setup_work():
    os.makedirs(os.path.join(WORK, 'models'), exist_ok=True)
    os.makedirs(os.path.join(WORK, 'tmp'), exist_ok=True)
    for f in glob.glob(os.path.join(C.PFBN_COPY1, '*.py')) + glob.glob(os.path.join(C.PFBN_COPY1, '*.yaml')):
        dst = os.path.join(WORK, os.path.basename(f))
        if not os.path.exists(dst):
            shutil.copy2(f, dst)
    for sub in ('encoder_decoder', 'diffusion'):
        dst = os.path.join(WORK, 'models', sub)
        if not os.path.lexists(dst):
            os.symlink(os.path.realpath(os.path.join(C.CHECKPOINTS, sub)), dst)
    for f in ('fs_LR_32k_medial_mask.mat', 'lh_rsfmri.dtseries.nii', 'rh_rsfmri.dtseries.nii'):
        dst = os.path.join(WORK, 'tmp', f)
        if not os.path.lexists(dst):
            os.symlink(os.path.join(C.S4TMP, f), dst)
    tdst = os.path.join(WORK, 'tmp', 'template')
    if not os.path.lexists(tdst):
        os.symlink(os.path.join(C.S4TMP, 'template'), tdst)

def fps_sphere(xyz_unit, n, start):
    P = torch.from_numpy(xyz_unit.astype(np.float32)).to(dev)
    sel = [start]
    dmin = torch.arccos((P @ P[start]).clamp(-1, 1))
    for _ in range(n - 1):
        k = int(torch.argmax(dmin))
        sel.append(k)
        dmin = torch.minimum(dmin, torch.arccos((P @ P[k]).clamp(-1, 1)))
    return (np.array(sel), float(dmin.max()), float(dmin.mean()))

def seeded_cmd(script, args):
    code = "import sys, random, numpy, torch, runpy; random.seed(0); numpy.random.seed(0); torch.manual_seed(0); sys.argv=[%r]+%r; runpy.run_path(%r, run_name='__main__')" % (script, list(args), script)
    return [PY, '-c', code]

def run_sub(cmd, logname):
    logf = os.path.join(C.LOGS, logname)
    with open(logf, 'w') as f:
        r = subprocess.run(cmd, cwd=WORK, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise RuntimeError(f'subprocess failed rc={r.returncode}, see {logf}')
    return logf

def main():
    if all((os.path.exists(os.path.join(C.RESULTS, f'init_{h}.npy')) for h in C.HEMIS)):
        log('seed: exists')
        return
    setup_work()
    sys.path.insert(0, WORK)
    info = {}
    t_all = time.time()
    base = C.NAME
    for h in C.HEMIS:
        name = f'{base}_{h}'
        d = os.path.join(WORK, 'tmp', name, 'FeatExt_Prepare')
        os.makedirs(d, exist_ok=True)
        cidx = C.cortex_idx(h)
        (xyz, _) = C.sphere(h)
        u = xyz[cidx]
        u = u / np.linalg.norm(u, axis=1, keepdims=True)
        start = int(np.argmax(u @ (u.mean(0) / np.linalg.norm(u.mean(0)))))
        (sel, dmax, dmean) = fps_sphere(u, 1024, start)
        z = np.load(os.path.join(C.MODES, f'modes_{C.HEMI_LETTER[h]}.npz'))
        fiedler = z['psi'][:, 1]
        idx = sel[np.argsort(fiedler[sel])]
        X = torch.from_numpy(np.asarray(np.load(os.path.join(C.STACKS, f'X_filt_{h}.npy'), mmap_mode='r')[:, idx])).to(dev)
        X = C.unit_cols(X, torch)
        r = (X.T @ X).cpu().numpy().astype(np.float64)
        r = np.clip(np.nan_to_num(r), -1, 1)
        img = ((r + 1.0) / 2.0 * 255.0).astype(np.uint8)
        Image.fromarray(np.stack([img] * 3, -1), mode='RGB').save(os.path.join(d, 'avgconnectivity.tiff'), compression=None)
        np.save(os.path.join(d, 'index.npy'), idx.astype(np.int64))
        info[h] = {'fps_start': start, 'tiff_offdiag_mean': float(img[~np.eye(1024, dtype=bool)].mean())}
    lora_dir = os.path.join(C.MODELS, 'lora')
    t0 = time.time()
    if not all((os.path.exists(os.path.join(lora_dir, f)) for f in ('encoder.pt', 'decoder.pt', 'diffusion.pt'))):
        run_sub(seeded_cmd(os.path.join(WORK, 'finetune_lora.py'), ['--work_dir', WORK, '--dataset_base', base, '--epochs', '10', '--lora_r', '8', '--output_dir', lora_dir]), 'seed_finetune_lora.log')
    C.Timings.put('seed', 'lora', time.time() - t0)
    from feature_extraction import extract_zpattern_and_feature
    from d1_model import D1Decoder
    from sklearn.cluster import KMeans
    for h in C.HEMIS:
        name = f'{base}_{h}'
        d = os.path.join(WORK, 'tmp', name)
        tiff = os.path.join(d, 'FeatExt_Prepare', 'avgconnectivity.tiff')
        (z, feat) = extract_zpattern_and_feature(tiff, os.path.join(WORK, 'models/encoder_decoder/encoder_decoder_latest.pth'), os.path.join(WORK, 'models/diffusion/fc_diffusion_latest.pth'), save_zpattern_path=os.path.join(d, 'zpattern.pth'), save_feature_2d_path=os.path.join(d, 'lgd_ravgconnectivity_feature.pth'), lora_ckpt=lora_dir, lora_r=8)
        t0 = time.time()
        ck = os.path.join(C.MODELS, 'd1', f'd1_latest_{h}.pth')
        if not os.path.exists(ck):
            run_sub(seeded_cmd(os.path.join(WORK, 'train_d1.py'), ['--encoder', os.path.join(WORK, 'models/encoder_decoder/encoder_decoder_latest.pth'), '--avgfc', tiff, '--fmri_avg', os.path.join(C.RESULTS, f'fmri_avg_tmean_{h}.npy'), '--hemisphere', h, '--epochs', '1000', '--out', os.path.join(C.MODELS, 'd1')]), f'seed_train_d1_{h}.log')
        C.Timings.put('seed', f'd1_{h}', time.time() - t0)
        V = C.HEMI_N[h]
        d1 = D1Decoder(latent_channels=4, latent_h=32, latent_w=32, vertex_count=V, time_steps=1, hidden_dim=512, down_h=8, down_w=8).to(dev)
        d1.load_state_dict(torch.load(ck, map_location=dev, weights_only=False))
        d1.eval()
        with torch.no_grad():
            zz = z.to(dev)
            zz = zz.unsqueeze(0) if zz.dim() == 3 else zz
            out = d1(zz)
        data_vt = out[0].cpu().numpy().reshape(V, -1)
        km = KMeans(n_clusters=100, random_state=0).fit(data_vt)
        Lp = km.labels_.astype(np.int64)
        idx = np.load(os.path.join(d, 'FeatExt_Prepare', 'index.npy'))
        ref = Lp[idx]
        cidx = C.cortex_idx(h)
        (xyz, _) = C.sphere(h)
        u = xyz[cidx]
        u = u / np.linalg.norm(u, axis=1, keepdims=True)
        P = torch.from_numpy(u.astype(np.float32)).to(dev)
        nearest = torch.argmax(P @ P[torch.from_numpy(idx).to(dev)].T, dim=1).cpu().numpy()
        init = ref[nearest]
        init[idx] = ref
        np.save(os.path.join(C.RESULTS, f'init_{h}.npy'), init.astype(np.int16))
        info[h].update({'d1_out_std': float(data_vt.std()), 'ref_unique_labels': int(len(np.unique(ref))), 'init_unique_labels': int(len(np.unique(init)))})
        log(f"seed {h}: seeds cover {info[h]['ref_unique_labels']} / 100 KMeans labels")
    info['total_seconds'] = time.time() - t_all
    json.dump(info, open(os.path.join(C.RESULTS, 'seed_info.json'), 'w'), indent=1)
    C.Timings.put('seed', 'total', time.time() - t_all)
    log('SEED_END')
if __name__ == '__main__':
    main()
