import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, rankdata
from sklearn.covariance import LedoitWolf
from train_fc_adapter import load_data, input_root, fc_score, corr
from brainmass_fc import pearson_fc

def baseline_fc(ts):
    p = pearson_fc(ts)
    cov = LedoitWolf().fit(ts).covariance_
    sd = np.sqrt(np.diag(cov))
    return dict(Pearson=p, LedoitWolf=(cov / sd[:, None] / sd[None, :]).astype(np.float32), FC_profile_correlation=np.corrcoef(p).astype(np.float32))

def response_score(p, y, metric):
    if metric.endswith('strength'):
        p = np.abs(p)
        y = np.abs(y)
    if metric.startswith('Spearman'):
        p = rankdata(p)
        y = rankdata(y)
    return corr(p, y)

def holm(p):
    p = np.asarray(p)
    order = np.argsort(p)
    q = np.empty_like(p)
    q[order] = np.minimum(1, np.maximum.accumulate(p[order] * (len(p) - np.arange(len(p)))))
    return q

def main(root):
    cases = load_data(root)
    src = input_root(root)
    mf = json.loads((src / 'manifest.json').read_text())
    records = {x['subject']: x for x in mf['records']}
    labels = np.load(src / 'atlas_labels.npz')
    common = np.all([labels[a] > 0 for a in mf['atlases']], axis=0)
    assert common.sum() == 58609
    models = ['Pearson', 'LedoitWolf', 'FC_profile_correlation', 'BrainMass_v7', 'Random_backbone']
    matrices = {}
    test_matrices = {}
    for (name, branch) in [('BrainMass_v7', 'pretrained'), ('Random_backbone', 'random')]:
        matrices[name] = np.mean([np.load(root / 'results' / f'fc_{branch}_seed{k}.npz')['fc'] for k in range(3)], axis=0)
        test_matrices[name] = np.mean([np.load(root / 'results' / f'test_{branch}_seed{k}.npz')['fc'] for k in range(3)], axis=0)
    rows = []
    diagnostics = []
    checks = []
    rng = np.random.default_rng(0)
    (root / 'results/fc').mkdir(exist_ok=True)
    for (i, c) in enumerate(cases):
        s = c['subject']
        a = c['atlas']
        full = np.concatenate(c['runs'])
        train = np.concatenate(c['runs'][:3])
        pred = baseline_fc(full)
        test = baseline_fc(train)
        for m in matrices:
            pred[m] = matrices[m][i]
            test[m] = test_matrices[m][i]
        truth_fc = pearson_fc(c['runs'][3])
        index = np.searchsorted(c['ks'], labels[a][common])
        assert np.array_equal(c['ks'][index], labels[a][common])
        with np.load(src / f'{s}__responses.npz') as responses:
            for (method, p) in pred.items():
                mineig = float(np.linalg.eigvalsh(p.astype(np.float64)).min())
                check = dict(subject=s, atlas=a, method=method, finite=bool(np.isfinite(p).all()), symmetry_error=float(np.max(np.abs(p - p.T))), diagonal_error=float(np.max(np.abs(np.diag(p) - 1))), min_eigenvalue=mineig, max_abs=float(np.max(np.abs(p))))
                assert check['finite'] and check['symmetry_error'] < 1e-05 and (check['diagonal_error'] < 1e-05) and (mineig > -0.0001) and (check['max_abs'] < 1.00001), check
                checks.append(check)
                diagnostics.append(dict(subject=s, atlas=a, method=method, **fc_score(test[method], truth_fc)))
                v = p[0, 1:201][index]
                for target in mf['targets']:
                    y = responses[target][common].astype(float)
                    available = records[s]['response_availability'][target]
                    for metric in ['Pearson_strength', 'Pearson_signed', 'Spearman_strength', 'Spearman_signed']:
                        valid = available['valid'] and np.isfinite(v).all() and np.isfinite(y).all() and (np.std(v) > 1e-12) and (np.std(y) > 1e-12)
                        rows.append(dict(subject=s, atlas=a, predictor=method, target=target, metric=metric, r=response_score(v, y, metric) if valid else np.nan, valid=valid, reason='ok' if valid else available['reason'] if not available['valid'] else 'invalid_prediction_or_constant', n_vertices=int(common.sum())))
        np.savez_compressed(root / 'results/fc' / f'{s}__{a}.npz', **pred, ks=c['ks'], BrainMass_v7_cortex=pred['BrainMass_v7'][1:201, 1:201], Pearson_cortex=pred['Pearson'][1:201, 1:201])
    df = pd.DataFrame(rows)
    diag = pd.DataFrame(diagnostics)
    summary = []
    for ((target, method, metric), part) in df.groupby(['target', 'predictor', 'metric']):
        w = part.pivot(index='subject', columns='atlas', values='r').reindex(mf['subjects']).dropna()
        delta = (w.PFBN_v5 - w.Micapipe).to_numpy()
        n = len(delta)
        p = float(wilcoxon(delta).pvalue) if np.any(delta != 0) else 1.0
        boot = np.median(delta[rng.integers(0, n, (2000, n))], axis=1)
        summary.append(dict(target=target, predictor=method, metric=metric, n=n, Micapipe_median=w.Micapipe.median(), PFBN_median=w.PFBN_v5.median(), paired_difference_median=np.median(delta), CI_low=np.percentile(boot, 2.5), CI_high=np.percentile(boot, 97.5), n_PFBN_higher=int((delta > 0).sum()), P=p))
    sm = pd.DataFrame(summary)
    primary = (sm.target == 'month1') & (sm.metric == 'Pearson_strength') & sm.predictor.isin(['Pearson', 'BrainMass_v7', 'Random_backbone'])
    assert primary.sum() == 3
    sm['Holm_P_primary'] = np.nan
    sm.loc[primary, 'Holm_P_primary'] = holm(sm.loc[primary, 'P'])
    changes = []
    for a in mf['atlases']:
        for m in ['BrainMass_v7', 'Random_backbone']:
            w = diag[diag.atlas == a].pivot(index='subject', columns='method', values='fc_r')
            d = (w[m] - w.Pearson).to_numpy()
            changes.append(dict(atlas=a, method=m, metric='heldout_FC_reliability', n=len(d), baseline_median=w.Pearson.median(), v7_median=w[m].median(), paired_change_median=np.median(d), P=float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0))
            part = df[(df.atlas == a) & (df.target == 'month1') & (df.metric == 'Pearson_strength')]
            w = part.pivot(index='subject', columns='predictor', values='r').dropna()
            d = (w[m] - w.Pearson).to_numpy()
            changes.append(dict(atlas=a, method=m, metric='DBS_month1_strength', n=len(d), baseline_median=w.Pearson.median(), v7_median=w[m].median(), paired_change_median=np.median(d), P=float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0))
    pd.DataFrame(changes).to_csv(root / 'results/change_vs_Pearson.csv', index=False)
    diag.to_csv(root / 'results/heldout_FC_diagnostics.csv', index=False)
    df.to_csv(root / 'results/individual_metrics.csv', index=False)
    sm.to_csv(root / 'results/comparison_summary.csv', index=False)
    pd.DataFrame(checks).to_csv(root / 'results/matrix_checks.csv', index=False)
    wide = df.pivot(index='subject', columns=['target', 'predictor', 'metric', 'atlas'], values='r')
    wide.columns = ['__'.join(x) for x in wide.columns]
    wide.to_csv(root / 'results/Prism_single_header.csv')
    cfg = json.loads((root / 'results/selected.json').read_text())
    med = diag.groupby(['method', 'atlas']).fc_r.median().unstack()
    lines = ['# V7：BrainMass 融合 FC 实验结果', '', '仅修改 FC。PFBN v5 与 Micapipe 分区、NPI 均保持原样。两种分区使用同一个静息训练适配器。', '', f'静息验证冻结选择：`{json.dumps(cfg, ensure_ascii=False)}`。alpha=0 表示静息验证选择退回普通 Pearson。', '', '## 留出 run4 的 FC 可靠性（皮层上三角相关，患者中位数）', '', '| FC方法 | Micapipe | PFBN v5 |', '|---|---:|---:|']
    for m in models:
        lines.append(f"| {m} | {med.loc[m, 'Micapipe']:.4f} | {med.loc[m, 'PFBN_v5']:.4f} |")
    lines += ['', '## DBS month1 响应强度：固定公共空间', '', '| FC方法 | n | Micapipe中位r | PFBN中位r | 配对差值中位数 | PFBN提高人数 | P | Holm P |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for (_, r) in sm[(sm.target == 'month1') & (sm.metric == 'Pearson_strength')].iterrows():
        lines.append(f'| {r.predictor} | {int(r.n)} | {r.Micapipe_median:.4f} | {r.PFBN_median:.4f} | {r.paired_difference_median:.4f} | {int(r.n_PFBN_higher)} | {r.P:.4f} | {r.Holm_P_primary:.4f} |')
    lines += ['', '## 相对 Pearson 的变化', '', pd.DataFrame(changes).to_string(index=False), '', '所有患者、目标、评价指标均保留在 CSV；不存在按刺激结果选择融合权重或训练轮数。month1 缺失患者保留为空值。', '适配器使用静息 run 之间的 FC 一致性训练。独立 run4 是本轮 FC 模块诊断；不代表图谱构建全过程独立留出，也不证明真实刺激的因果可预测性。', 'BrainMass 迁移使用了官方预训练骨干；未将它冒称为 v0 缺失的 PPMI_Our.pth 或 baseline.pth。', '']
    (root / 'FINAL_REPORT.md').write_text('\n'.join(lines))
    (root / 'results/COMPLETE.json').write_text(json.dumps(dict(n_cases=len(cases), n_matrices=len(checks), n_score_rows=len(df), selected=cfg, n_common_vertices=int(common.sum())), indent=2))
    print(sm[primary].to_string(index=False), flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    a = p.parse_args()
    main(Path(a.root))
