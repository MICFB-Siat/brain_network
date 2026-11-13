import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
from pfbn import settings
import json, os, csv, glob, re, sys
import nibabel as nib
from concurrent.futures import ThreadPoolExecutor
CJ = settings.out('cohorts')
os.makedirs(CJ, exist_ok=True)
B = settings.path('datasets')
MAN = f'{B}/derivatives/Lifespan4Groups_20261021/Lifespan4Groups_Pearson219_HalfSplit_20261023/selected_scan_manifest.csv'
OUT = settings.out('atlases')
DROP = 5

def nframes(p):
    return nib.load(p).shape[0]

def write(name, subs, out_sub, denoise='detrend_gsr'):
    j = {'name': name, 'out': f'{OUT}/{name}', 'denoise': denoise, 'drop': DROP, 'subjects': subs}
    json.dump(j, open(f'{CJ}/{name}.json', 'w'), indent=1)
    print(name, len(subs), sum((len(s['runs']) for s in subs)))

def halves(subs, T):
    (A, Bh) = ([], [])
    for s in subs:
        (ra, rb) = ([], [])
        for p in s['runs']:
            n = (T[p] - DROP) // 2
            ra.append({'path': p, 'start': 0, 'stop': n})
            rb.append({'path': p, 'start': n, 'stop': 2 * n})
        A.append(dict(s, runs=ra))
        Bh.append(dict(s, runs=rb))
    return (A, Bh)
rows = list(csv.DictReader(open(MAN)))
life = [{'sid': r['subject_id'], 'group': r['group'], 'dataset': r['dataset'], 'diagnosis': r['diagnosis'], 'age': float(r['age']) if r['age'] else None, 'sex': r['sex'], 'runs': [r['resolved_source_path']]} for r in rows]
pd = []
tw = {}
for line in open(f'{B}/taowu/taowu_patients.tsv').read().replace('\r', '').split('\n')[1:]:
    pass
nc_meta = {}
for r in csv.DictReader(open(f'{B}/_repeated_runs_audit_20261021/neurocon_repeated_run_subjects.csv')):
    nc_meta[r['group'], r['subject_id']] = r
for (site, sub) in (('TaoWu', 'taowu/TaoWu_{g}_fs32k'), ('Neurocon', 'neurocon/Neurocon_{g}_fs32k'), ('PPMI', 'PPMI/PPMI_{g}_fs32k')):
    for g in ('NC', 'PD'):
        files = sorted(glob.glob(f'{B}/{sub.format(g=g)}/*.dtseries.nii'))
        by = {}
        for f in files:
            sid = re.match('(sub-[^_]+)_', os.path.basename(f)).group(1)
            by.setdefault(sid, []).append(f)
        for (sid, fs) in sorted(by.items()):
            age = None
            if site == 'Neurocon' and (g, sid) in nc_meta and nc_meta[g, sid]['age_years']:
                age = float(nc_meta[g, sid]['age_years'])
            pd.append({'sid': f'{site}_{g}_{sid}', 'group': g, 'site': site, 'age': age, 'runs': fs})
with ThreadPoolExecutor(16) as ex:
    allp = sorted({p for s in life + pd for p in s['runs']})
    T = dict(zip(allp, ex.map(nframes, allp)))
json.dump(T, open(os.path.join(CJ, 'frames.json'), 'w'), indent=1)
bad = [s['sid'] for s in pd if min((T[p] for p in s['runs'])) < 100]
print('pd subjects with a run < 100 frames (excluded):', bad)
pd = [dict(s, runs=[p for p in s['runs'] if T[p] >= 100]) for s in pd if s['sid'] not in bad]
with open(os.path.join(CJ, 'subjects_pd.csv'), 'w') as f:
    w = csv.writer(f)
    w.writerow(['sid', 'group', 'site', 'age', 'n_runs', 'frames'])
    for s in pd:
        w.writerow([s['sid'], s['group'], s['site'], s['age'], len(s['runs']), ';'.join((str(T[p]) for p in s['runs']))])
with open(os.path.join(CJ, 'subjects_lifespan.csv'), 'w') as f:
    w = csv.writer(f)
    w.writerow(['sid', 'group', 'dataset', 'diagnosis', 'age', 'sex', 'frames'])
    for s in life:
        w.writerow([s['sid'], s['group'], s['dataset'], s['diagnosis'], s['age'], s['sex'], T[s['runs'][0]]])
write('lifespan_all', life, None)
for g in ('children', 'adolescents', 'young_adults', 'older_adults'):
    (A, Bh) = halves([s for s in life if s['group'] == g], T)
    write(f'life_{g}_A', A, None)
    write(f'life_{g}_B', Bh, None)
write('pd_all', pd, None)
for g in ('PD', 'NC'):
    (A, Bh) = halves([s for s in pd if s['group'] == g], T)
    write(f'pd_{g}_A', A, None)
    write(f'pd_{g}_B', Bh, None)
