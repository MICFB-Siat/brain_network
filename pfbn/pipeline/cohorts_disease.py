import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
from pfbn import settings
import os, re, csv, json, glob
import nibabel as nib
from concurrent.futures import ThreadPoolExecutor
CJ = settings.out('cohorts')
os.makedirs(CJ, exist_ok=True)
B = settings.path('datasets')
OUT = settings.out('atlases')
MINF = 80
SETS = {'asd_all': [('ABIDE-I', 'ABIDE/ABIDE-I_ASD_Standard', 'case'), ('ABIDE-I', 'ABIDE/ABIDE-I_NC_Standard', 'NC'), ('ABIDE-II', 'ABIDE/ABIDE-II_ASD_Standard', 'case'), ('ABIDE-II', 'ABIDE/ABIDE-II_NC_Standard', 'NC'), ('CRF', 'CRF/CRF_ASD_Standard', 'case'), ('CRF', 'CRF/CRF_NC_Standard', 'NC')], 'mdd_all': [('ds002748', 'RFDP/ds002748_MDD_Standard', 'case'), ('ds002748', 'RFDP/ds002748_NC_Standard', 'NC'), ('ds000171', 'BAML/ds000171_MDD_Standard', 'case'), ('ds000171', 'BAML/ds000171_NC_Standard', 'NC')], 'adhd_all': [('UCLA-CNP', 'UCLA-CNP/UCLA-CNP_ADHD_Standard', 'case'), ('UCLA-CNP', 'UCLA-CNP/UCLA-CNP_NC_Standard', 'NC'), ('KKI', 'ADHD/ADHD_BIDS_Standard', 'KKI')], 'ad_all': [('ADNI', 'ADNI/ADNI_AD_BIDS_Standard', 'case'), ('ADNI', 'ADNI/ADNI_NC_BIDS_Standard', 'NC')]}
kki = {r['ScanDir ID'].lstrip('0'): 'NC' if r['DX'] == '0' else 'case' for r in csv.DictReader(open(f'{B}/ADHD/ADHD_RAW/KKI_preproc/KKI/KKI_phenotypic.csv'))}
allsubs = {}
for (name, parts) in SETS.items():
    subs = []
    for (site, d, grp) in parts:
        by = {}
        for f in sorted(glob.glob(f'{B}/{d}/sub-*.dtseries.nii')):
            sid = re.match('sub-([^_]+)_', os.path.basename(f)).group(1)
            by.setdefault(sid, []).append(f)
        for (sid, fs) in sorted(by.items()):
            g = kki.get(sid.lstrip('0')) if grp == 'KKI' else grp
            if g is None:
                continue
            subs.append({'sid': f'{site}_{g}_{sid}', 'group': g, 'site': site, 'runs': fs})
    allsubs[name] = subs
paths = sorted({p for v in allsubs.values() for s in v for p in s['runs']})
with ThreadPoolExecutor(16) as ex:
    T = dict(zip(paths, ex.map(lambda p: nib.load(p).shape[0], paths)))
json.dump(T, open(os.path.join(CJ, 'frames_disease.json'), 'w'), indent=1)
for (name, subs) in allsubs.items():
    subs = [dict(s, runs=[p for p in s['runs'] if T[p] >= MINF]) for s in subs]
    subs = [s for s in subs if s['runs']]
    json.dump({'name': name, 'out': f'{OUT}/{name}', 'denoise': 'detrend_gsr', 'drop': 5, 'subjects': subs}, open(f'{CJ}/{name}.json', 'w'), indent=1)
    import collections
    print(name, len(subs), dict(collections.Counter(((s['site'], s['group']) for s in subs))), 'runs', sum((len(s['runs']) for s in subs)))
