import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
from pfbn import settings
import json, os, re, sys
from pathlib import Path
FW = Path(settings.path('framework'))
EXP = FW / 'experiments/Lifespan4Groups_50each_20261021'
OLD = FW / 'experiments/HCP3T_50_run_PDiv_20261021'
G = Path(settings.path('gradient_cohort'))
ANALYSIS = Path(settings.code('pfbn', 'eigenmodes'))
OLDER_RUN = G / 'adaptive/older_run'
D = Path(settings.path('lifespan'))
W = D / 'MSM_full_20261023_0205'
ASSETS = D / 'Lifespan4Groups_Pearson219_HalfSplit_20261023/assets'
SUBJECT_SOURCES = G / 'runs/full_20261023_0205/subject_sources.json'
OUT = Path(settings.out('boundary_agreement'))
FILL = OUT / 'adaptive_fill'
CACHE = OUT / 'cache'
ITER = 8
METHODS = ('Ours', 'Micapipe200', 'CATO219')
GROUPS = ('children', 'adolescents', 'young_adults', 'older_adults')
sys.path.insert(0, str(ANALYSIS))

def manifest():
    return json.load(open(W / 'manifest.json'))['subjects']

def run_id(sid):
    src = json.load(open(SUBJECT_SOURCES))[sid][0]
    m = re.search('(rfMRI_REST\\d_[LR]{2})', src) or re.search('(run-\\d+)', src)
    return (m.group(1) if m else 'run-01', src)

def ours_dlabel(sid, group):
    (run, _) = run_id(sid)
    if group == 'young_adults':
        cands = [OLD / 'subjects' / sid / run / 'adaptive_parcellation.dlabel.nii']
    elif group == 'older_adults':
        cands = [OLDER_RUN / 'subjects' / sid / 'run-01' / 'adaptive_parcellation.dlabel.nii', OLDER_RUN / 'subjects' / sid / run / 'adaptive_parcellation.dlabel.nii']
    else:
        cands = [EXP / 'subjects' / sid / run / 'adaptive_parcellation.dlabel.nii']
    cands.append(FILL / 'subjects' / sid / run / 'adaptive_parcellation.dlabel.nii')
    for c in cands:
        if c.exists() and (not str(c).startswith(str(FILL)) or (c.parent / 'complete.json').exists()):
            return c
    return None
