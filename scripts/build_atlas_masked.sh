#!/bin/bash
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=${PYTHON:-python}
cfg() { "$PY" "$ROOT/pfbn/settings.py" "$@"; }
c=$1; G=${2:-0}
A=$(cfg outputs atlases); AM=$(cfg outputs atlases_masked); CJ=$(cfg outputs cohorts)
mkdir -p "$AM/$c/results" "$CJ/masked"
J=$CJ/masked/$c.json
[ -f "$J" ] || "$PY" -c "import json,sys;d=json.load(open(sys.argv[1]));d['name']=sys.argv[2]+'_v5m';d['out']=sys.argv[3];json.dump(d,open(sys.argv[4],'w'),indent=1)" "$CJ/$c.json" "$c" "$AM/$c" "$J"
for f in init_lh.npy init_rh.npy gmap_lh.npy gmap_rh.npy gmap_info.json seed_info.json; do [ -f "$AM/$c/results/$f" ] || cp -p "$A/$c/results/$f" "$AM/$c/results/$f"; done
export CUDA_VISIBLE_DEVICES=$G PFBN_COHORT=$J
cd "$ROOT/pfbn/pipeline"
echo "== START $(date) $c gpu=$G"
"$PY" prep.py --stages reduce || { echo PREP_FAILED; exit 2; }
"$PY" mrf_masked.py || { echo MRF_FAILED; exit 4; }
rm -rf "$(cfg tmp pfbn_stacks)/${c}_v5m"
echo "MASKED_END $(date)"
