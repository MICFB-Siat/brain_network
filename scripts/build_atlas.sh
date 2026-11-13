#!/bin/bash
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=${PYTHON:-python}
cfg() { "$PY" "$ROOT/pfbn/settings.py" "$@"; }
J=$(readlink -f "$1"); G=${2:-0}; M=${3:-v5_stim_t03_r30}
export CUDA_VISIBLE_DEVICES=$G PFBN_COHORT=$J
NAME=$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['name'])" "$J")
STACKS=${PFBN_STACKS_ROOT:-$(cfg tmp pfbn_stacks)}
cd "$ROOT/pfbn/pipeline"
echo "== START $(date) $J gpu=$G methods=$M"
"$PY" prep.py --stages reduce,filter,grad || { echo PREP_FAILED; exit 2; }
"$PY" seed.py || { echo SEED_FAILED; exit 3; }
"$PY" mrf.py --method "$M" || { echo MRF_FAILED; exit 4; }
[ "${KEEP_STACKS:-0}" = 1 ] || rm -rf "$STACKS/$NAME"
echo "COHORT_END $(date)"
