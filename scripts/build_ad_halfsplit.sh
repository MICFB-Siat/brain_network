#!/bin/bash
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=${PYTHON:-python}
cfg() { "$PY" "$ROOT/pfbn/settings.py" "$@"; }
E=$(cfg outputs ad_halfsplit); GA=${1:-0}; GB=${2:-0}
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
mkdir -p "$E/logs"
( bash "$ROOT/scripts/build_atlas.sh" "$E/cohorts/adsplit_ad_case_A.json" "$GA" v5_stim_t03_r30 > "$E/logs/build_A.log" 2>&1; echo "DONE A rc=$? $(date)" >> "$E/logs/build_done.log" ) &
( cd "$ROOT/pfbn/pipeline" && CUDA_VISIBLE_DEVICES=$GB PFBN_COHORT="$E/cohorts/adsplit_ad_case_B.json" "$PY" prep.py --stages reduce,filter,grad > "$E/logs/build_B.log" 2>&1
  echo "DONE B rc=$? $(date)" >> "$E/logs/build_done.log"; rm -rf "$(cfg tmp pfbn_stacks)/adsplit_ad_case_B" ) &
wait; rm -rf "$(cfg tmp adsplit_nas)"
echo "ALL_BUILT $(date)" >> "$E/logs/build_done.log"
