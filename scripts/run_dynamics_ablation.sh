#!/bin/bash
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=${PYTHON:-python}
cfg() { "$PY" "$ROOT/pfbn/settings.py" "$@"; }
E=$(cfg outputs dynamics_ablation)
read -r -a GP <<< "${GPUS:-0 0 0}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
mkdir -p "$E/logs"
run() { c=$1; g=$2
  bash "$ROOT/scripts/build_atlas.sh" "$E/cohorts/ablation_${c}_A.json" "$g" v5_stim_t03_r30,v4_stim_t03_c1 > "$E/logs/build_${c}.log" 2>&1
  echo "DONE $c rc=$? $(date)" >> "$E/logs/build_done.log"; rm -rf "$(cfg tmp ablation_nas)/ablation_${c}_A"; }
for wave in "life_children life_adolescents life_young_adults" "life_older_adults pd_NC pd_PD"; do
  i=0; for c in $wave; do run "$c" "${GP[$i]}" & i=$((i + 1)); sleep 15; done; wait
done
echo "ALL_BUILT $(date)" >> "$E/logs/build_done.log"
