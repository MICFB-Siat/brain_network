#!/bin/bash
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=${PYTHON:-python}
cfg() { "$PY" "$ROOT/pfbn/settings.py" "$@"; }
L=$1; G=${2:-0}; M=${3:-v5_stim_t03_r30}; NEED=${4:-20}
E=$(cfg outputs reproducibility); T=$(cfg tmp reproducibility)
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} CUDA_VISIBLE_DEVICES=$G
export PFBN_STACKS_ROOT=$T/stacks PFBN_TMPFS=$T/tmpfs PFBN_LOAD_THREADS=${PFBN_LOAD_THREADS:-3}
mkdir -p "$E/logs/claims" "$T"
while read -r n; do
  [ -z "$n" ] && continue
  mkdir "$E/logs/claims/$n" 2>/dev/null || continue
  [ -f "$E/atlases/$n/results/DONE" ] && continue
  while [ $(df -BG --output=avail "$T" | tail -1 | tr -dc 0-9) -lt $NEED ]; do sleep 30; done
  export PFBN_COHORT=$E/cohorts/repro_$n.json; lg=$E/logs/build_$n.log
  echo "== START $(date) $n gpu=$G $M" > "$lg"
  cd "$ROOT/pfbn/pipeline"
  ( "$PY" prep.py --stages reduce,filter,grad && "$PY" seed.py && "$PY" mrf.py --method "$M" ) >> "$lg" 2>&1; rc=$?
  S=$T/atlases/$n/results; D=$E/atlases/$n/results; mkdir -p "$D"
  cp "$S"/labels_*.npy "$S"/*.json "$S"/gmap_*.npy "$D/" 2>/dev/null; cp -r "$T/atlases/$n/logs" "$E/atlases/$n/" 2>/dev/null
  [ $rc = 0 ] && touch "$D/DONE"
  echo "END rc=$rc $(date)" >> "$lg"; echo "$n rc=$rc gpu=$G $(date)" >> "$E/logs/done.log"
  rm -rf "$T/atlases/$n" "$T/stacks/repro_$n" "$T/tmpfs/repro_$n" "$T/nas/repro_$n"
done < "$L"
