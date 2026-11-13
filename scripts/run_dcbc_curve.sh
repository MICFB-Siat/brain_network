#!/bin/bash
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=${PYTHON:-python}
cfg() { "$PY" "$ROOT/pfbn/settings.py" "$@"; }
E=$(cfg outputs dcbc_curve); mkdir -p "$E/logs"
read -r -a GP <<< "${GPUS:-0 0 0}"
run() { GY_GPU=$1 "$PY" "$ROOT/evaluation/dcbc/dcbc_curve.py" "$2" "$3" > "$E/logs/bins_$2.log" 2>&1; }
( run "${GP[0]}" adolescents life_adolescents; run "${GP[0]}" children life_children ) &
( run "${GP[1]}" older_adults life_older_adults; run "${GP[1]}" young_adults life_young_adults ) &
( run "${GP[2]}" PD pd_PD; run "${GP[2]}" NC pd_NC ) &
wait
echo ALL_BINS_DONE > "$E/logs/bins_ALL_DONE.txt"
