#!/bin/bash
set -e
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=${PYTHON:-python}
cfg() { "$PY" "$ROOT/pfbn/settings.py" "$@"; }
E=$(cfg outputs r1_figure); V=$ROOT/visualization
mkdir -p "$E/logs" "$E/results"; cd "$E"
export PYTHONDONTWRITEBYTECODE=1
[ -e results/halves_agreement_spin.csv ] || "$PY" "$V/r1_figure_data.py" hnull > logs/hnull.log 2>&1
"$PY" "$V/r1_figure_data.py" render > logs/render.log 2>&1
[ -e results/panel_d_distance.csv ] || "$PY" "$V/r1_figure_data.py" drender > logs/drender.log 2>&1
"$PY" "$V/r1_figure_data.py" fnull > logs/fnull.log 2>&1
"$PY" "$V/fig_r1.py" > logs/fig.log 2>&1
