#!/bin/sh
# Full ULB grid: one process per model so each loads a single OpenMP runtime.
cd "$(dirname "$0")"
run() {  # model threads
  for exp in main smote timefeat; do
    N_THREADS=$2 .venv/bin/python -W ignore src/run.py --exp $exp --dataset ulb --models $1 >> logs/ulb_$1.log 2>&1
  done
}
run xgb 3 & run lgbm 3 & run rf 5 & run mlp 2 & run lr 1 &
wait
