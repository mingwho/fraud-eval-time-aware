#!/bin/sh
# Blocked vs random 5-fold CV on ULB; starts each model once its main chain has finished.
cd "$(dirname "$0")"
run() {  # model threads
  while pgrep -f "dataset ulb --models $1" > /dev/null; do sleep 20; done
  N_THREADS=$2 .venv/bin/python -W ignore src/run.py --exp cv --dataset ulb --models $1 --seeds 0 >> logs/ulb_cv_$1.log 2>&1
}
run lr 1 & run xgb 3 & run lgbm 3 & run rf 6 & run mlp 2 &
wait
