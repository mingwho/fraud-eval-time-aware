#!/bin/sh
cd "$(dirname "$0")"
for exp in main smote timefeat; do
  N_THREADS=6 .venv/bin/python -W ignore src/run.py --exp $exp --dataset ulb --models rf >> logs/ulb_rf.log 2>&1
done
