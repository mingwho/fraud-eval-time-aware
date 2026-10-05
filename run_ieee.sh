#!/bin/sh
# Run one IEEE-CIS experiment, one process per model:  ./run_ieee.sh <exp> model:threads ... [-- extra run.py args]
cd "$(dirname "$0")"
exp=$1; shift
specs=""; while [ $# -gt 0 ] && [ "$1" != "--" ]; do specs="$specs $1"; shift; done
[ "$1" = "--" ] && shift
for spec in $specs; do
  m=${spec%:*}; n=${spec#*:}
  # Cap BLAS threads too: an unlimited BLAS pool thrashes when several jobs share the machine.
  OMP_NUM_THREADS=$n OPENBLAS_NUM_THREADS=$n VECLIB_MAXIMUM_THREADS=$n N_THREADS=$n MLP_DEVICE=mps \
    .venv/bin/python -W ignore src/run.py --exp $exp --dataset ieee --models $m "$@" >> logs/ieee_${exp}_$m.log 2>&1 &
done
wait
