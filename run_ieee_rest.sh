#!/bin/sh
# After a model's main IEEE-CIS grid has finished, run its remaining experiments:
#   ./run_ieee_rest.sh model:threads
cd "$(dirname "$0")"
m=${1%:*}
while pgrep -f "exp main --dataset ieee --models $m" > /dev/null; do sleep 30; done
./run_ieee.sh gap $1
./run_ieee.sh rolling $1 -- --seeds 0
./run_ieee.sh smote $1 -- --seeds 0,1,2 --max-rows 200000 --arms weight,smote,smote_leaky
./run_ieee.sh cv $1 -- --seeds 0
