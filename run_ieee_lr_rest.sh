#!/bin/sh
cd "$(dirname "$0")"
while pgrep -f "exp rolling --dataset ieee --models lr" > /dev/null; do sleep 20; done
./run_ieee.sh cv lr:2 -- --seeds 0
./run_ieee.sh smote lr:2 -- --seeds 0,1,2 --max-rows 200000 --arms weight,smote,smote_leaky
