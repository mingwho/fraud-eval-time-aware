# Random Splits Overstate Fraud Detection

Code and paper source for a time-aware re-evaluation of classical fraud-detection
models on public payment data (ULB credit card fraud, IEEE-CIS). Every model is
trained and tested under a random split and under chronological protocols with the
same tuning budget, so that the difference between the scores can be attributed to
the split.

## Layout

- `src/data.py` – dataset loading; rows are sorted by transaction time
- `src/splits.py` – random, chronological, gap, rolling-window and cross-validation splits
- `src/models.py` – the five models and their search spaces
- `src/metrics.py` – PR-AUC, ROC-AUC and alert-budget metrics
- `src/run.py` – runs experiments and stores test scores under `results/`
- `src/analyze.py` – builds the tables and numbers in `paper/generated/`
- `paper/` – LaTeX source (ACM single-column manuscript format)

## Data

The datasets are not redistributed here.

- ULB: save `creditcard.csv` as `data/ulb/creditcard.csv`
  (https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
- IEEE-CIS: accept the competition rules at
  https://www.kaggle.com/competitions/ieee-fraud-detection, then save
  `train_transaction.csv` and `train_identity.csv` in `data/ieee/`.

## Reproducing

```sh
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt

# ULB: random vs chronological, oversampling leak, timestamp ablation, cross-validation
for exp in main smote timefeat; do .venv/bin/python src/run.py --exp $exp --dataset ulb; done
.venv/bin/python src/run.py --exp cv --dataset ulb --seeds 0

# IEEE-CIS (slow: about a day on a 14-core laptop; one process per model is faster)
for exp in main gap; do .venv/bin/python src/run.py --exp $exp --dataset ieee; done
for exp in rolling cv; do .venv/bin/python src/run.py --exp $exp --dataset ieee --seeds 0; done
.venv/bin/python src/run.py --exp smote --dataset ieee --seeds 0,1,2 --max-rows 200000 \
    --arms weight,smote,smote_leaky

.venv/bin/python src/analyze.py      # tables, figure and number macros in paper/generated/
cd paper && tectonic main.tex        # paper/main.pdf
```

Finished runs are skipped, so any command can be re-issued after an interruption.
`N_THREADS` sets the number of threads a process uses; set `OMP_NUM_THREADS` and
`OPENBLAS_NUM_THREADS` to the same value when several processes share a machine.
`MLP_DEVICE=mps` (or `cuda`) trains the MLP on a GPU. The `run_*.sh` scripts are the
exact invocations used for the paper.

Every run stores its test scores under `results/<exp>/<dataset>/<protocol>/<model>/<arm>/`
next to a JSON file with the tuned hyperparameters and metrics; `src/analyze.py`
recomputes all metrics from the stored scores.

## Headline result

On IEEE-CIS, PR-AUC under a chronological split is 24-37% lower than under a random
split for all five models, and controls that score both protocols on the same
transactions attribute this to leakage rather than a harder test period. On ULB most
of the apparent drop is a harder final period, while applying SMOTE before the split
inflates PR-AUC on genuine test rows from about 0.86 to 0.99 for the non-linear models.
