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
.venv/bin/python src/run.py --exp main --dataset ulb     # random vs chronological
.venv/bin/python src/run.py --exp smote --dataset ulb    # oversampling leak
.venv/bin/python src/run.py --exp cv --dataset ulb --seeds 0
.venv/bin/python src/analyze.py                          # tables and summary
cd paper && tectonic main.tex
```

Finished runs are skipped, so any command can be re-issued after an interruption.
`N_THREADS` sets the number of threads a process uses.
