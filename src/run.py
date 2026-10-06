"""Run experiments. Each run tunes a model on the validation set, refits it on
train+val, scores the test set, and stores the test scores for later analysis.

    python src/run.py --exp main --dataset ulb --models xgb --seeds 0,1,2,3,4

Experiments
    main      random vs chronological split, class weights
    timefeat  same as main, with the raw timestamp kept as a feature
    gap       chronological split with a 7-day gap (reuses hyperparameters of main/chrono)
    rolling   rolling windows, chronological vs random within each window
    cv        5-fold cross-validation with contiguous time blocks vs random folds
    smote     random split with SMOTE applied correctly (training rows only) and
              incorrectly (to the whole dataset before splitting)
Finished runs are skipped, so the command can be re-issued after an interruption.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import optuna
from sklearn.metrics import average_precision_score

import data
import splits
from metrics import evaluate
from models import MODELS, DensePrep

RESULTS = data.ROOT / "results"
optuna.logging.set_verbosity(optuna.logging.WARNING)


def stage(M, imb, amount_col, X, y, fit_rows, eval_rows, seed):
    """Build the matrices for one fit: preprocessing and oversampling see `fit_rows` only."""
    Xf, yf, Xe = X[fit_rows], y[fit_rows], X[eval_rows]
    if imb != "smote_leaky" and (M.dense or imb == "smote"):
        prep = DensePrep(amount_col).fit(Xf)
        Xf, Xe = prep.transform(Xf), prep.transform(Xe)
    if imb == "smote":
        from imblearn.over_sampling import SMOTE

        Xf, yf = SMOTE(random_state=seed).fit_resample(Xf, yf)
    return Xf, yf, Xe


def progress(study, trial):
    if (trial.number + 1) % 10 == 0:
        print(f"  trial {trial.number + 1}: best validation PR-AUC {study.best_value:.4f}", flush=True)


def run(X, y, amount_col, tr, va, te, model, imb, seed, trials, tune_rows=None, fixed=None):
    """Tune on (tr, va), refit on tr+va, score te. `fixed` = (params, n_iter) skips tuning."""
    M = MODELS[model]
    weighted = imb == "weight"
    info = {}
    if fixed is None:
        t0 = time.time()
        fit_rows = tr
        if tune_rows and len(tr) > tune_rows:
            fit_rows = np.sort(np.random.default_rng(seed).choice(tr, tune_rows, replace=False))
        Xf, yf, Xv = stage(M, imb, amount_col, X, y, fit_rows, va, seed)
        yv = y[va]

        def objective(trial):
            m, n_iter = M.fit(M.suggest(trial, weighted), Xf, yf, seed, Xv, yv)
            trial.set_user_attr("n_iter", n_iter)
            return average_precision_score(yv, M.predict(m, Xv))

        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
        )
        study.optimize(objective, n_trials=trials, callbacks=[progress] if trials >= 10 else [])
        params, n_iter = study.best_params, study.best_trial.user_attrs["n_iter"]
        info.update(val_pr_auc=study.best_value, tune_seconds=time.time() - t0)
    else:
        params, n_iter = fixed
    t0 = time.time()
    dev = np.concatenate([tr, va])
    Xf, yf, Xt = stage(M, imb, amount_col, X, y, dev, te, seed)
    m, _ = M.fit(params, Xf, yf, seed, n_iter=n_iter)
    score = M.predict(m, Xt).astype(np.float32)
    info.update(params=params, n_iter=n_iter, fit_seconds=time.time() - t0)
    return score, info


def leaky_smote(ds, seed):
    """The flawed pipeline: scale and oversample the whole dataset, then split."""
    from imblearn.over_sampling import SMOTE

    Xp = DensePrep(ds.amount_col).fit(ds.X).transform(ds.X)
    Xr, yr = SMOTE(random_state=seed).fit_resample(Xp, ds.y)
    return Xr.astype(np.float32), yr.astype(np.int8)


def jobs(args, ds):
    """Yield (protocol tag, imbalance arm, seed, train, val, test)."""
    n = len(ds)
    for seed in args.seeds:
        if args.exp in ("main", "timefeat"):
            if "random" in args.protocols:
                yield ("random", "weight", seed) + splits.random_split(ds.y, seed)
            if "chrono" in args.protocols:
                yield ("chrono", "weight", seed) + splits.chrono_split(n)
        elif args.exp == "gap":
            yield ("chrono_gap", "weight", seed) + splits.chrono_gap_split(ds.t)
        elif args.exp == "rolling":
            for w, proto, tr, va, te in splits.rolling_windows(ds.t, ds.y, seed):
                if proto in args.protocols:
                    yield (f"{proto}_w{w}", "weight", seed, tr, va, te)
        elif args.exp == "cv":
            only = [x for x in args.protocols if x in ("blockcv", "randcv")]
            for f, proto, tr, va, te in splits.cv_folds(ds.y, seed):
                if not only or proto in only:
                    yield (f"{proto}_f{f}", "weight", seed, tr, va, te)
        elif args.exp == "smote":
            # "weight" repeats the class-weight baseline; only needed with --max-rows,
            # otherwise the runs of the main experiment are the same.
            yield ("random", "weight", seed) + splits.random_split(ds.y, seed)
            yield ("random", "smote", seed) + splits.random_split(ds.y, seed)
            yield ("random", "smote_leaky", seed, None, None, None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp", required=True, choices=["main", "timefeat", "gap", "rolling", "cv", "smote"])
    p.add_argument("--dataset", required=True, choices=["ulb", "ieee"])
    p.add_argument("--models", default="lr,rf,xgb,lgbm,mlp", type=lambda s: s.split(","))
    p.add_argument("--seeds", default="0,1,2,3,4", type=lambda s: [int(v) for v in s.split(",")])
    p.add_argument("--protocols", default="random,chrono", type=lambda s: s.split(","))
    p.add_argument("--arms", default="smote,smote_leaky", type=lambda s: s.split(","))
    p.add_argument("--trials", default=30, type=int)
    p.add_argument("--tune-rows", default=None, type=int,
                   help="tune on at most this many training rows (final fit uses all)")
    p.add_argument("--max-rows", default=None, type=int,
                   help="keep only a random subsample of the dataset (smote on large data)")
    args = p.parse_args()

    ds = data.load(args.dataset, with_time=args.exp == "timefeat")
    if args.max_rows and len(ds) > args.max_rows:
        keep = np.sort(np.random.default_rng(0).choice(len(ds), args.max_rows, replace=False))
        ds = data.Dataset(ds.name, ds.X[keep], ds.y[keep], ds.t[keep], ds.amount[keep],
                          ds.features, ds.amount_col)
        RESULTS.mkdir(exist_ok=True)
        np.save(RESULTS / f"{args.exp}_{args.dataset}_rows.npy", keep)

    for model in args.models:
        for proto, imb, seed, tr, va, te in jobs(args, ds):
            if args.exp == "smote" and imb not in args.arms:
                continue
            out = RESULTS / args.exp / args.dataset / proto / model / imb / f"seed{seed}"
            if out.with_suffix(".json").exists():
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            extra = {}
            if imb == "smote_leaky":
                Xr, yr = leaky_smote(ds, seed)
                tr, va, te = splits.random_split(yr, seed)
                score, info = run(Xr, yr, ds.amount_col, tr, va, te, model, imb, seed,
                                  args.trials, args.tune_rows)
                real = te < len(ds)  # SMOTE appends synthetic rows after the originals
                extra = {"y_test": yr[te], "is_real": real}
                # Report the score the flawed pipeline would publish, and the score on
                # the genuine transactions of its test set.
                info["metrics_as_reported"] = evaluate(yr[te], score, np.ones(len(te)))
                info["metrics"] = evaluate(ds.y[te[real]], score[real], ds.amount[te[real]])
            else:
                fixed = None
                if args.exp == "gap":
                    src = RESULTS / "main" / args.dataset / "chrono" / model / imb / f"seed{seed}.json"
                    ref = json.loads(src.read_text())
                    fixed = (ref["params"], ref["n_iter"])
                score, info = run(ds.X, ds.y, ds.amount_col, tr, va, te, model, imb, seed,
                                  args.trials, args.tune_rows, fixed)
                info["metrics"] = evaluate(ds.y[te], score, ds.amount[te])
            np.savez_compressed(out.with_suffix(".npz"), test_idx=te, score=score, **extra)
            info.update(exp=args.exp, dataset=args.dataset, protocol=proto, model=model,
                        imb=imb, seed=seed, n_train=int(len(tr) + len(va)), n_test=int(len(te)))
            out.with_suffix(".json").write_text(json.dumps(info, indent=1))
            m = info["metrics"]
            print(f"{args.dataset} {args.exp} {proto} {model} {imb} seed{seed}: "
                  f"PR-AUC {m['pr_auc']:.4f} ROC-AUC {m['roc_auc']:.4f} "
                  f"R@1% {m['recall@0.01']:.3f} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
