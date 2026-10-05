"""Turn stored runs into the tables, figures and numbers used in the paper.

    python src/analyze.py            # writes paper/generated/*
"""
import json
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

import data
from metrics import BUDGETS, evaluate

RESULTS = data.ROOT / "results"
OUT = data.ROOT / "paper" / "generated"
MODELS = ["lr", "rf", "xgb", "lgbm", "mlp"]
MODEL_NAMES = {"lr": "Logistic regression", "rf": "Random forest", "xgb": "XGBoost",
               "lgbm": "LightGBM", "mlp": "MLP"}
DATASETS = {"ieee": "IEEE-CIS", "ulb": "ULB"}
TEST_START = 0.8  # the chronological test set is the last 20% of rows


@lru_cache
def dataset(name):
    return data.load(name)


def collect():
    """One row per run, metrics flattened."""
    rows = []
    for f in RESULTS.rglob("seed*.json"):
        info = json.loads(f.read_text())
        row = {k: info[k] for k in ("exp", "dataset", "protocol", "model", "imb", "seed",
                                    "n_train", "n_test")}
        row["path"] = str(f.with_suffix(".npz"))
        row.update(info["metrics"])
        row.update({f"reported_{k}": v for k, v in info.get("metrics_as_reported", {}).items()})
        row["val_pr_auc"] = info.get("val_pr_auc")
        rows.append(row)
    return pd.DataFrame(rows)


def pm(mean, sd, digits=3):
    return f"{mean:.{digits}f} {{\\scriptsize$\\pm$ {sd:.{digits}f}}}"


def agg(df, metric):
    g = df.groupby(["dataset", "protocol", "model"])[metric]
    return g.mean().unstack("protocol"), g.std(ddof=1).unstack("protocol")


def available(df, exp):
    return [d for d in DATASETS if ((df.exp == exp) & (df.dataset == d)).any()]


# ------------------------------------------------------------------ main comparison
def table_main(df, metrics, caption, label, fname):
    """Random vs chronological split, one row per dataset and model."""
    main = df[df.exp == "main"]
    lines = [r"\begin{table}[t]", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\small", r"\begin{tabular}{ll" + "rrr" * len(metrics) + "}", r"\toprule",
             " & " + "".join(rf" & \multicolumn{{3}}{{c}}{{{name}}}" for _, name in metrics) + r" \\",
             "".join(rf"\cmidrule(lr){{{3 + 3 * i}-{5 + 3 * i}}}" for i in range(len(metrics))),
             "Dataset & Model" + " & Random & Chrono. & $\\Delta$" * len(metrics) + r" \\",
             r"\midrule"]
    for d in available(df, "main"):
        for i, m in enumerate(MODELS):
            cells = [DATASETS[d] if i == 0 else "", MODEL_NAMES[m]]
            for metric, _ in metrics:
                mean, sd = agg(main, metric)
                if (d, m) not in mean.index:
                    cells += ["--"] * 3
                    continue
                r, c = mean.loc[(d, m), "random"], mean.loc[(d, m), "chrono"]
                cells += [pm(r, sd.loc[(d, m), "random"]), pm(c, sd.loc[(d, m), "chrono"]),
                          f"{c - r:+.3f}"]
            lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\addlinespace")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table}"]
    (OUT / fname).write_text("\n".join(lines) + "\n")


def period_matched(df):
    """Score random-split models only on their test rows that fall inside the
    chronological test period, so both protocols are judged on the same weeks."""
    rows = []
    for r in df[(df.exp == "main") & (df.protocol == "random")].itertuples():
        ds = dataset(r.dataset)
        z = np.load(r.path)
        keep = z["test_idx"] >= int(len(ds) * TEST_START)
        idx = z["test_idx"][keep]
        m = evaluate(ds.y[idx], z["score"][keep], ds.amount[idx])
        rows.append({"dataset": r.dataset, "model": r.model, "seed": r.seed, **m})
    return pd.DataFrame(rows)


def fast_ap(y, s):
    order = np.argsort(-s, kind="stable")
    y = y[order]
    tp = np.cumsum(y)
    return (tp[y == 1] / (np.flatnonzero(y == 1) + 1)).mean()


def bootstrap_gap(df, n_boot=1000, seed=0):
    """Bootstrap the test rows of every run; CI for mean(chrono) - mean(random) PR-AUC."""
    rng = np.random.default_rng(seed)
    out = []
    main = df[df.exp == "main"]
    for (d, m), grp in main.groupby(["dataset", "model"]):
        ds = dataset(d)
        runs = {p: [np.load(x) for x in g.path] for p, g in grp.groupby("protocol")}
        diffs = np.empty(n_boot)
        for b in range(n_boot):
            means = {}
            for p, zs in runs.items():
                aps = []
                for z in zs:
                    i = rng.integers(0, len(z["score"]), len(z["score"]))
                    aps.append(fast_ap(ds.y[z["test_idx"][i]], z["score"][i]))
                means[p] = np.mean(aps)
            diffs[b] = means["chrono"] - means["random"]
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        out.append({"dataset": d, "model": m, "lo": lo, "hi": hi})
    return pd.DataFrame(out)


# ------------------------------------------------------------------ SMOTE
def table_smote(df):
    w = df[(df.exp == "main") & (df.protocol == "random")]
    s = df[df.exp == "smote"]
    cols = [("Class weights", w, "pr_auc", "f1_at_0.5"),
            ("SMOTE on training rows", s[s.imb == "smote"], "pr_auc", "f1_at_0.5"),
            ("SMOTE before split, as reported", s[s.imb == "smote_leaky"],
             "reported_pr_auc", "reported_f1_at_0.5"),
            ("SMOTE before split, real test rows", s[s.imb == "smote_leaky"],
             "pr_auc", "f1_at_0.5")]
    lines = [r"\begin{table}[t]",
             r"\caption{Oversampling leak under a random split. ``As reported'' scores the "
             r"flawed pipeline on its own oversampled test set; ``real test rows'' scores the "
             r"same models on the genuine transactions of that test set. Mean over 5 seeds.}",
             r"\label{tab:smote}", r"\small",
             r"\begin{tabular}{ll" + "rr" * len(cols) + "}", r"\toprule",
             " & " + "".join(rf" & \multicolumn{{2}}{{c}}{{{c[0]}}}" for c in cols) + r" \\",
             "".join(rf"\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}" for i in range(len(cols))),
             "Dataset & Model" + " & PR-AUC & F1" * len(cols) + r" \\", r"\midrule"]
    for d in available(df, "smote"):
        for i, m in enumerate(MODELS):
            cells = [DATASETS[d] if i == 0 else "", MODEL_NAMES[m]]
            for _, part, a, b in cols:
                part = part[(part.dataset == d) & (part.model == m)]
                cells += [f"{part[a].mean():.3f}", f"{part[b].mean():.3f}"] if len(part) else ["--"] * 2
            lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\addlinespace")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table}"]
    (OUT / "table_smote.tex").write_text("\n".join(lines) + "\n")


# ------------------------------------------------------------------ dataset table
def table_data(names):
    lines = [r"\begin{table}[t]",
             r"\caption{Datasets. The last two columns give the fraud rate in the first 80\% "
             r"and the last 20\% of transactions in time order.}",
             r"\label{tab:data}", r"\small", r"\begin{tabular}{lrrrrrrr}", r"\toprule",
             r"Dataset & Transactions & Fraud cases & Fraud rate & Days & Features & "
             r"First 80\% & Last 20\% \\", r"\midrule"]
    for d in names:
        ds = dataset(d)
        cut = int(len(ds) * TEST_START)
        lines.append(f"{DATASETS[d]} & {len(ds):,} & {int(ds.y.sum()):,} & "
                     f"{100 * ds.y.mean():.2f}\\% & {(ds.t.max() - ds.t.min()) / 86400:.0f} & "
                     f"{ds.X.shape[1]} & {100 * ds.y[:cut].mean():.2f}\\% & "
                     f"{100 * ds.y[cut:].mean():.2f}\\% \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "table_data.tex").write_text("\n".join(lines) + "\n")


# ------------------------------------------------------------------ blocked vs random CV
def cv_frame(df):
    cv = df[df.exp == "cv"].copy()
    cv["scheme"] = cv.protocol.str.split("_").str[0]
    return cv


def table_cv(df):
    cv = cv_frame(df)
    lines = [r"\begin{table}[t]",
             r"\caption{Five-fold cross-validation with random folds versus contiguous time "
             r"blocks. Both schemes test every transaction exactly once. \prauc{}, mean $\pm$ "
             r"standard deviation over the five folds.}",
             r"\label{tab:cv}", r"\small", r"\begin{tabular}{llrrr}", r"\toprule",
             r"Dataset & Model & Random folds & Time blocks & $\Delta$ \\", r"\midrule"]
    for d in [x for x in DATASETS if (cv.dataset == x).any()]:
        for i, m in enumerate(MODELS):
            part = cv[(cv.dataset == d) & (cv.model == m)].groupby("scheme").pr_auc
            if len(part) < 2:
                continue
            mean, sd = part.mean(), part.std(ddof=1)
            lines.append(" & ".join([DATASETS[d] if i == 0 else "", MODEL_NAMES[m],
                                     pm(mean["randcv"], sd["randcv"]),
                                     pm(mean["blockcv"], sd["blockcv"]),
                                     f"{mean['blockcv'] - mean['randcv']:+.3f}"]) + r" \\")
        lines.append(r"\addlinespace")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table}"]
    (OUT / "table_cv.tex").write_text("\n".join(lines) + "\n")


# ------------------------------------------------------------------ mechanism
def proximity(name, n_seeds=5):
    """How close in time, and how similar, are test fraud cases to training fraud cases?"""
    import splits

    ds = dataset(name)
    key = pd.util.hash_pandas_object(pd.DataFrame(ds.X), index=False).to_numpy()

    def stats(dev, te):
        trf, tef = dev[ds.y[dev] == 1], te[ds.y[te] == 1]
        tt = np.sort(ds.t[trf])
        pos = np.searchsorted(tt, ds.t[tef])
        before = np.abs(ds.t[tef] - tt[np.clip(pos - 1, 0, len(tt) - 1)])
        after = np.abs(tt[np.clip(pos, 0, len(tt) - 1)] - ds.t[tef])
        gap = np.minimum(before, after)
        return {"exact_duplicate_in_train": np.isin(key[tef], key[trf]).mean(),
                "train_fraud_within_1min": (gap <= 60).mean(),
                "train_fraud_within_10min": (gap <= 600).mean(),
                "train_fraud_within_1h": (gap <= 3600).mean(),
                "median_gap_seconds": np.median(gap)}

    rnd = []
    for seed in range(n_seeds):
        tr, va, te = splits.random_split(ds.y, seed)
        rnd.append(stats(np.concatenate([tr, va]), te))
    tr, va, te = splits.chrono_split(len(ds))
    return pd.DataFrame({"random": pd.DataFrame(rnd).mean(),
                         "chrono": pd.Series(stats(np.concatenate([tr, va]), te))})


def summary(df):
    """Plain-text digest of every number quoted in the prose."""
    lines = []
    main = df[df.exp == "main"]
    pmatch = period_matched(df)
    boot = bootstrap_gap(df).set_index(["dataset", "model"])
    for d in available(df, "main"):
        ds = dataset(d)
        n, cut = len(ds), int(len(ds) * TEST_START)
        lines.append(f"== {d}: n={n} fraud={ds.y.sum()} rate={ds.y.mean():.5f} "
                     f"days={(ds.t.max() - ds.t.min()) / 86400:.1f} features={ds.X.shape[1]} "
                     f"chrono-test fraud rate={ds.y[cut:].mean():.5f} (n_fraud={ds.y[cut:].sum()}) "
                     f"first-80% rate={ds.y[:cut].mean():.5f}")
        for metric in ["pr_auc", "roc_auc"] + [f"{k}@{b:g}" for b in BUDGETS
                                                 for k in ("recall", "value_recall", "precision")]:
            mean, sd = agg(main[main.dataset == d], metric)
            mean, sd = mean.loc[d], sd.loc[d]
            lines.append(f"-- {metric}")
            for m in MODELS:
                if m not in mean.index:
                    continue
                r, c = mean.loc[m, "random"], mean.loc[m, "chrono"]
                extra = ""
                if metric == "pr_auc":
                    pmm = pmatch[(pmatch.dataset == d) & (pmatch.model == m)].pr_auc
                    lo, hi = boot.loc[(d, m), ["lo", "hi"]]
                    extra = (f" | period-matched random {pmm.mean():.4f}+-{pmm.std(ddof=1):.4f}"
                             f" | boot CI of gap [{lo:+.4f}, {hi:+.4f}]")
                lines.append(f"{m:5s} random {r:.4f}+-{sd.loc[m, 'random']:.4f} chrono {c:.4f}"
                             f"+-{sd.loc[m, 'chrono']:.4f} gap {c - r:+.4f} "
                             f"rel {100 * (c - r) / r:+.1f}%{extra}")
            order_r = mean["random"].reindex(MODELS).rank(ascending=False)
            order_c = mean["chrono"].reindex(MODELS).rank(ascending=False)
            tau = kendalltau(order_r, order_c).statistic
            lines.append(f"rank random {order_r.astype(int).to_dict()} chrono "
                         f"{order_c.astype(int).to_dict()} kendall tau {tau:.2f}; mean gap "
                         f"{(mean['chrono'] - mean['random']).mean():+.4f}")
        pm_n = pmatch[pmatch.dataset == d]
        lines.append(f"period-matched test: n={pm_n.n.mean():.0f} n_fraud={pm_n.n_fraud.mean():.1f}")
    for d in available(df, "main"):
        lines.append(f"== proximity of test fraud to training fraud: {d}")
        lines.append(proximity(d).round(4).to_string())
    cv = cv_frame(df)
    if len(cv):
        lines.append("== cv (per-fold PR-AUC)")
        g = cv.groupby(["dataset", "model", "scheme"])[["pr_auc", "roc_auc", "recall@0.01"]]
        lines.append(g.agg(["mean", "std"]).round(4).to_string())
    for exp in ("smote", "timefeat", "gap", "rolling"):
        part = df[df.exp == exp]
        if len(part) == 0:
            continue
        lines.append(f"== {exp}")
        cols = ["pr_auc", "roc_auc", "recall@0.01", "value_recall@0.01", "f1_at_0.5"]
        cols += [c for c in ("reported_pr_auc", "reported_roc_auc", "reported_f1_at_0.5")
                 if c in part and part[c].notna().any()]
        g = part.groupby(["dataset", "protocol", "imb", "model"])[cols]
        lines.append(g.mean().round(4).to_string())
        lines.append("std")
        lines.append(g.std(ddof=1).round(4).to_string())
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n")
    return "\n".join(lines)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = collect()
    df.drop(columns="path").to_csv(OUT / "runs.csv", index=False)
    table_main(df, [("pr_auc", "PR-AUC"), ("roc_auc", "ROC-AUC")],
               "Random versus chronological split. Mean $\\pm$ standard deviation over 5 seeds; "
               "$\\Delta$ = chronological $-$ random.", "tab:main", "table_main.tex")
    table_main(df, [("recall@0.005", "Recall at 0.5\\% budget"), ("recall@0.01", "Recall at 1\\% budget"),
                    ("value_recall@0.01", "Value recall at 1\\% budget")],
               "Alert-budget metrics under the two splits: share of fraud cases (recall) and of "
               "fraud value caught when the highest-scored 0.5\\% or 1\\% of test transactions "
               "are flagged.", "tab:budget", "table_budget.tex")
    table_data([d for d in DATASETS if d in available(df, "main")])
    if (df.exp == "smote").any():
        table_smote(df)
    if (df.exp == "cv").any():
        table_cv(df)
    print(summary(df))


if __name__ == "__main__":
    main()
