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
    """One row per run. Metrics are recomputed from the stored test scores."""
    rows = []
    for f in RESULTS.rglob("seed*.json"):
        info = json.loads(f.read_text())
        row = {k: info[k] for k in ("exp", "dataset", "protocol", "model", "imb", "seed",
                                    "n_train", "n_test")}
        row["path"] = str(f.with_suffix(".npz"))
        if info["exp"] == "smote":
            row.update(info["metrics"])  # rows may index a subsample; keep stored metrics
        else:
            ds = dataset(info["dataset"])
            z = np.load(row["path"])
            row.update(evaluate(ds.y[z["test_idx"]], z["score"], ds.amount[z["test_idx"]]))
        row.update({f"reported_{k}": v for k, v in info.get("metrics_as_reported", {}).items()})
        row["val_pr_auc"] = info.get("val_pr_auc")
        rows.append(row)
    return pd.DataFrame(rows)


def signed(v, digits=3):
    return f"${'+' if v >= 0 else '-'}${abs(v):.{digits}f}"


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
                          signed(c - r)]
            lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\addlinespace")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table}"]
    (OUT / fname).write_text("\n".join(lines) + "\n")


def table_budget(df):
    """Alert-budget metrics, means over seeds."""
    main = df[df.exp == "main"]
    cols = [("recall@0.005", "Recall, 0.5\\%"), ("recall@0.01", "Recall, 1\\%"),
            ("recall@0.05", "Recall, 5\\%"), ("value_recall@0.01", "Value recall, 1\\%"),
            ("value_recall@0.05", "Value recall, 5\\%")]
    lines = [r"\begin{table}[t]",
             r"\caption{Alert-budget metrics: share of fraud cases (recall) and of fraud value "
             r"(value recall) caught when the highest-scored 0.5\%, 1\% or 5\% of test "
             r"transactions are flagged. R = random split, C = chronological split; mean over 5 "
             r"seeds. Recall cannot exceed the budget divided by the fraud rate, which is 0.14 and "
             r"0.29 for the two smaller budgets on IEEE-CIS.}",
             r"\label{tab:budget}", r"\small",
             r"\begin{tabular}{ll" + "rr" * len(cols) + "}", r"\toprule",
             " & " + "".join(rf" & \multicolumn{{2}}{{c}}{{{name}}}" for _, name in cols) + r" \\",
             "".join(rf"\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}" for i in range(len(cols))),
             "Dataset & Model" + " & R & C" * len(cols) + r" \\", r"\midrule"]
    for d in available(df, "main"):
        for i, m in enumerate(MODELS):
            cells = [DATASETS[d] if i == 0 else "", MODEL_NAMES[m]]
            for metric, _ in cols:
                mean, _sd = agg(main, metric)
                if (d, m) not in mean.index:
                    cells += ["--"] * 2
                    continue
                cells += [f"{mean.loc[(d, m), 'random']:.3f}", f"{mean.loc[(d, m), 'chrono']:.3f}"]
            lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\addlinespace")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table}"]
    (OUT / "table_budget.tex").write_text("\n".join(lines) + "\n")


def table_controls(df, pmatch):
    """Both controls that hold the test transactions fixed, PR-AUC."""
    main = df[df.exp == "main"]
    mean, _ = agg(main, "pr_auc")
    cv = cv_frame(df).groupby(["dataset", "model", "scheme"]).pr_auc.mean().unstack()
    pmm = pmatch.groupby(["dataset", "model"]).pr_auc.mean()
    lines = [r"\begin{table}[t]",
             r"\caption{Controls that score both protocols on the same transactions (\prauc{}). "
             r"Left: random-split models scored only on their test transactions inside the "
             r"chronological test period, against the chronological models. Right: five-fold "
             r"cross-validation with random folds against contiguous time blocks.}",
             r"\label{tab:controls}", r"\small", r"\begin{tabular}{llrrrrrr}", r"\toprule",
             r" & & \multicolumn{3}{c}{Same test period} & \multicolumn{3}{c}{Five-fold "
             r"cross-validation} \\", r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
             r"Dataset & Model & Random & Chrono. & $\Delta$ & Random folds & Time blocks & "
             r"$\Delta$ \\", r"\midrule"]
    for d in available(df, "main"):
        for i, m in enumerate(MODELS):
            cells = [DATASETS[d] if i == 0 else "", MODEL_NAMES[m]]
            if (d, m) in pmm.index and (d, m) in mean.index and not np.isnan(mean.loc[(d, m), "chrono"]):
                r, c = pmm.loc[(d, m)], mean.loc[(d, m), "chrono"]
                cells += [f"{r:.3f}", f"{c:.3f}", signed(c - r)]
            else:
                cells += ["--"] * 3
            if (d, m) in cv.index and cv.loc[(d, m)].notna().all():
                r, c = cv.loc[(d, m), "randcv"], cv.loc[(d, m), "blockcv"]
                cells += [f"{r:.3f}", f"{c:.3f}", signed(c - r)]
            else:
                cells += ["--"] * 3
            lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\addlinespace")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table}"]
    (OUT / "table_controls.tex").write_text("\n".join(lines) + "\n")


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


def bootstrap_gap(df, n_boot=500, seed=0):
    """Bootstrap the test rows of every run; CI for mean(chrono) - mean(random) PR-AUC."""
    rng = np.random.default_rng(seed)
    out = []
    main = df[df.exp == "main"]
    for (d, m), grp in main.groupby(["dataset", "model"]):
        ds = dataset(d)
        if grp.protocol.nunique() < 2:
            continue
        runs = {}
        for p, g in grp.groupby("protocol"):
            zs = [np.load(x) for x in g.path]
            runs[p] = [(ds.y[z["test_idx"]], z["score"]) for z in zs]
        diffs = np.empty(n_boot)
        for b in range(n_boot):
            means = {}
            for p, pairs in runs.items():
                aps = []
                for y, score in pairs:
                    i = rng.integers(0, len(y), len(y))
                    aps.append(fast_ap(y[i], score[i]))
                means[p] = np.mean(aps)
            diffs[b] = means["chrono"] - means["random"]
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        out.append({"dataset": d, "model": m, "lo": lo, "hi": hi})
    return pd.DataFrame(out)


# ------------------------------------------------------------------ SMOTE
def table_smote(df):
    s = df[df.exp == "smote"]
    # Class-weight baseline: rerun inside the smote experiment where the data were
    # subsampled, otherwise identical to the random-split runs of the main experiment.
    sub = s[s.imb == "weight"]
    w = pd.concat([sub, df[(df.exp == "main") & (df.protocol == "random")
                           & ~df.dataset.isin(sub.dataset.unique())]])
    cols = [("\\shortstack{Class\\\\weights}", w, "pr_auc", "f1_at_0.5"),
            ("\\shortstack{SMOTE on\\\\training rows}", s[s.imb == "smote"], "pr_auc", "f1_at_0.5"),
            ("\\shortstack{SMOTE before split,\\\\as reported}", s[s.imb == "smote_leaky"],
             "reported_pr_auc", "reported_f1_at_0.5"),
            ("\\shortstack{SMOTE before split,\\\\genuine test rows}", s[s.imb == "smote_leaky"],
             "pr_auc", "f1_at_0.5")]
    lines = [r"\begin{table}[t]",
             r"\caption{Oversampling leak under a random split. ``As reported'' scores the "
             r"flawed pipeline on its own oversampled test set; ``genuine test rows'' scores the "
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


def entity_overlap(n_seeds=5):
    """IEEE-CIS: share of test transactions whose card key also occurs in training.
    The key (card1, card2, card3, card5, addr1, first-seen day = day - D1) is a
    heuristic proxy for one payment card."""
    import splits

    ds = dataset("ieee")
    col = lambda c: ds.X[:, ds.features.index(c)]
    parts = {c: col(c) for c in ("card1", "card2", "card3", "card5", "addr1")}
    parts["start"] = np.floor(ds.t / 86400) - col("D1")
    key = pd.util.hash_pandas_object(pd.DataFrame(parts).fillna(-1), index=False).to_numpy()

    def stats(dev, te):
        trf = dev[ds.y[dev] == 1]
        tef, ten = te[ds.y[te] == 1], te[ds.y[te] == 0]
        return {"test_fraud_key_among_train_fraud": np.isin(key[tef], key[trf]).mean(),
                "test_fraud_key_in_train": np.isin(key[tef], key[dev]).mean(),
                "test_legit_key_among_train_fraud": np.isin(key[ten], key[trf]).mean(),
                "test_legit_key_in_train": np.isin(key[ten], key[dev]).mean()}

    rnd = []
    for seed in range(n_seeds):
        tr, va, te = splits.random_split(ds.y, seed)
        rnd.append(stats(np.concatenate([tr, va]), te))
    out = {"random": pd.DataFrame(rnd).mean()}
    for name, (tr, va, te) in {"chrono": splits.chrono_split(len(ds)),
                               "chrono_gap": splits.chrono_gap_split(ds.t)}.items():
        out[name] = pd.Series(stats(np.concatenate([tr, va]), te))
    g = pd.DataFrame({"key": key, "y": ds.y}).groupby("key").y.agg(["mean", "size"])
    multi = g[g["size"] > 1]
    note = (f"distinct keys {len(g)}; keys with >1 transaction {len(multi)}; of those, share "
            f"with a single label {((multi['mean'] == 0) | (multi['mean'] == 1)).mean():.4f}; "
            f"keys with any fraud {(g['mean'] > 0).sum()}")
    return pd.DataFrame(out), note


# ------------------------------------------------------------------ figures
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1a1a19", "#6b6a63", "#e6e5df"
SERIES = {"random": ("Random split", BLUE, "o"), "chrono": ("Chronological split", ORANGE, "s")}


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def weekly_frame(df, d="ieee"):
    """PR-AUC by week of the chronological test period, for chronological models and
    for random-split models scored on their test rows of the same week."""
    ds = dataset(d)
    start = ds.t[int(len(ds) * TEST_START)]
    rows = []
    for r in df[(df.exp == "main") & (df.dataset == d)].itertuples():
        z = np.load(r.path)
        idx, score = z["test_idx"], z["score"]
        week = np.floor((ds.t[idx] - start) / (7 * 86400)).astype(int)
        for wk in range(6):
            keep = week == wk
            if ds.y[idx[keep]].sum() < 20:
                continue
            rows.append({"model": r.model, "protocol": r.protocol, "seed": r.seed, "week": wk + 1,
                         "pr_auc": fast_ap(ds.y[idx[keep]], score[keep]),
                         "n_fraud": int(ds.y[idx[keep]].sum())})
    return pd.DataFrame(rows)


def figure_time(df):
    """Top row: PR-AUC per rolling window. Bottom row: PR-AUC by week of the final 20%
    of the timeline. One column per model, random vs chronological."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    roll = df[(df.exp == "rolling") & (df.dataset == "ieee")].copy()
    roll["split"] = roll.protocol.str.split("_").str[0]
    roll["x"] = roll.protocol.str[-1].astype(int) + 4
    wk = weekly_frame(df).rename(columns={"protocol": "split", "week": "x"})
    models = [m for m in MODELS if (roll.model == m).any() and (wk.model == m).any()]
    fig, axes = plt.subplots(2, len(models), figsize=(7.0, 3.5), sharey=True, squeeze=False)
    rows = [(roll, [4, 5, 6], (3.7, 6.3), "Test month (model trained on the three months before it)"),
            (wk, list(range(1, 7)), (0.6, 6.4), "Week of the final 20% of the timeline")]
    for r, (frame, ticks, xlim, xlabel) in enumerate(rows):
        for c, m in enumerate(models):
            ax = axes[r, c]
            for split, (label, color, marker) in SERIES.items():
                part = frame[(frame.model == m) & (frame.split == split)].groupby("x").pr_auc.mean()
                ax.plot(part.index, part.values, color=color, marker=marker, markersize=4.5,
                        linewidth=2, label=label, markeredgecolor="white", markeredgewidth=0.8)
            if r == 0:
                ax.set_title(MODEL_NAMES[m], fontsize=9, color=INK)
            ax.set_xticks(ticks)
            ax.set_xlim(*xlim)
            _style(ax)
        axes[r, 0].set_ylabel("PR-AUC", fontsize=9, color=INK)
        axes[r, len(models) // 2].set_xlabel(xlabel, fontsize=9, color=INK)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.04, 1, 1), h_pad=1.2)
    fig.savefig(OUT / "fig_time.pdf", bbox_inches="tight")
    plt.close(fig)
    return wk


# ------------------------------------------------------------------ numbers for the prose
SHORT = {"pr_auc": "pr", "roc_auc": "roc", "recall@0.005": "r005", "recall@0.01": "r01",
         "recall@0.05": "r05", "value_recall@0.01": "v01", "value_recall@0.05": "v05",
         "precision@0.005": "p005", "precision@0.01": "p01", "precision@0.05": "p05",
         "f1_at_0.5": "f1"}


def numbers(df, pmatch, boot):
    """Every figure quoted in the text, as \\num{key} macros, so prose cannot drift
    from the stored results."""
    N = {}
    f3, pct = (lambda v: f"{v:.3f}"), (lambda v: f"{100 * v:.0f}")
    main = df[df.exp == "main"]
    for d in available(df, "main"):
        ds = dataset(d)
        cut = int(len(ds) * TEST_START)
        N[f"{d}.n"], N[f"{d}.fraud"] = f"{len(ds):,}", f"{int(ds.y.sum()):,}"
        N[f"{d}.rate"] = f"{100 * ds.y.mean():.2f}"
        N[f"{d}.testfraud"] = f"{int(ds.y[cut:].sum()):,}"
        part = main[main.dataset == d]
        for metric, short in SHORT.items():
            mean = part.groupby(["model", "protocol"])[metric].mean().unstack()
            sd = part.groupby(["model", "protocol"])[metric].std(ddof=1).unstack()
            mean = mean.dropna()
            for m in mean.index:
                r, c = mean.loc[m, "random"], mean.loc[m, "chrono"]
                N[f"{d}.{m}.random.{short}"], N[f"{d}.{m}.chrono.{short}"] = f3(r), f3(c)
                N[f"{d}.{m}.random.{short}.sd"] = f3(sd.loc[m, "random"])
                N[f"{d}.{m}.drop.{short}"] = f3(r - c)
                N[f"{d}.{m}.rel.{short}"] = pct((r - c) / r)
            drop = mean["random"] - mean["chrono"]
            rel = drop / mean["random"]
            N[f"{d}.mean.random.{short}"], N[f"{d}.mean.chrono.{short}"] = (
                f3(mean["random"].mean()), f3(mean["chrono"].mean()))
            N[f"{d}.mean.drop.{short}"], N[f"{d}.mean.rel.{short}"] = f3(drop.mean()), pct(rel.mean())
            N[f"{d}.min.drop.{short}"], N[f"{d}.max.drop.{short}"] = f3(drop.min()), f3(drop.max())
            N[f"{d}.min.rel.{short}"], N[f"{d}.max.rel.{short}"] = pct(rel.min()), pct(rel.max())
            if len(mean) >= 3:
                tau = kendalltau(mean["random"].rank(), mean["chrono"].rank()).statistic
                N[f"{d}.tau.{short}"] = f"{tau:.2f}"
            for proto in ("random", "chrono"):
                N[f"{d}.range.{proto}.{short}"] = f"{mean[proto].max() - mean[proto].min():.2f}"
                N[f"{d}.best.{proto}.{short}"] = MODEL_NAMES[mean[proto].idxmax()]
        for m, g in pmatch[pmatch.dataset == d].groupby("model"):
            N[f"{d}.{m}.matched.pr"] = f3(g.pr_auc.mean())
            if f"{d}.{m}.chrono.pr" in N:
                N[f"{d}.{m}.matcheddrop.pr"] = f3(g.pr_auc.mean() - float(N[f"{d}.{m}.chrono.pr"]))
        pm_d = pmatch[pmatch.dataset == d]
        N[f"{d}.matched.nfraud"] = f"{pm_d.n_fraud.mean():.0f}"
        N[f"{d}.mean.matched.pr"] = f3(pm_d.groupby("model").pr_auc.mean().mean())
        for (dd, m), row in boot.iterrows():
            if dd == d:
                N[f"{d}.{m}.ci.lo"], N[f"{d}.{m}.ci.hi"] = f3(-row.hi), f3(-row.lo)
        prox = proximity(d)
        for k, short in (("exact_duplicate_in_train", "dup"), ("train_fraud_within_1min", "min1"),
                         ("train_fraud_within_10min", "min10")):
            N[f"{d}.prox.{short}.random"] = pct(prox.loc[k, "random"])
            N[f"{d}.prox.{short}.chrono"] = pct(prox.loc[k, "chrono"])
    if "ieee" in available(df, "main"):
        table, _ = entity_overlap()
        for col in table.columns:
            N[f"ieee.key.fraud.{col}"] = pct(table.loc["test_fraud_key_among_train_fraud", col])
            N[f"ieee.key.legit.{col}"] = pct(table.loc["test_legit_key_among_train_fraud", col])
    # 7-day gap
    gap = df[df.exp == "gap"].groupby("model").pr_auc.mean()
    for m, v in gap.items():
        N[f"ieee.{m}.gap7.pr"] = f3(v)
        if f"ieee.{m}.chrono.pr" in N:
            N[f"ieee.{m}.gap7.drop"] = f3(float(N[f"ieee.{m}.chrono.pr"]) - v)
    if len(gap):
        N["ieee.mean.gap7.pr"] = f3(gap.mean())
        drops = [float(N[f"ieee.{m}.gap7.drop"]) for m in gap.index if f"ieee.{m}.gap7.drop" in N]
        N["ieee.min.gap7.drop"], N["ieee.max.gap7.drop"] = f3(min(drops)), f3(max(drops))
    # rolling windows
    roll = df[df.exp == "rolling"].copy()
    if len(roll):
        roll["split"] = roll.protocol.str.split("_").str[0]
        mean = roll.groupby(["model", "split"]).pr_auc.mean().unstack().dropna()
        for m in mean.index:
            N[f"ieee.{m}.roll.random"], N[f"ieee.{m}.roll.chrono"] = (
                f3(mean.loc[m, "random"]), f3(mean.loc[m, "chrono"]))
            N[f"ieee.{m}.roll.drop"] = f3(mean.loc[m, "random"] - mean.loc[m, "chrono"])
        per = roll.groupby(["model", "protocol"]).pr_auc.mean().unstack()
        drops = pd.concat([per[f"random_w{w}"] - per[f"chrono_w{w}"] for w in range(3)]).dropna()
        N["ieee.roll.mindrop"], N["ieee.roll.maxdrop"] = f3(drops.min()), f3(drops.max())
        N["ieee.roll.meandrop"] = f3((mean["random"] - mean["chrono"]).mean())
    # cross-validation
    cv = cv_frame(df)
    if len(cv):
        mean = cv.groupby(["dataset", "model", "scheme"]).pr_auc.mean().unstack().dropna()
        for (d, m), row in mean.iterrows():
            N[f"{d}.{m}.randcv"], N[f"{d}.{m}.blockcv"] = f3(row.randcv), f3(row.blockcv)
            N[f"{d}.{m}.cvdrop"] = f3(row.randcv - row.blockcv)
        for d, g in mean.groupby("dataset"):
            N[f"{d}.mean.cvdrop"] = f3((g.randcv - g.blockcv).mean())
        bb = cv_by_block(df).groupby(["dataset", "block"])[["random_folds", "time_block", "drop"]].mean()
        for (d, b), row in bb.iterrows():
            N[f"{d}.block{b}.rand"], N[f"{d}.block{b}.time"] = f3(row.random_folds), f3(row.time_block)
            N[f"{d}.block{b}.drop"] = f3(row["drop"]) if row["drop"] >= 0 else signed(row["drop"])
        for d, g in bb.groupby("dataset"):
            N[f"{d}.blocks.mindrop"], N[f"{d}.blocks.maxdrop"] = f3(g["drop"].min()), f3(g["drop"].max())
            N[f"{d}.blocks.restdrop"] = f3(g["drop"].sort_values().iloc[:-1].mean())
    # oversampling
    s = df[df.exp == "smote"]
    for d in s.dataset.unique():
        sub = s[(s.dataset == d) & (s.imb == "weight")]
        base = sub if len(sub) else df[(df.exp == "main") & (df.dataset == d) & (df.protocol == "random")]
        arms = {"weight": (base, "pr_auc", "f1_at_0.5"),
                "smote": (s[(s.dataset == d) & (s.imb == "smote")], "pr_auc", "f1_at_0.5"),
                "leakrep": (s[(s.dataset == d) & (s.imb == "smote_leaky")], "reported_pr_auc",
                            "reported_f1_at_0.5"),
                "leakreal": (s[(s.dataset == d) & (s.imb == "smote_leaky")], "pr_auc", "f1_at_0.5")}
        for arm, (part, a, b) in arms.items():
            g = part.groupby("model")[[a, b]].mean()
            for m, row in g.iterrows():
                N[f"{d}.{m}.{arm}.pr"], N[f"{d}.{m}.{arm}.f1"] = f3(row[a]), f3(row[b])
            if len(g):
                N[f"{d}.mean.{arm}.pr"], N[f"{d}.mean.{arm}.f1"] = f3(g[a].mean()), f3(g[b].mean())
                N[f"{d}.min.{arm}.pr"], N[f"{d}.max.{arm}.pr"] = f3(g[a].min()), f3(g[a].max())
    # timestamp as a feature
    tf = df[df.exp == "timefeat"].groupby(["dataset", "model", "protocol"]).pr_auc.mean().unstack()
    for (d, m), row in tf.dropna().iterrows():
        N[f"{d}.{m}.tf.random"], N[f"{d}.{m}.tf.chrono"] = f3(row.random), f3(row.chrono)
    for d, g in tf.dropna().groupby("dataset"):
        N[f"{d}.mean.tf.random"], N[f"{d}.mean.tf.chrono"] = f3(g.random.mean()), f3(g.chrono.mean())
    lines = [r"\makeatletter",
             r"\newcommand{\num}[1]{\@ifundefined{num@#1}{\textbf{??#1??}}{\@nameuse{num@#1}}}"]
    lines += [rf"\@namedef{{num@{k}}}{{{v}}}" for k, v in sorted(N.items())]
    lines.append(r"\makeatother")
    (OUT / "numbers.tex").write_text("\n".join(lines) + "\n")
    return N


def cv_by_block(df, k=5):
    """Split the cross-validation comparison by time block. For block b: PR-AUC of the
    blocked fold that holds out b, against the mean over the random folds of PR-AUC on
    the part of each fold's test set that lies in b."""
    rows = []
    cv = cv_frame(df)
    for (d, m), grp in cv.groupby(["dataset", "model"]):
        if grp.scheme.nunique() < 2 or len(grp) < 2 * k:
            continue
        ds = dataset(d)
        edges = np.linspace(0, len(ds), k + 1).astype(int)
        rand = [np.load(x) for x in grp[grp.scheme == "randcv"].path]
        for b in range(k):
            zb = np.load(grp[grp.protocol == f"blockcv_f{b}"].path.iloc[0])
            aps = []
            for z in rand:
                keep = (z["test_idx"] >= edges[b]) & (z["test_idx"] < edges[b + 1])
                aps.append(fast_ap(ds.y[z["test_idx"][keep]], z["score"][keep]))
            rows.append({"dataset": d, "model": m, "block": b + 1, "random_folds": np.mean(aps),
                         "time_block": fast_ap(ds.y[zb["test_idx"]], zb["score"]),
                         "n_fraud": int(ds.y[zb["test_idx"]].sum())})
    out = pd.DataFrame(rows)
    if len(out):
        out["drop"] = out.random_folds - out.time_block
    return out


def summary(df):
    """Plain-text digest of every number quoted in the prose."""
    lines = []
    main = df[df.exp == "main"]
    pmatch = period_matched(df)
    boot = bootstrap_gap(df).set_index(["dataset", "model"])
    numbers(df, pmatch, boot)
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
                    extra = f" | period-matched random {pmm.mean():.4f}+-{pmm.std(ddof=1):.4f}"
                    if (d, m) in boot.index:
                        lo, hi = boot.loc[(d, m), ["lo", "hi"]]
                        extra += f" | boot CI of gap [{lo:+.4f}, {hi:+.4f}]"
                lines.append(f"{m:5s} random {r:.4f}+-{sd.loc[m, 'random']:.4f} chrono {c:.4f}"
                             f"+-{sd.loc[m, 'chrono']:.4f} gap {c - r:+.4f} "
                             f"rel {100 * (c - r) / r:+.1f}%{extra}")
            both = mean.dropna(subset=["random", "chrono"])
            if len(both) < 3:
                continue
            order_r = both["random"].rank(ascending=False)
            order_c = both["chrono"].rank(ascending=False)
            tau = kendalltau(order_r, order_c).statistic
            lines.append(f"rank random {order_r.astype(int).to_dict()} chrono "
                         f"{order_c.astype(int).to_dict()} kendall tau {tau:.2f}; mean gap "
                         f"{(both['chrono'] - both['random']).mean():+.4f}")
        pm_n = pmatch[pmatch.dataset == d]
        lines.append(f"period-matched test: n={pm_n.n.mean():.0f} n_fraud={pm_n.n_fraud.mean():.1f}")
    for d in available(df, "main"):
        lines.append(f"== proximity of test fraud to training fraud: {d}")
        lines.append(proximity(d).round(4).to_string())
    if "ieee" in available(df, "main"):
        table, note = entity_overlap()
        lines += ["== card-key overlap (ieee)", note, table.round(4).to_string()]
    byblock = cv_by_block(df)
    if len(byblock):
        lines.append("== cv by time block (PR-AUC)")
        lines.append(byblock.round(4).to_string(index=False))
        lines.append(byblock.groupby(["dataset", "block"])[["random_folds", "time_block", "drop",
                                                             "n_fraud"]].mean().round(4).to_string())
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
    table_budget(df)
    table_data([d for d in DATASETS if d in available(df, "main")])
    if (df.exp == "smote").any():
        table_smote(df)
    table_controls(df, period_matched(df))
    if ((df.exp == "rolling") & (df.dataset == "ieee")).any():
        wk = figure_time(df)
        wk.groupby(["model", "split", "x"])[["pr_auc", "n_fraud"]].mean().round(4).to_csv(
            OUT / "weekly.csv")
    print(summary(df))


if __name__ == "__main__":
    main()
