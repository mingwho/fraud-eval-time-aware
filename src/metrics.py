"""Evaluation metrics, including alert-budget metrics."""
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

BUDGETS = (0.005, 0.01)


def budget_metrics(y, score, amount, budget):
    """Flag the top `budget` share of transactions by score and report the share of
    fraud cases (recall), of fraud value (value recall) and the precision."""
    k = max(1, int(np.ceil(budget * len(y))))
    top = np.argsort(-score, kind="stable")[:k]
    pos = y == 1
    fraud_value = amount[pos].sum()
    return {
        "recall": y[top].sum() / max(pos.sum(), 1),
        "value_recall": amount[top][y[top] == 1].sum() / fraud_value if fraud_value > 0 else np.nan,
        "precision": y[top].mean(),
    }


def evaluate(y, score, amount):
    out = {
        "n": len(y),
        "n_fraud": int(y.sum()),
        "prevalence": float(y.mean()),
        "pr_auc": average_precision_score(y, score),
        "roc_auc": roc_auc_score(y, score),
        "f1_at_0.5": f1_score(y, score >= 0.5, zero_division=0),
    }
    for b in BUDGETS:
        for k, v in budget_metrics(y, score, amount, b).items():
            out[f"{k}@{b:g}"] = float(v)
    return out
