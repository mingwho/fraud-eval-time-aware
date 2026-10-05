"""Model definitions with a uniform interface.

Each model exposes
    suggest(trial) -> hyperparameters
    fit(params, Xtr, ytr, seed, Xval=None, yval=None, n_iter=None)
        -> (fitted model, number of boosting rounds / epochs or None)
    predict(model, X) -> fraud scores in [0, 1]

With a validation set, iterative models early-stop on validation PR-AUC; without
one they train for exactly `n_iter` rounds (used when refitting on train+val).
With `weighted`, the positive-class weight is tuned as (n_neg / n_pos) ** alpha,
alpha in [0, 1]; it is off (weight 1) when the training set was oversampled.
Heavy libraries are imported lazily so that each process loads one OpenMP runtime.
"""
import os

import numpy as np
from sklearn.metrics import average_precision_score

THREADS = int(os.environ.get("N_THREADS", os.cpu_count()))


class DensePrep:
    """Median imputation, log1p of the amount, standardisation, clipping.
    Fitted on training rows only."""

    def __init__(self, amount_col):
        self.amount_col = amount_col

    def _log_amount(self, X):
        X = X.copy()
        X[:, self.amount_col] = np.log1p(np.maximum(X[:, self.amount_col], 0))
        return X

    def fit(self, X):
        X = self._log_amount(X)
        with np.errstate(all="ignore"):
            self.median = np.nan_to_num(np.nanmedian(X, axis=0))
            self.mean = np.nan_to_num(np.nanmean(X, axis=0))
            self.std = np.nan_to_num(np.nanstd(X, axis=0))
        self.std[self.std < 1e-8] = 1.0
        return self

    def transform(self, X):
        X = self._log_amount(X)
        X = np.where(np.isnan(X), self.median, X)
        return np.clip((X - self.mean) / self.std, -10, 10).astype(np.float32)


def _alpha(trial, weighted):
    """Class weighting: the positive class is weighted by (n_neg / n_pos) ** alpha."""
    return {"alpha": trial.suggest_float("alpha", 0.0, 1.0)} if weighted else {}


def _split(params, y):
    """Separate the positive-class weight from the model's own hyperparameters."""
    params = dict(params)
    ratio = (y == 0).sum() / max((y == 1).sum(), 1)
    return float(ratio ** params.pop("alpha", 0.0)), params


# ---------------------------------------------------------------- logistic regression
class LR:
    dense = True

    @staticmethod
    def suggest(trial, weighted):
        return {"C": trial.suggest_float("C", 1e-4, 1e2, log=True), **_alpha(trial, weighted)}

    @staticmethod
    def fit(params, Xtr, ytr, seed, Xval=None, yval=None, n_iter=None):
        from sklearn.linear_model import LogisticRegression

        w, params = _split(params, ytr)
        m = LogisticRegression(C=params["C"], class_weight={0: 1.0, 1: w}, max_iter=300)
        return m.fit(Xtr, ytr), None

    @staticmethod
    def predict(m, X):
        return m.predict_proba(X)[:, 1]


# ---------------------------------------------------------------- random forest
class RF:
    dense = False

    @staticmethod
    def suggest(trial, weighted):
        return {
            "max_depth": trial.suggest_int("max_depth", 4, 24),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50, log=True),
            "max_features": trial.suggest_float("max_features", 0.03, 0.3),
            **_alpha(trial, weighted),
        }

    @staticmethod
    def fit(params, Xtr, ytr, seed, Xval=None, yval=None, n_iter=None):
        from sklearn.ensemble import RandomForestClassifier

        w, params = _split(params, ytr)
        m = RandomForestClassifier(
            n_estimators=100,
            class_weight={0: 1.0, 1: w},
            max_samples=min(1.0, 100_000 / len(ytr)),
            n_jobs=THREADS,
            random_state=seed,
            **params,
        )
        return m.fit(Xtr, ytr), None

    @staticmethod
    def predict(m, X):
        return m.predict_proba(X)[:, 1]


# ---------------------------------------------------------------- XGBoost
class XGB:
    dense = False
    MAX_ROUNDS = 2000

    @staticmethod
    def suggest(trial, weighted):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_float("min_child_weight", 1, 50, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10, log=True),
            **_alpha(trial, weighted),
        }

    @staticmethod
    def fit(params, Xtr, ytr, seed, Xval=None, yval=None, n_iter=None):
        import xgboost as xgb

        w, params = _split(params, ytr)
        common = dict(
            tree_method="hist",
            scale_pos_weight=w,
            n_jobs=THREADS,
            random_state=seed,
            **params,
        )
        if Xval is None:
            m = xgb.XGBClassifier(n_estimators=n_iter, **common)
            return m.fit(Xtr, ytr), n_iter
        m = xgb.XGBClassifier(
            n_estimators=XGB.MAX_ROUNDS, early_stopping_rounds=50, eval_metric="aucpr", **common
        )
        m.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
        return m, m.best_iteration + 1

    @staticmethod
    def predict(m, X):
        return m.predict_proba(X)[:, 1]


# ---------------------------------------------------------------- LightGBM
class LGBM:
    dense = False
    MAX_ROUNDS = 2000

    @staticmethod
    def suggest(trial, weighted):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 200, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10, log=True),
            **_alpha(trial, weighted),
        }

    @staticmethod
    def fit(params, Xtr, ytr, seed, Xval=None, yval=None, n_iter=None):
        import lightgbm as lgb

        w, params = _split(params, ytr)
        common = dict(
            subsample_freq=1,
            scale_pos_weight=w,
            n_jobs=THREADS,
            random_state=seed,
            verbose=-1,
            **params,
        )
        if Xval is None:
            m = lgb.LGBMClassifier(n_estimators=n_iter, **common)
            return m.fit(Xtr, ytr), n_iter
        m = lgb.LGBMClassifier(n_estimators=LGBM.MAX_ROUNDS, **common)
        m.fit(
            Xtr,
            ytr,
            eval_set=[(Xval, yval)],
            eval_metric="average_precision",
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        return m, max(m.best_iteration_, 1)

    @staticmethod
    def predict(m, X):
        return m.predict_proba(X)[:, 1]


# ---------------------------------------------------------------- MLP
class MLP:
    dense = True
    MAX_EPOCHS = 30
    PATIENCE = 4
    BATCH = 1024

    @staticmethod
    def suggest(trial, weighted):
        return {
            "width": trial.suggest_categorical("width", [64, 128, 256]),
            "depth": trial.suggest_int("depth", 2, 3),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            **_alpha(trial, weighted),
        }

    @staticmethod
    def fit(params, Xtr, ytr, seed, Xval=None, yval=None, n_iter=None):
        import torch
        from torch import nn

        w, params = _split(params, ytr)
        torch.set_num_threads(THREADS)
        torch.manual_seed(seed)
        layers, d = [], Xtr.shape[1]
        for _ in range(params["depth"]):
            layers += [nn.Linear(d, params["width"]), nn.ReLU(), nn.Dropout(params["dropout"])]
            d = params["width"]
        net = nn.Sequential(*layers, nn.Linear(d, 1))
        opt = torch.optim.AdamW(
            net.parameters(), lr=params["lr"], weight_decay=params["weight_decay"]
        )
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(w))
        X = torch.from_numpy(np.ascontiguousarray(Xtr))
        y = torch.from_numpy(ytr.astype(np.float32))
        gen = torch.Generator().manual_seed(seed)
        best, best_epoch, best_state = -1.0, 0, None
        epochs = MLP.MAX_EPOCHS if Xval is not None else n_iter
        for epoch in range(1, epochs + 1):
            net.train()
            perm = torch.randperm(len(y), generator=gen)
            for i in range(0, len(y), MLP.BATCH):
                b = perm[i : i + MLP.BATCH]
                opt.zero_grad()
                loss_fn(net(X[b]).squeeze(1), y[b]).backward()
                opt.step()
            if Xval is not None:
                ap = average_precision_score(yval, MLP.predict(net, Xval))
                if ap > best:
                    best, best_epoch = ap, epoch
                    best_state = {k: v.clone() for k, v in net.state_dict().items()}
                elif epoch - best_epoch >= MLP.PATIENCE:
                    break
        if Xval is None:
            return net, n_iter
        net.load_state_dict(best_state)
        return net, best_epoch

    @staticmethod
    def predict(net, X):
        import torch

        net.eval()
        with torch.no_grad():
            out = [
                torch.sigmoid(net(torch.from_numpy(np.ascontiguousarray(X[i : i + 65536]))))
                for i in range(0, len(X), 65536)
            ]
        return torch.cat(out).squeeze(1).numpy()


MODELS = {"lr": LR, "rf": RF, "xgb": XGB, "lgbm": LGBM, "mlp": MLP}
