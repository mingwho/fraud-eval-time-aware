"""Dataset loading. Every dataset is returned with rows sorted by transaction time,
so integer row positions double as a chronological order."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


@dataclass
class Dataset:
    name: str
    X: np.ndarray  # float32, NaN marks a missing value
    y: np.ndarray  # int8, 1 = fraud
    t: np.ndarray  # seconds since the first day of the dataset
    amount: np.ndarray  # transaction amount in the dataset's currency
    features: list
    amount_col: int  # column of X holding the amount

    def __len__(self):
        return len(self.y)


def _finish(name, df, label, time, amount, with_time):
    df = df.sort_values(time, kind="stable").reset_index(drop=True)
    y = df[label].to_numpy(np.int8)
    t = df[time].to_numpy(np.float64)
    amt = df[amount].to_numpy(np.float64)
    drop = [label] if with_time else [label, time]
    feats = df.drop(columns=drop)
    for c in feats.columns:
        if not pd.api.types.is_numeric_dtype(feats[c]):
            # Lexicographic codes: uses no label or frequency information.
            codes = feats[c].astype("category").cat.codes.astype(np.float32)
            feats[c] = codes.where(codes >= 0, np.nan)
    X = feats.to_numpy(np.float32)
    names = list(feats.columns)
    return Dataset(name, X, y, t, amt, names, names.index(amount))


def load_ulb(with_time=False):
    df = pd.read_csv(DATA / "ulb" / "creditcard.csv")
    return _finish("ulb", df, "Class", "Time", "Amount", with_time)


def load_ieee(with_time=False):
    cache = DATA / "ieee" / "merged.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
    else:
        tx = pd.read_csv(DATA / "ieee" / "train_transaction.csv")
        idn = pd.read_csv(DATA / "ieee" / "train_identity.csv")
        df = tx.merge(idn, on="TransactionID", how="left").drop(columns="TransactionID")
        df.to_parquet(cache)
    return _finish("ieee", df, "isFraud", "TransactionDT", "TransactionAmt", with_time)


def load(name, with_time=False):
    return {"ulb": load_ulb, "ieee": load_ieee}[name](with_time)
