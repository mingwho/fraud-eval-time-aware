"""Split protocols. All functions return (train, val, test) arrays of row positions
into a time-sorted dataset."""
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

DAY = 86400.0


def random_split(y, seed, test=0.2, val=0.1, pool=None):
    """Stratified random split. `test` and `val` are fractions of the pool."""
    idx = np.arange(len(y)) if pool is None else np.asarray(pool)
    rest, te = train_test_split(idx, test_size=test, stratify=y[idx], random_state=seed)
    tr, va = train_test_split(
        rest, test_size=val / (1 - test), stratify=y[rest], random_state=seed
    )
    return np.sort(tr), np.sort(va), np.sort(te)


def chrono_split(n, train=0.7, val=0.1):
    """First 70% of transactions for training, next 10% validation, last 20% test."""
    a, b = int(n * train), int(n * (train + val))
    return np.arange(a), np.arange(a, b), np.arange(b, n)


def chrono_gap_split(t, gap_days=7.0, train=0.7, val=0.1):
    """Chronological split whose test set is unchanged, but whose development data
    ends `gap_days` before the first test transaction (label delay)."""
    n = len(t)
    test_start = int(n * (train + val))
    dev = np.flatnonzero(t < t[test_start] - gap_days * DAY)
    a = int(len(dev) * train / (train + val))
    return dev[:a], dev[a:], np.arange(test_start, n)


def time_blocks(t, n_blocks):
    """Assign each row to one of `n_blocks` equal-width time blocks."""
    edges = np.linspace(t.min(), t.max(), n_blocks + 1)
    return np.clip(np.searchsorted(edges, t, side="right") - 1, 0, n_blocks - 1)


def rolling_windows(t, y, seed, n_blocks=6, train_blocks=3, val=0.125):
    """Yield (window, protocol, train, val, test). Each window pools `train_blocks`
    + 1 consecutive blocks. The chronological variant trains on the first
    `train_blocks` and tests on the next one; the random variant draws a test set
    of the same size at random from the same pool."""
    block = time_blocks(t, n_blocks)
    for w in range(n_blocks - train_blocks):
        dev = np.flatnonzero((block >= w) & (block < w + train_blocks))
        te = np.flatnonzero(block == w + train_blocks)
        a = int(len(dev) * (1 - val))
        yield w, "chrono", dev[:a], dev[a:], te
        pool = np.concatenate([dev, te])
        frac_te = len(te) / len(pool)
        yield (w, "random") + random_split(
            y, seed, test=frac_te, val=val * (1 - frac_te), pool=pool
        )


def cv_folds(y, seed, k=5, val=0.125):
    """Yield (fold, protocol, train, val, test) for two k-fold schemes that test every
    transaction exactly once. "blockcv" uses k contiguous time blocks as folds;
    "randcv" uses stratified random folds."""
    n = len(y)
    edges = np.linspace(0, n, k + 1).astype(int)
    for f in range(k):
        te = np.arange(edges[f], edges[f + 1])
        dev = np.concatenate([np.arange(edges[f]), np.arange(edges[f + 1], n)])
        a = int(len(dev) * (1 - val))
        yield f, "blockcv", dev[:a], dev[a:], te
    folds = StratifiedKFold(k, shuffle=True, random_state=seed).split(np.zeros(n), y)
    for f, (dev, te) in enumerate(folds):
        tr, va = train_test_split(dev, test_size=val, stratify=y[dev], random_state=seed)
        yield f, "randcv", np.sort(tr), np.sort(va), te
