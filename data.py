"""
Data pipeline for Artemis QRC.

Loads S&P 500 daily OHLC data, computes monthly realised volatility,
and builds HAR-style features for the QRC and classical baselines.

Reference: Li, Mukhopadhyay, Bayat, Habibnia (2025), arXiv:2505.13933.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path


def load_sp500_daily(csv_path: str | Path = "sp500_daily.csv") -> pd.DataFrame:
    """Load daily S&P 500 OHLC. Returns DataFrame indexed by date."""
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    # Use Close (Adj Close has many duplicates in early periods)
    df = df[["Close"]].rename(columns={"Close": "close"})
    df = df[df["close"] > 0]
    return df


def load_vix_monthly(csv_path: str | Path = "vix_daily.csv") -> pd.Series:
    """
    Monthly-mean CBOE VIX (1990-2026). VIX is an annualised implied-vol index in
    percentage points; we rescale to a monthly volatility fraction comparable to
    realised volatility: sigma_month = (VIX/100) / sqrt(12).
    """
    v = pd.read_csv(csv_path)
    v["DATE"] = pd.to_datetime(v["DATE"])
    v = v.set_index("DATE").sort_index()
    monthly = v["CLOSE"].groupby(pd.Grouper(freq="ME")).mean()
    return (monthly / 100.0 / np.sqrt(12.0)).rename("vix_vol")


def compute_daily_log_returns(df: pd.DataFrame) -> pd.Series:
    """r_t = log(P_t / P_{t-1})"""
    return np.log(df["close"]).diff().dropna()


def compute_monthly_rv(daily_returns: pd.Series) -> pd.Series:
    """
    Monthly realised volatility from daily log returns.
    RV_m = sqrt( sum_{t in month m} r_t^2 ).
    """
    grouped = (daily_returns ** 2).groupby(pd.Grouper(freq="ME")).sum()
    # Require at least 15 trading days per month
    counts = daily_returns.groupby(pd.Grouper(freq="ME")).count()
    grouped = grouped[counts >= 15]
    return np.sqrt(grouped).rename("rv")


def compute_monthly_rq(daily_returns: pd.Series) -> pd.Series:
    """
    Realised quarticity proxy per Barndorff-Nielsen & Shephard:
        RQ_m = (N_m / 3) * sum_{t in m} r_t^4 ,
    used as the fourth-moment (tail) feature. Returned as RQ^{1/4} to keep it on
    a volatility-comparable scale for encoding.
    """
    r4 = (daily_returns ** 4).groupby(pd.Grouper(freq="ME")).sum()
    counts = daily_returns.groupby(pd.Grouper(freq="ME")).count()
    rq = (counts / 3.0) * r4
    rq = rq[counts >= 15]
    return (rq ** 0.25).rename("rq")


def compute_monthly_market_return(daily_returns: pd.Series) -> pd.Series:
    """Monthly cumulative log return (MKT feature; sign carries the leverage effect)."""
    mkt = daily_returns.groupby(pd.Grouper(freq="ME")).sum()
    counts = daily_returns.groupby(pd.Grouper(freq="ME")).count()
    return mkt[counts >= 15].rename("mkt")


def build_features(
    rv: pd.Series,
    n_input: int = 3,
    rq: pd.Series | None = None,
    mkt: pd.Series | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, list[str]]:
    """
    HAR-style features plus optional macro features.

    Core HAR (Corsi 2009 spirit at monthly cadence):
        lag1   = RV_{t-1}
        mean3  = mean(RV_{t-3:t-1})    weekly-analogue
        mean12 = mean(RV_{t-12:t-1})   monthly/annual-analogue

    Optional macro features (used when n_input > 3):
        rq_lag1  = realised quarticity^{1/4}, lag 1   (tail / fourth moment)
        mkt_lag1 = monthly market log return, lag 1   (leverage effect)
        mean6    = mean(RV_{t-6:t-1})                 medium-horizon memory

    Returns (X, y, index, feature_names).
    """
    df = pd.DataFrame({"rv": rv})
    df["lag1"] = df["rv"].shift(1)
    df["mean3"] = df["rv"].shift(1).rolling(3).mean()
    df["mean12"] = df["rv"].shift(1).rolling(12).mean()
    if rq is not None:
        df["rq_lag1"] = rq.reindex(df.index).shift(1)
    if mkt is not None:
        df["mkt_lag1"] = mkt.reindex(df.index).shift(1)
    df["mean6"] = df["rv"].shift(1).rolling(6).mean()

    # Preferred ordering: core HAR first, then macro features.
    ordered = ["lag1", "mean3", "mean12", "rq_lag1", "mkt_lag1", "mean6"]
    available = [c for c in ordered if c in df.columns]
    feature_cols = available[:n_input]

    df = df.dropna(subset=["rv"] + feature_cols)
    X = df[feature_cols].to_numpy()
    y = df["rv"].to_numpy()
    return X, y, df.index, feature_cols


# Backwards-compatible alias used by earlier scripts.
def build_har_features(rv: pd.Series, n_input: int = 3):
    X, y, idx, _ = build_features(rv, n_input=n_input)
    return X, y, idx


def normalise_features(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Min-max scale to [0, 1] using train statistics, so features can be mapped
    to R_Y(pi * x) rotation angles on the input qubits.
    """
    mn = X_train.min(axis=0)
    mx = X_train.max(axis=0)
    span = np.where(mx - mn > 1e-12, mx - mn, 1.0)
    Xtr = (X_train - mn) / span
    Xte = (X_test - mn) / span
    Xte = np.clip(Xte, 0.0, 1.0)
    return Xtr, Xte, mn, span


def train_test_split_temporal(
    X: np.ndarray,
    y: np.ndarray,
    index: pd.DatetimeIndex,
    test_frac: float = 0.30,
) -> dict:
    """Temporal (non-shuffled) split."""
    n = len(y)
    n_test = int(np.ceil(test_frac * n))
    n_train = n - n_test
    return {
        "X_train": X[:n_train],
        "y_train": y[:n_train],
        "X_test": X[n_train:],
        "y_test": y[n_train:],
        "idx_train": index[:n_train],
        "idx_test": index[n_train:],
        "n_train": n_train,
        "n_test": n_test,
    }


def prepare_dataset(
    csv_path: str | Path = "sp500_daily.csv",
    n_input: int = 3,
    test_frac: float = 0.30,
    start: str | None = None,
    end: str | None = None,
    use_macro: bool = False,
    target: str = "rv",
    vix_csv: str | Path = "vix_daily.csv",
) -> dict:
    """
    End-to-end dataset builder.

    target == "rv"  : monthly realised volatility of the S&P 500 (primary task).
    target == "vix" : monthly volatility from the CBOE VIX (full COVID coverage).

    use_macro adds realised-quarticity and market-return features when n_input > 3.
    """
    if target == "vix":
        rv = load_vix_monthly(vix_csv)
        rq = mkt = None
    else:
        df = load_sp500_daily(csv_path)
        if start is not None:
            df = df.loc[df.index >= pd.Timestamp(start)]
        if end is not None:
            df = df.loc[df.index <= pd.Timestamp(end)]
        rets = compute_daily_log_returns(df)
        rv = compute_monthly_rv(rets)
        rq = compute_monthly_rq(rets) if use_macro else None
        mkt = compute_monthly_market_return(rets) if use_macro else None

    if target == "vix":
        # Trim to a requested window on the monthly series directly.
        if start is not None:
            rv = rv.loc[rv.index >= pd.Timestamp(start)]
        if end is not None:
            rv = rv.loc[rv.index <= pd.Timestamp(end)]

    X, y, idx, feat_names = build_features(rv, n_input=n_input, rq=rq, mkt=mkt)
    split = train_test_split_temporal(X, y, idx, test_frac=test_frac)
    Xtr, Xte, mn, span = normalise_features(split["X_train"], split["X_test"])
    split["X_train_norm"] = Xtr
    split["X_test_norm"] = Xte
    split["feat_mean"] = mn
    split["feat_span"] = span
    split["rv"] = rv
    split["feature_names"] = feat_names
    split["X_all"] = X
    split["y_all"] = y
    split["idx_all"] = idx
    return split


def walk_forward_windows(n: int, min_train: int, step: int = 1):
    """
    Yield (train_end, test_idx) for an expanding-window walk-forward evaluation.
    For each test point t >= min_train, the model trains on [0, t) and predicts t.
    `step` controls stride; step=1 predicts every month.
    """
    t = min_train
    while t < n:
        yield t, list(range(t, min(t + step, n)))
        t += step


if __name__ == "__main__":
    out = prepare_dataset(n_input=3, start="1990-01-01")
    print(f"Train: {out['n_train']} obs   Test: {out['n_test']} obs")
    print(f"Train span: {out['idx_train'][0].date()} -> {out['idx_train'][-1].date()}")
    print(f"Test  span: {out['idx_test'][0].date()} -> {out['idx_test'][-1].date()}")
    print(f"X_train_norm shape: {out['X_train_norm'].shape}, range [{out['X_train_norm'].min():.3f}, {out['X_train_norm'].max():.3f}]")
    print(f"y_train mean: {out['y_train'].mean():.4f}, std: {out['y_train'].std():.4f}")
