"""
Forecast evaluation metrics for realised volatility forecasting.

Implements the three Track A metrics specified by the GIC 2026 challenge brief:

    - RMSE (standard L2 regression metric)
    - QLIKE (Patton 2011 asymmetric loss for variance forecasts)
    - Mincer-Zarnowitz regression (Mincer & Zarnowitz 1969).

Also implements the Diebold-Mariano test of equal predictive accuracy
(Diebold & Mariano 1995, Harvey-Leybourne-Newbold 1997 small-sample correction).

References:
    Patton, A. J. (2011). Volatility forecast comparison using imperfect
        volatility proxies. Journal of Econometrics 160, 246-256.
    Mincer, J., Zarnowitz, V. (1969). The Evaluation of Economic Forecasts.
        NBER.
    Diebold, F. X., Mariano, R. S. (1995). Comparing predictive accuracy.
        Journal of Business & Economic Statistics 13, 253-263.
    Harvey, D., Leybourne, S., Newbold, P. (1997). Testing the equality of
        prediction mean squared errors. Int. J. Forecasting 13, 281-291.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def qlike(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """
    QLIKE on *variances*. We feed in volatilities and square them internally
    so that the loss is computed on sigma^2 as in Patton (2011):

        QLIKE = mean( sigma2_true / sigma2_pred  -  log(sigma2_true / sigma2_pred) - 1 )
    """
    s2_true = np.maximum(y_true ** 2, eps)
    s2_pred = np.maximum(y_pred ** 2, eps)
    r = s2_true / s2_pred
    return float(np.mean(r - np.log(r) - 1.0))


def mincer_zarnowitz(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Regress y_true on a constant + y_pred and test the joint hypothesis
    (intercept = 0, slope = 1) using a Wald-style test on the OLS coefficients.

    Returns dict with intercept, slope, R^2, and a p-value for the joint
    hypothesis (approximate F test).
    """
    X = y_pred.reshape(-1, 1)
    reg = LinearRegression().fit(X, y_true)
    yhat = reg.predict(X)
    n = len(y_true)
    p = 2  # intercept + slope
    resid = y_true - yhat
    sigma2 = np.sum(resid ** 2) / (n - p)

    # Build the design matrix with constant column.
    Xd = np.column_stack([np.ones(n), y_pred])
    XtX_inv = np.linalg.inv(Xd.T @ Xd)
    cov_beta = sigma2 * XtX_inv

    beta = np.array([float(reg.intercept_), float(reg.coef_[0])])
    R = np.eye(2)
    r = np.array([0.0, 1.0])
    diff = R @ beta - r
    wald_num = diff @ np.linalg.inv(R @ cov_beta @ R.T) @ diff
    F_stat = wald_num / 2.0  # two restrictions
    pval = 1.0 - stats.f.cdf(F_stat, 2, n - p)

    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "intercept": float(reg.intercept_),
        "slope": float(reg.coef_[0]),
        "r2": r2,
        "F": float(F_stat),
        "p_joint": float(pval),
    }


def diebold_mariano(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    loss: str = "se",
    h: int = 1,
) -> dict:
    """
    Diebold-Mariano test on loss differential between forecasts A and B with
    Harvey-Leybourne-Newbold small-sample correction. Two-sided.

    Negative DM statistic means model A has lower loss (A is better).
    """
    if loss == "se":
        d = (y_true - y_pred_a) ** 2 - (y_true - y_pred_b) ** 2
    elif loss == "ae":
        d = np.abs(y_true - y_pred_a) - np.abs(y_true - y_pred_b)
    elif loss == "qlike":
        eps = 1e-8
        s2t = np.maximum(y_true ** 2, eps)
        la = np.maximum(y_pred_a ** 2, eps)
        lb = np.maximum(y_pred_b ** 2, eps)
        ra, rb = s2t / la, s2t / lb
        d = (ra - np.log(ra) - 1.0) - (rb - np.log(rb) - 1.0)
    else:
        raise ValueError(f"Unknown loss: {loss}")

    T = len(d)
    d_mean = float(np.mean(d))
    # Newey-West style long-run variance up to lag h-1.
    gamma0 = float(np.mean((d - d_mean) ** 2))
    gammas = []
    for k in range(1, h):
        gk = float(np.mean((d[k:] - d_mean) * (d[:-k] - d_mean)))
        gammas.append(gk)
    var_d = (gamma0 + 2.0 * sum(gammas)) / T
    if var_d <= 0:
        var_d = gamma0 / T
    dm = d_mean / np.sqrt(var_d)
    # HLN small-sample correction.
    correction = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    dm_hln = dm * correction
    pval = 2.0 * (1.0 - stats.t.cdf(abs(dm_hln), df=T - 1))
    return {"DM": float(dm), "DM_HLN": float(dm_hln), "p_value": float(pval), "mean_loss_diff": d_mean}


def summarise(y_true: np.ndarray, y_pred: np.ndarray, name: str = "model") -> dict:
    return {
        "name": name,
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "QLIKE": qlike(y_true, y_pred),
        "MZ": mincer_zarnowitz(y_true, y_pred),
    }
