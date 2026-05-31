"""
End-to-end Phase 2 prototype experiment for Team Artemis (v2).

Adds, relative to v1:
  - Walk-forward (expanding-window) evaluation: reservoir features computed once,
    readout refit on each expanding window. This is the standard volatility-forecast
    protocol and removes any single-split artefact.
  - Strictly matched baseline: ESN (N=8) with the *same* degree-2 polynomial ridge
    readout as the QRC, so any QRC win is attributable to the quantum reservoir.
  - Regime-conditional metrics: RMSE/QLIKE split by high- vs low-volatility months.
  - Data re-uploading ablation (QR-RU): repeated encoding within a timestep.
  - QR1 at tau/2 (the second ensemble member) reported on its own.
  - Mincer-Zarnowitz R^2 column for full transparency.

Reproducibility: all randomness seeded; re-running reproduces the numbers.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from baselines import EchoStateNetwork, HAR
from data import prepare_dataset, walk_forward_windows
from metrics import diebold_mariano, summarise
from qrc import QRCConfig, QuantumReservoir
from readout import PolynomialRidgeReadout

OUT_DIR = Path("results"); OUT_DIR.mkdir(exist_ok=True)
FIG_DIR = Path("figures"); FIG_DIR.mkdir(exist_ok=True)

DATA_CFG = dict(csv_path="sp500_daily.csv", n_input=3, test_frac=0.30, start="1990-01-01")
READOUT_DEG, READOUT_ALPHA = 2, 1e-3


def walk_forward_predict(feats_all, y_all, min_train, degree=READOUT_DEG, alpha=READOUT_ALPHA):
    """Expanding-window: for each t >= min_train, fit readout on [0,t), predict t."""
    preds = np.zeros(len(y_all) - min_train)
    for i, (t_end, test_idx) in enumerate(walk_forward_windows(len(y_all), min_train, step=1)):
        head = PolynomialRidgeReadout(degree=degree, alpha=alpha).fit(
            feats_all[:t_end], y_all[:t_end])
        preds[i] = head.predict(feats_all[test_idx])[0]
    return preds


def wf_esn(X_all, y_all, min_train, N, poly_degree, seed=42):
    preds = np.zeros(len(y_all) - min_train)
    for i, (t_end, test_idx) in enumerate(walk_forward_windows(len(y_all), min_train, step=1)):
        esn = EchoStateNetwork(n_reservoir=N, spectral_radius=0.95, leak=0.3,
                               ridge_alpha=1e-3, seed=seed, poly_degree=poly_degree)
        esn.fit(X_all[:t_end], y_all[:t_end])
        preds[i] = esn.predict(X_all[test_idx])[0]
    return preds


def wf_har(X_all, y_all, min_train):
    preds = np.zeros(len(y_all) - min_train)
    for i, (t_end, test_idx) in enumerate(walk_forward_windows(len(y_all), min_train, step=1)):
        har = HAR().fit(X_all[:t_end], y_all[:t_end])
        preds[i] = har.predict(X_all[test_idx])[0]
    return preds


def regime_split(y_true, y_pred, hi_threshold):
    hi = y_true >= hi_threshold
    def block(mask):
        if mask.sum() < 3:
            return None
        yt, yp = y_true[mask], y_pred[mask]
        s2t, s2p = np.maximum(yt**2, 1e-8), np.maximum(yp**2, 1e-8)
        r = s2t / s2p
        return {"n": int(mask.sum()),
                "RMSE": float(np.sqrt(np.mean((yt - yp) ** 2))),
                "QLIKE": float(np.mean(r - np.log(r) - 1))}
    return block(hi), block(~hi)


def main():
    t0 = time.time()
    print("=" * 72)
    print("Artemis QRC | Phase 2 Prototype (v2: walk-forward, matched ESN, regimes)")
    print("=" * 72)

    ds = prepare_dataset(**DATA_CFG)
    y_all = ds["y_all"]
    X_all_norm = np.vstack([ds["X_train_norm"], ds["X_test_norm"]])
    min_train = ds["n_train"]
    n_test = len(y_all) - min_train
    y_te = y_all[min_train:]
    idx_te = ds["idx_all"][min_train:]
    print(f"\nData: {len(y_all)} months, walk-forward test = {n_test} months "
          f"({idx_te[0].date()} -> {idx_te[-1].date()})")
    hi_thr = float(np.quantile(y_te, 0.70))
    print(f"High-vol regime threshold (70th pct of test RV) = {hi_thr:.4f}")

    results, predictions = {}, {"truth": y_te}

    def evaluate(name, preds):
        summ = summarise(y_te, preds, name=name)
        hi, lo = regime_split(y_te, preds, hi_thr)
        summ["regime_high"], summ["regime_low"] = hi, lo
        results[name] = summ
        predictions[name] = preds
        hi_rmse = hi["RMSE"] if hi else float("nan")
        print(f"  {name:20s} | RMSE={summ['RMSE']:.5f}  QLIKE={summ['QLIKE']:.4f}  "
              f"MZslope={summ['MZ']['slope']:.3f}  R2={summ['MZ']['r2']:.3f}  hiRMSE={hi_rmse:.5f}")
        return preds

    print("\n[QRC] computing reservoir features once each, then walk-forward readout")
    cfg_t10 = QRCConfig(n_input=3, n_hidden=5, tau=10.0, seed=42)
    cfg_t5  = QRCConfig(n_input=3, n_hidden=5, tau=5.0,  seed=43)
    cfg_ru  = QRCConfig(n_input=3, n_hidden=5, tau=10.0, seed=42, reuploads=2)

    t1 = time.time()
    feats_t10 = QuantumReservoir(cfg_t10).run(X_all_norm)
    feats_t5  = QuantumReservoir(cfg_t5).run(X_all_norm)
    feats_ru  = QuantumReservoir(cfg_ru).run(X_all_norm)
    feats_qr2 = np.concatenate([feats_t10, feats_t5], axis=1)
    print(f"  reservoir evolution done in {time.time()-t1:.1f}s")

    evaluate("QR1 (tau=10)", walk_forward_predict(feats_t10, y_all, min_train))
    evaluate("QR1 (tau=5)",  walk_forward_predict(feats_t5,  y_all, min_train))
    evaluate("QR-RU (reupload)", walk_forward_predict(feats_ru, y_all, min_train))
    evaluate("QR2 (ensemble)", walk_forward_predict(feats_qr2, y_all, min_train))

    print("\n[Baselines] walk-forward")
    evaluate("Persistence", np.concatenate([[y_all[min_train-1]], y_te[:-1]]))
    evaluate("HAR-OLS", wf_har(X_all_norm, y_all, min_train))
    evaluate("ESN-8 (poly, matched)", wf_esn(X_all_norm, y_all, min_train, N=8, poly_degree=2))
    evaluate("ESN-50 (linear)",  wf_esn(X_all_norm, y_all, min_train, N=50,  poly_degree=1))
    evaluate("ESN-100 (linear)", wf_esn(X_all_norm, y_all, min_train, N=100, poly_degree=1))

    print("\n[DM] QR2 (ensemble) vs baselines (squared-error loss)")
    dm_tests = {}
    for name, yhat in predictions.items():
        if name in ("truth", "QR2 (ensemble)"):
            continue
        dm = diebold_mariano(y_te, predictions["QR2 (ensemble)"], yhat, loss="se", h=1)
        dm_tests[name] = dm
        flag = "QR2 better" if dm["DM_HLN"] < 0 else "baseline better"
        print(f"  QR2 vs {name:20s} | DM={dm['DM_HLN']:+.3f}  p={dm['p_value']:.3f}  ({flag})")

    serial = {
        "metadata": {"n_total": len(y_all), "n_test": int(n_test),
                     "test_start": str(idx_te[0].date()), "test_end": str(idx_te[-1].date()),
                     "hi_threshold": hi_thr, "protocol": "expanding walk-forward",
                     "readout": {"degree": READOUT_DEG, "alpha": READOUT_ALPHA},
                     "runtime_s": time.time() - t0},
        "main": {k: {kk: vv for kk, vv in v.items() if kk != "name"} for k, v in results.items()},
        "dm_tests": dm_tests,
    }
    json.dump(serial, open(OUT_DIR / "results.json", "w"), indent=2)
    np.savez(OUT_DIR / "predictions.npz", y_test=y_te,
             idx_test=np.array([str(d) for d in idx_te]),
             **{k.replace(" ","_").replace("(","").replace(")","").replace("=","").replace("+","plus").replace(",","").replace("/","")
                : v for k, v in predictions.items() if k != "truth"})
    print(f"\n[+] results.json + predictions.npz written ({time.time()-t0:.1f}s)")

    make_figures(idx_te, predictions, hi_thr)
    return results


def make_figures(idx_te, predictions, hi_thr):
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.0})

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4), constrained_layout=True)
    ax = axes[0]
    ax.plot(idx_te, predictions["truth"], color="black", label="Realised", linewidth=1.0)
    ax.plot(idx_te, predictions["QR2 (ensemble)"], color="C3", label="QR2 (ours)", linewidth=1.0)
    ax.plot(idx_te, predictions["ESN-100 (linear)"], color="C0", label="ESN-100", linewidth=0.9, linestyle="--", alpha=0.85)
    ax.plot(idx_te, predictions["HAR-OLS"], color="C2", label="HAR", linewidth=0.8, linestyle=":", alpha=0.8)
    ax.axhline(hi_thr, color="gray", linewidth=0.5, linestyle="-.", alpha=0.6)
    ax.set_ylabel("Monthly RV"); ax.set_title("(a) Walk-forward forecast, S&P 500")
    ax.legend(loc="upper left", frameon=False, fontsize=7.5, ncol=2)

    ax = axes[1]
    ax.plot(idx_te, predictions["QR2 (ensemble)"] - predictions["truth"], color="C3", label="QR2 error", linewidth=1.0)
    ax.plot(idx_te, predictions["ESN-100 (linear)"] - predictions["truth"], color="C0", label="ESN error", linewidth=0.9, linestyle="--", alpha=0.85)
    ax.axhline(0, color="gray", linewidth=0.4)
    ax.set_ylabel("Forecast error"); ax.set_title("(b) Forecast errors")
    ax.legend(loc="upper left", frameon=False, fontsize=7.5)
    for a in axes:
        a.tick_params(axis="x", rotation=20, labelsize=7.5); a.tick_params(axis="y", labelsize=7.5)
    fig.savefig(FIG_DIR / "forecast.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "forecast.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[+] figures/forecast.pdf")


if __name__ == "__main__":
    main()
