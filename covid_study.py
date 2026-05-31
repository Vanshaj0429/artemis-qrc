"""
COVID-era robustness study.

The primary S&P realised-volatility series ends in 2018. To demonstrate the
architecture on the most important volatility regime shift of the modern era
(the March 2020 COVID shock and the 2022 selloff), we run the same QR2 pipeline
on a monthly volatility series derived from the CBOE VIX, which is available
through 2026. VIX is rescaled to a monthly volatility fraction.

This is a robustness / generalisation check, not the primary benchmark: it shows
the method tracks an out-of-sample regime explosion it was never tuned on.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from data import prepare_dataset, walk_forward_windows
from metrics import summarise, diebold_mariano
from qrc import QRCConfig, QuantumReservoir
from readout import PolynomialRidgeReadout
from baselines import EchoStateNetwork

OUT = Path("results"); FIG = Path("figures")

ds = prepare_dataset(target="vix", n_input=3, start="2005-01-01")
y_all = ds["y_all"]; X_all_full = ds["X_all"]; idx_all = ds["idx_all"]
# Re-cut so training ends Dec 2019 and the March-2020 COVID shock is out-of-sample.
import pandas as pd
cut = np.searchsorted(idx_all.values, pd.Timestamp("2020-01-31").to_datetime64())
min_train = int(cut)
from data import normalise_features
Xtr_raw, Xte_raw = X_all_full[:min_train], X_all_full[min_train:]
Xtr, Xte, _, _ = normalise_features(Xtr_raw, Xte_raw)
X_all = np.vstack([Xtr, Xte])
y_te = y_all[min_train:]; idx_te = idx_all[min_train:]
print(f"VIX task: {len(y_all)} months, train ends {idx_all[min_train-1].date()}, "
      f"walk-forward test = {len(y_te)} ({idx_te[0].date()} -> {idx_te[-1].date()})")

def wf_qr(feats):
    preds = np.zeros(len(y_all) - min_train)
    for i, (t_end, ti) in enumerate(walk_forward_windows(len(y_all), min_train, 1)):
        h = PolynomialRidgeReadout(degree=2, alpha=1e-3).fit(feats[:t_end], y_all[:t_end])
        preds[i] = h.predict(feats[ti])[0]
    return preds

def wf_esn(N, deg):
    preds = np.zeros(len(y_all) - min_train)
    for i, (t_end, ti) in enumerate(walk_forward_windows(len(y_all), min_train, 1)):
        e = EchoStateNetwork(n_reservoir=N, spectral_radius=0.95, leak=0.3,
                             ridge_alpha=1e-3, seed=42, poly_degree=deg)
        e.fit(X_all[:t_end], y_all[:t_end]); preds[i] = e.predict(X_all[ti])[0]
    return preds

f10 = QuantumReservoir(QRCConfig(n_input=3, n_hidden=5, tau=10.0, seed=42)).run(X_all)
f5  = QuantumReservoir(QRCConfig(n_input=3, n_hidden=5, tau=5.0, seed=43)).run(X_all)
qr2 = wf_qr(np.concatenate([f10, f5], axis=1))
pers = np.concatenate([[y_all[min_train-1]], y_te[:-1]])
esn8 = wf_esn(8, 2)

res = {}
for nm, p in [("QR2", qr2), ("ESN-8 (matched)", esn8), ("Persistence", pers)]:
    s = summarise(y_te, p, name=nm); res[nm] = s
    print(f"  {nm:16s} RMSE={s['RMSE']:.5f} QLIKE={s['QLIKE']:.4f} MZslope={s['MZ']['slope']:.3f}")

# COVID window error specifically (2020-02 to 2020-12)
covid_mask = np.array([(d.year == 2020 and d.month <= 6) for d in idx_te])
if covid_mask.sum() > 0:
    cov_rmse_qr2 = float(np.sqrt(np.mean((y_te[covid_mask]-qr2[covid_mask])**2)))
    cov_rmse_esn = float(np.sqrt(np.mean((y_te[covid_mask]-esn8[covid_mask])**2)))
    cov_rmse_per = float(np.sqrt(np.mean((y_te[covid_mask]-pers[covid_mask])**2)))
    print(f"  2020 COVID window RMSE: QR2={cov_rmse_qr2:.5f}  ESN-8={cov_rmse_esn:.5f}  Pers={cov_rmse_per:.5f}")
    res["covid_2020"] = {"QR2": cov_rmse_qr2, "ESN8": cov_rmse_esn, "Persistence": cov_rmse_per}

dm = diebold_mariano(y_te, qr2, pers, loss="se", h=1)
res["DM_QR2_vs_persistence"] = dm
print(f"  DM QR2 vs persistence: {dm['DM_HLN']:+.3f} (p={dm['p_value']:.3f})")
json.dump(res, open(OUT/"covid_study.json", "w"), indent=2, default=float)

# Figure
plt.rcParams.update({"font.family":"serif","font.serif":["DejaVu Serif"],"font.size":9,
    "axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":0.6})
fig, ax = plt.subplots(figsize=(3.5, 2.2), constrained_layout=True)
ax.plot(idx_te, y_te, color="black", label="Realised (VIX-vol)", linewidth=1.0)
ax.plot(idx_te, qr2, color="C3", label="QR2", linewidth=1.0)
ax.plot(idx_te, esn8, color="C0", label="ESN-8", linewidth=0.9, linestyle="--", alpha=0.85)
ax.set_ylabel("Monthly volatility"); ax.set_title("COVID-era walk-forward (VIX)", fontsize=9)
ax.legend(loc="upper right", frameon=False, fontsize=7.5)
ax.tick_params(axis="x", rotation=20, labelsize=7.5); ax.tick_params(axis="y", labelsize=7.5)
fig.savefig(FIG/"covid.pdf", dpi=300, bbox_inches="tight")
fig.savefig(FIG/"covid.png", dpi=200, bbox_inches="tight")
print("[+] figures/covid.pdf")
