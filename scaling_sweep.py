"""Walk-forward scaling sweep: QR1 vs n, reservoir computed once per config."""
import json, time
from pathlib import Path
import numpy as np
from data import prepare_dataset, walk_forward_windows
from metrics import summarise
from qrc import QRCConfig, QuantumReservoir
from readout import PolynomialRidgeReadout

OUT = Path("results"); OUT.mkdir(exist_ok=True)
CONFIGS = {6: (3, 3), 8: (3, 5), 10: (3, 7)}
SEEDS = [42, 7]  # 2 seeds keeps it tractable at n=10

ds = prepare_dataset(csv_path="sp500_daily.csv", n_input=3, test_frac=0.30, start="1990-01-01")
y_all = ds["y_all"]; X_all = np.vstack([ds["X_train_norm"], ds["X_test_norm"]])
min_train = ds["n_train"]; y_te = y_all[min_train:]

def wf(feats):
    preds = np.zeros(len(y_all) - min_train)
    for i, (t_end, ti) in enumerate(walk_forward_windows(len(y_all), min_train, 1)):
        h = PolynomialRidgeReadout(degree=2, alpha=1e-3).fit(feats[:t_end], y_all[:t_end])
        preds[i] = h.predict(feats[ti])[0]
    return preds

scan = {}
if (OUT/"scaling.json").exists():
    scan = json.load(open(OUT/"scaling.json"))
for n_total, (ni, nh) in CONFIGS.items():
    key = f"n={n_total}"
    if key in scan and scan[key].get("protocol") == "walk-forward":
        print(f"{key} done"); continue
    rmses, qlks = [], []; t0 = time.time()
    for s in SEEDS:
        feats = QuantumReservoir(QRCConfig(n_input=ni, n_hidden=nh, tau=10.0, seed=s)).run(X_all)
        p = wf(feats); m = summarise(y_te, p, name=key)
        rmses.append(m["RMSE"]); qlks.append(m["QLIKE"])
    scan[key] = {"n_total": n_total, "protocol": "walk-forward",
                 "RMSE_mean": float(np.mean(rmses)), "RMSE_std": float(np.std(rmses)),
                 "QLIKE_mean": float(np.mean(qlks)), "QLIKE_std": float(np.std(qlks)),
                 "RMSE_per_seed": rmses, "QLIKE_per_seed": qlks, "wall_s": time.time()-t0}
    json.dump(scan, open(OUT/"scaling.json", "w"), indent=2)
    r = scan[key]
    print(f"{key}: RMSE={r['RMSE_mean']:.5f}+/-{r['RMSE_std']:.5f} QLIKE={r['QLIKE_mean']:.4f} ({r['wall_s']:.0f}s)")
print("DONE")
