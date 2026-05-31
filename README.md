# Team Artemis | GIC 2026 Phase 2 | Quantum Reservoir Computing for Realised Volatility

End-to-end prototype and technical paper for Track A (Financial Volatility) of the
qBraid / MITRE / JonesTrading Global Industry Challenge 2026.

We forecast one-month-ahead realised volatility (RV) of the S&P 500 with a
transverse-field Ising quantum reservoir, an angle-encoded input register, a
partial-trace memory channel, and a polynomial-ridge readout. The single
reservoir (QR1) and a two-timescale ensemble (QR2) are benchmarked against
persistence, HAR-OLS, and Echo State Networks.

## Headline results (101-month walk-forward out-of-sample test set)

Expanding-window walk-forward; readout refit before every one-month-ahead forecast.

| Model | RMSE | QLIKE | MZ slope | MZ R2 | hi-RMSE | DM vs QR2 (p) |
|---|---:|---:|---:|---:|---:|---|
| **QR2 (ensemble, n=8)** | **0.01738** | 0.415 | 0.745 | 0.208 | 0.02615 | — |
| QR1 (tau=5, n=8) | 0.01748 | 0.406 | **0.799** | 0.194 | 0.02583 | -0.24 (0.81) |
| HAR-OLS | 0.01740 | 0.407 | 0.666 | 0.253 | 0.02567 | -0.03 (0.98) |
| ESN-8 (poly, matched) | 0.01740 | **0.388** | 0.677 | 0.241 | **0.02484** | -0.03 (0.98) |
| QR-RU (re-upload, n=8) | 0.01786 | 0.395 | 0.668 | 0.213 | 0.02541 | -0.89 (0.38) |
| ESN-100 (linear) | 0.01893 | 0.473 | 0.526 | 0.202 | 0.02763 | -2.15 (0.03) |
| Persistence (lag-1) | 0.01942 | 0.688 | 0.490 | 0.248 | 0.02939 | -1.85 (0.07) |

At n=8 the QRC is **genuinely competitive** with strong classical
baselines. QR2 has the lowest RMSE of any model, but the gaps to HAR and the
strictly matched ESN-8 (same degree-2 polynomial readout) are not significant
(DM p ~ 0.98). QR2 significantly beats only the oversized linear ESN-100 (p=0.03)
and persistence. The case for the quantum route rests on three signals:
(i) walk-forward RMSE decreases monotonically with size (0.0187 -> 0.0182 -> 0.0172
for n = 6, 8, 10); (ii) at matched readout the QRC has higher memory capacity than
an ESN (2.92 vs 2.31 linear; 3.04 vs 2.38 nonlinear); (iii) data re-uploading
improves QLIKE and explained variance. A VIX-derived COVID-era study (out-of-sample
through 2026) shows the pipeline generalises and stays competitive with matched
baselines.

Earlier single-split numbers showed a much larger QRC win; that was an artefact of
the single split and an unmatched (polynomial vs linear) readout. The walk-forward
protocol plus the strictly matched ESN baseline corrected it.

## Repository layout

```
data.py            Load S&P 500 daily OHLC, compute monthly RV, HAR features, scaling, split
qrc.py             Pauli ops, random Ising H, U=exp(-iH tau), partial trace,
                   fast diagonal Z observables, gate-by-gate encoding, QuantumReservoir, ensemble
readout.py         Degree-2 polynomial expansion + ridge regression (the only trained part)
baselines.py       Persistence, HAR-OLS, EchoStateNetwork (optional polynomial readout, matched to QRC)
metrics.py         RMSE, MAE, QLIKE (Patton 2011), Mincer-Zarnowitz, Diebold-Mariano (HLN)
experiment.py      Walk-forward runner: QR1/QR2/QR-RU, matched ESN, regime metrics, DM tests, figures
scaling_sweep.py   Walk-forward scaling sweep (n=6,8,10 x 2 seeds) -> results/scaling.json
analysis.py        Memory capacity (QRC vs ESN, matched readout), MC vs tau, RV timescale
covid_study.py     VIX-derived COVID-era out-of-sample robustness study
paper.tex          3-page Phase 2 technical paper (11pt Times New Roman); paper.pdf is compiled
sp500_daily.csv    S&P 500 daily close, 1950-2018
vix_daily.csv      CBOE VIX daily, 1990-2026 (COVID-era study)
figures/           forecast.pdf, analysis.pdf, covid.pdf (used by the paper)
results/           results.json, scaling.json, predictions.npz, memory_capacity.json, covid_study.json
```

## Reproduce

```bash
pip install -r requirements.txt
python experiment.py        # ~15 s: walk-forward comparison, DM tests, figures
python analysis.py          # ~1 min: memory capacity + RV timescale
python scaling_sweep.py     # ~5 min: walk-forward scaling (n=10 is the slow part)
python covid_study.py       # ~30 s: VIX COVID-era robustness
```

All randomness is seeded. Re-running reproduces the numbers above exactly.

## Data note

The prototype uses public S&P 500 daily closing prices (1950-2018, GitHub
historical mirror). Phase 3 will refresh the series through 2025 via Yahoo
Finance and cross-check realised variance against the Oxford-Man library.

## Phase 2 design choices (mapped to the rubric)

- **Architecture**: fully-connected TFI Ising reservoir; angle encoding;
  partial-trace memory; degree-2 polynomial ridge readout; QR2 two-timescale ensemble.
- **Justification**: Hilbert-space dimensionality (expressivity), partial-trace
  dissipation (echo-state property, Ahmed 2025), tunable memory-nonlinearity
  tradeoff (Cindrak 2026).
- **Baselines and metrics**: ESN is the primary baseline; RMSE, QLIKE,
  Mincer-Zarnowitz, Diebold-Mariano.
- **Platform**: simulator-first (Aer statevector / density-matrix), tensor-network
  for n >= 20, one QPU validation run with ZNE in Phase 3.

## Disclosure

Code scaffolding and the paper layout were produced with LLM assistance. The QRC
formulation, experiment design, results, and analysis are the team's own work.
