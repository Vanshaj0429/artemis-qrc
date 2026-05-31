"""
Reservoir analysis: linear memory capacity (MC), a simple nonlinear capacity
proxy, and the empirical autocorrelation timescale of realised volatility.

Linear memory capacity (Jaeger 2001; for QRC see Fujii-Nakajima 2017,
Goetting et al. 2023): drive the reservoir with i.i.d. uniform scalar input
u(t), then for each delay k fit a linear readout from the reservoir state to
u(t-k). The capacity at delay k is the coefficient of determination of that
fit, and MC = sum_k MC_k. This quantifies how much past input the reservoir
linearly retains, i.e. it turns the qualitative claim of "fading memory" into
a number that is directly comparable between the QRC and a classical ESN.

A quadratic-memory proxy repeats the procedure with target u(t-k)^2 to probe
nonlinear memory, the resource a linear ESN cannot supply on its own.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge

from qrc import QRCConfig, QuantumReservoir
from baselines import EchoStateNetwork


from sklearn.preprocessing import PolynomialFeatures


def _poly_expand(states: np.ndarray, degree: int) -> np.ndarray:
    if degree <= 1:
        return states
    return PolynomialFeatures(degree=degree, include_bias=False).fit_transform(states)


def _capacity_from_states(states: np.ndarray, u: np.ndarray, max_delay: int,
                          washout: int = 50, nonlinear: bool = False,
                          alpha: float = 1e-6, poly_degree: int = 1) -> tuple[float, np.ndarray]:
    """
    Given a reservoir state trajectory `states` (T, F) driven by scalar input
    `u` (T,), compute summed capacity over delays 1..max_delay. If poly_degree>1
    the readout acts on the degree-`poly_degree` polynomial expansion of the
    states, matching the forecasting readout so the comparison is fair.
    """
    T = len(u)
    feats = _poly_expand(states, poly_degree)
    caps = np.zeros(max_delay)
    Xs = feats[washout:]
    for k in range(1, max_delay + 1):
        target = u[washout - k: T - k]
        target = target[: len(Xs)]
        tgt = target ** 2 if nonlinear else target
        if len(tgt) < 5:
            break
        reg = Ridge(alpha=alpha, fit_intercept=True).fit(Xs[: len(tgt)], tgt)
        pred = reg.predict(Xs[: len(tgt)])
        var = np.var(tgt)
        mse = np.mean((tgt - pred) ** 2)
        caps[k - 1] = max(0.0, 1.0 - mse / var) if var > 1e-12 else 0.0
    return float(np.sum(caps)), caps


def qrc_memory_capacity(cfg: QRCConfig, max_delay: int = 20, n_steps: int = 600,
                        seed: int = 0, poly_degree: int = 2) -> dict:
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 1.0, size=n_steps)
    res = QuantumReservoir(cfg)
    states = res.run(np.repeat(u[:, None], cfg.n_input, axis=1))
    mc_lin, caps_lin = _capacity_from_states(states, u, max_delay, nonlinear=False, poly_degree=poly_degree)
    mc_nl, caps_nl = _capacity_from_states(states, u, max_delay, nonlinear=True, poly_degree=poly_degree)
    return {"MC_linear": mc_lin, "MC_nonlinear": mc_nl,
            "n_readout": res.n, "poly_degree": poly_degree,
            "caps_linear": caps_lin.tolist(), "caps_nonlinear": caps_nl.tolist()}


def esn_memory_capacity(n_reservoir: int = 100, max_delay: int = 20,
                        n_steps: int = 600, seed: int = 0,
                        spectral_radius: float = 0.95, leak: float = 0.3,
                        poly_degree: int = 1) -> dict:
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 1.0, size=n_steps)
    esn = EchoStateNetwork(n_reservoir=n_reservoir, spectral_radius=spectral_radius,
                           leak=leak, seed=seed)
    esn._build(n_in=1)
    states = esn._run(u[:, None])
    mc_lin, caps_lin = _capacity_from_states(states, u, max_delay, nonlinear=False, poly_degree=poly_degree)
    mc_nl, caps_nl = _capacity_from_states(states, u, max_delay, nonlinear=True, poly_degree=poly_degree)
    return {"MC_linear": mc_lin, "MC_nonlinear": mc_nl,
            "n_readout": n_reservoir, "poly_degree": poly_degree,
            "caps_linear": caps_lin.tolist(), "caps_nonlinear": caps_nl.tolist()}


def rv_acf_timescale(rv, max_lag: int = 24) -> dict:
    """
    Autocorrelation function of realised volatility and its integrated timescale
    tau_RV = 1 + 2 * sum_{k>=1} rho_k (truncated at first non-positive rho).
    Gives the memory horizon (in months) the reservoir should match.
    """
    x = np.asarray(rv, dtype=float)
    x = x - x.mean()
    var = np.dot(x, x)
    acf = [1.0]
    for k in range(1, max_lag + 1):
        acf.append(float(np.dot(x[:-k], x[k:]) / var))
    acf = np.array(acf)
    tau = 1.0
    for k in range(1, max_lag + 1):
        if acf[k] <= 0:
            break
        tau += 2.0 * acf[k]
    # also report lag at which acf drops below 1/e
    below = np.where(acf < np.exp(-1))[0]
    e_fold = int(below[0]) if len(below) else max_lag
    return {"acf": acf.tolist(), "integrated_timescale": tau, "e_fold_lag": e_fold}


if __name__ == "__main__":
    import json
    from data import prepare_dataset

    out = {}
    # Memory capacity at the readout the forecaster actually uses (degree-2 poly).
    # Fair comparison: QRC n=8 (8 observables) vs ESN N=8 (8 states), both degree 2.
    print("Memory capacity at matched readout (degree-2 polynomial):")
    cfg8 = QRCConfig(n_input=3, n_hidden=5, tau=10.0, seed=42)
    q = qrc_memory_capacity(cfg8, max_delay=15, n_steps=500, poly_degree=2)
    e8 = esn_memory_capacity(n_reservoir=8, max_delay=15, n_steps=500, poly_degree=2)
    out["qrc_n8_deg2"] = q
    out["esn_n8_deg2"] = e8
    print(f"  QRC n=8  | MC_lin={q['MC_linear']:.2f}  MC_nl={q['MC_nonlinear']:.2f}")
    print(f"  ESN N=8  | MC_lin={e8['MC_linear']:.2f}  MC_nl={e8['MC_nonlinear']:.2f}")

    # MC vs tau for the QRC (n=8), degree-2 readout.
    print("\nMemory capacity vs tau (QRC n=8, degree-2):")
    mc_by_tau = {}
    for tau in [2.0, 5.0, 10.0, 20.0, 40.0]:
        cfg = QRCConfig(n_input=3, n_hidden=5, tau=tau, seed=42)
        r = qrc_memory_capacity(cfg, max_delay=15, n_steps=500, poly_degree=2)
        mc_by_tau[str(tau)] = r
        print(f"  tau={tau:5.1f} | MC_lin={r['MC_linear']:.2f}  MC_nl={r['MC_nonlinear']:.2f}")
    out["qrc_mc_by_tau"] = mc_by_tau

    # RV autocorrelation timescale.
    ds = prepare_dataset(n_input=3, start="1990-01-01")
    ts = rv_acf_timescale(ds["y_all"], max_lag=24)
    out["rv_timescale"] = ts
    print(f"\nRV integrated timescale = {ts['integrated_timescale']:.2f} months, "
          f"e-fold lag = {ts['e_fold_lag']} months")

    json.dump(out, open("results/memory_capacity.json", "w"), indent=2)
    print("Wrote results/memory_capacity.json")
