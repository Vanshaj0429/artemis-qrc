"""
Classical baselines for the realised volatility forecasting task.

We implement:

    - Persistence: y_hat_t = y_{t-1}.
    - HAR (Heterogeneous Autoregressive, Corsi 2009 adapted to monthly RV):
        RV_t = b0 + b1 * RV_{t-1} + b2 * mean(RV_{t-3:t-1}) + b3 * mean(RV_{t-12:t-1}).
      Implemented as plain OLS on the HAR-style features already in `data.py`.
    - ESN (Echo State Network, Jaeger 2001): random recurrent reservoir with
      spectral radius and leak rate hyperparameters, trained linear readout.
      Documented as the most important baseline by the GIC challenge brief:
      "beating ESN is what would justify QRC."

References:
    Corsi, F. (2009). A simple approximate long-memory model of realized
        volatility. Journal of Financial Econometrics 7(2), 174-196.
    Jaeger, H. (2001). The "echo state" approach to analysing and training
        recurrent neural networks. GMD Report 148.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge, LinearRegression


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class Persistence:
    """y_hat_t = y_{t-1}. Equivalent to lag-1 RV."""

    def fit(self, X, y):
        return self

    def predict(self, X, y_prev=None):
        # Assumes X[:, 0] is lag1 RV (true for the build_har_features output).
        return X[:, 0]


# ---------------------------------------------------------------------------
# HAR
# ---------------------------------------------------------------------------


class HAR:
    """HAR on the lag1 / mean3 / mean12 features already built in `data.py`."""

    def __init__(self, ridge_alpha: float = 0.0):
        self.model = Ridge(alpha=ridge_alpha) if ridge_alpha > 0 else LinearRegression()

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


# ---------------------------------------------------------------------------
# ESN
# ---------------------------------------------------------------------------


class EchoStateNetwork:
    """
    Classical reservoir computer with a tanh-activated recurrent layer.

    State update:
        h_t = (1 - leak) * h_{t-1} + leak * tanh( W_in @ x_t + W_res @ h_{t-1} )
    W_res is rescaled to a target spectral radius.

    Readout: ridge regression on h_t, optionally with a degree-`poly_degree`
    polynomial expansion so the readout can be matched exactly to the QRC.
    """

    def __init__(
        self,
        n_reservoir: int = 100,
        spectral_radius: float = 0.95,
        leak: float = 0.3,
        input_scale: float = 1.0,
        ridge_alpha: float = 1e-3,
        seed: int = 42,
        sparsity: float = 0.1,
        poly_degree: int = 1,
    ):
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.leak = leak
        self.input_scale = input_scale
        self.ridge_alpha = ridge_alpha
        self.seed = seed
        self.sparsity = sparsity
        self.poly_degree = poly_degree

    def _build(self, n_in: int) -> None:
        rng = np.random.default_rng(self.seed)
        self.W_in = self.input_scale * rng.uniform(-1.0, 1.0, size=(self.n_reservoir, n_in))
        W_res = rng.uniform(-1.0, 1.0, size=(self.n_reservoir, self.n_reservoir))
        mask = rng.uniform(0, 1, size=W_res.shape) < self.sparsity
        W_res = W_res * mask
        # Rescale to target spectral radius.
        eigs = np.linalg.eigvals(W_res)
        rho = float(np.max(np.abs(eigs)))
        if rho > 1e-12:
            W_res = W_res * (self.spectral_radius / rho)
        self.W_res = W_res

    def _run(self, X: np.ndarray) -> np.ndarray:
        T = len(X)
        h = np.zeros(self.n_reservoir)
        H = np.zeros((T, self.n_reservoir))
        for t in range(T):
            pre = self.W_in @ X[t] + self.W_res @ h
            h = (1.0 - self.leak) * h + self.leak * np.tanh(pre)
            H[t] = h
        return H

    def states_over(self, X: np.ndarray, continue_state: bool = False) -> np.ndarray:
        """Return reservoir states over X. If continue_state, start from stored final state."""
        T = len(X)
        h = (self._final_state.copy() if (continue_state and hasattr(self, "_final_state"))
             else np.zeros(self.n_reservoir))
        H = np.zeros((T, self.n_reservoir))
        for t in range(T):
            pre = self.W_in @ X[t] + self.W_res @ h
            h = (1.0 - self.leak) * h + self.leak * np.tanh(pre)
            H[t] = h
        return H

    def _readout_features(self, H: np.ndarray) -> np.ndarray:
        if self.poly_degree > 1:
            from sklearn.preprocessing import PolynomialFeatures
            H = PolynomialFeatures(degree=self.poly_degree, include_bias=False).fit_transform(H)
        return np.hstack([H, np.ones((len(H), 1))])

    def fit(self, X, y):
        self._build(n_in=X.shape[1])
        H = self.states_over(X)
        Hb = self._readout_features(H)
        self.ridge = Ridge(alpha=self.ridge_alpha, fit_intercept=False)
        self.ridge.fit(Hb, y)
        self._final_state = H[-1].copy()
        return self

    def predict(self, X):
        H = self.states_over(X, continue_state=True)
        Hb = self._readout_features(H)
        return self.ridge.predict(Hb)
