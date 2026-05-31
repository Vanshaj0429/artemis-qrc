"""
Readout layer for the QRC.

Polynomial feature expansion (degree 2 by default) followed by ridge regression
on the reservoir's Z-expectation features. This is the only trainable part of
the QRC system.

References:
    - Fujii & Nakajima, Phys. Rev. Applied 8, 024030 (2017).
    - Li et al., arXiv:2505.13933 (2025).
    - Zhu et al., Phys. Rev. Research 7, 023290 (2025) - polynomial readout
      substantially reduces NRMSE.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures


class PolynomialRidgeReadout:
    def __init__(self, degree: int = 2, alpha: float = 1e-3, include_bias: bool = True):
        self.degree = degree
        self.alpha = alpha
        self.poly = PolynomialFeatures(degree=degree, include_bias=include_bias, interaction_only=False)
        self.ridge = Ridge(alpha=alpha, fit_intercept=False)

    def fit(self, features: np.ndarray, y: np.ndarray) -> "PolynomialRidgeReadout":
        Phi = self.poly.fit_transform(features)
        self.ridge.fit(Phi, y)
        self.feature_dim_ = Phi.shape[1]
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        Phi = self.poly.transform(features)
        return self.ridge.predict(Phi)
