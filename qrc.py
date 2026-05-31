"""
Quantum Reservoir Computing core.

Architecture (after Li et al. 2025, arXiv:2505.13933):

    - n_input qubits encode features via R_Y(pi * x_i) rotations.
    - n_hidden qubits carry temporal memory.
    - Reservoir Hamiltonian on all n = n_input + n_hidden qubits:
          H = sum_{i<j} J_ij X_i X_j  +  v sum_i Z_i
      with J_ij drawn iid from U(0, 1) and held fixed (seeded).
    - Evolution unitary U = exp(-i H tau).
    - At each time step:
          1. trace out the n_input qubits of the previous state;
          2. tensor a fresh |0...0> on the n_input register with the kept hidden
             reduced state;
          3. apply the feature-encoding R_Y rotations on the input register;
          4. evolve under U;
          5. measure <Z_i> for all n qubits.

This realises a non-Markovian map on the hidden register: information about
past inputs is retained between steps, the input register is refreshed
deterministically, and the encoding is gradient-free.

We work directly in density matrices so that an ensemble (QR2) and noise
channels (Phase 3) plug in without re-engineering. State sizes are
4^n complex numbers; n <= 12 is comfortable on a laptop.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Pauli operators and n-qubit embedding
# ---------------------------------------------------------------------------

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def _kron_list(ops: list[np.ndarray]) -> np.ndarray:
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


def single_qubit_op(op: np.ndarray, qubit: int, n: int) -> np.ndarray:
    """Embed a 2x2 operator acting on `qubit` (0-indexed, leftmost) into n qubits."""
    ops = [I2] * n
    ops[qubit] = op
    return _kron_list(ops)


def two_qubit_op(op_a: np.ndarray, qubit_a: int, op_b: np.ndarray, qubit_b: int, n: int) -> np.ndarray:
    ops = [I2] * n
    ops[qubit_a] = op_a
    ops[qubit_b] = op_b
    return _kron_list(ops)


# ---------------------------------------------------------------------------
# Reservoir Hamiltonian and evolution
# ---------------------------------------------------------------------------


def random_ising_hamiltonian(n: int, v: float, seed: int = 42, j_max: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Fully-connected transverse-field Ising Hamiltonian on n qubits.

        H = sum_{i<j} J_ij X_i X_j  +  v sum_i Z_i,    J_ij ~ U(0, j_max).

    Returns (H, J_matrix). H is a 2^n x 2^n Hermitian matrix.
    """
    rng = np.random.default_rng(seed)
    J = np.zeros((n, n))
    iu = np.triu_indices(n, k=1)
    J[iu] = rng.uniform(0.0, j_max, size=len(iu[0]))
    J = J + J.T

    H = np.zeros((2 ** n, 2 ** n), dtype=complex)
    for i in range(n):
        for j in range(i + 1, n):
            H += J[i, j] * two_qubit_op(X, i, X, j, n)
        H += v * single_qubit_op(Z, i, n)
    return H, J


def evolution_unitary(H: np.ndarray, tau: float) -> np.ndarray:
    """U = exp(-i H tau) via eigendecomposition of a Hermitian H."""
    # Hermitian eigendecomposition is faster and more stable than scipy.linalg.expm here.
    eigvals, eigvecs = np.linalg.eigh(H)
    phases = np.exp(-1j * eigvals * tau)
    return (eigvecs * phases) @ eigvecs.conj().T


# ---------------------------------------------------------------------------
# Density matrix utilities
# ---------------------------------------------------------------------------


def partial_trace(rho: np.ndarray, traced_qubits: list[int], n: int) -> np.ndarray:
    """
    Trace out qubits `traced_qubits` from an n-qubit density matrix rho
    (rho is 2^n x 2^n). Qubit indexing: leftmost = 0.
    """
    keep = [q for q in range(n) if q not in traced_qubits]
    n_keep = len(keep)
    n_trace = n - n_keep
    # Reshape rho into (2,)*2n tensor: first n indices row, next n indices column
    rho_t = rho.reshape([2] * (2 * n))
    # Permute axes so traced qubits come last on both row and column sides
    perm = keep + traced_qubits + [n + k for k in keep] + [n + t for t in traced_qubits]
    rho_t = np.transpose(rho_t, perm)
    # Now collapse to (D_keep, D_trace, D_keep, D_trace)
    D_keep = 2 ** n_keep
    D_trace = 2 ** n_trace
    rho_t = rho_t.reshape(D_keep, D_trace, D_keep, D_trace)
    # Trace over the trace indices (axis 1 and axis 3)
    return np.einsum("ikjk->ij", rho_t)


def ry_matrix(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2.0), np.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=complex)


def apply_single_qubit_gate(rho: np.ndarray, gate: np.ndarray, qubit: int, n: int) -> np.ndarray:
    """
    Apply a 2x2 unitary on `qubit` to a 2^n x 2^n density matrix rho:
        rho -> (I .. U .. I) rho (I .. U_dag .. I)
    via tensor reshape (O(D^2) instead of O(D^3) of a full Kronecker multiply).
    """
    D = 2 ** n
    # Reshape rho to (2,)*n on rows and (2,)*n on columns.
    rho_t = rho.reshape([2] * (2 * n))
    # Move row index `qubit` to position 0.
    rho_t = np.moveaxis(rho_t, qubit, 0)
    # Multiply on the left by `gate` along that axis.
    rho_t = np.tensordot(gate, rho_t, axes=([1], [0]))
    rho_t = np.moveaxis(rho_t, 0, qubit)
    # Move column index `n + qubit` to position 0.
    rho_t = np.moveaxis(rho_t, n + qubit, 0)
    rho_t = np.tensordot(gate.conj(), rho_t, axes=([1], [0]))
    rho_t = np.moveaxis(rho_t, 0, n + qubit)
    return rho_t.reshape(D, D)


def apply_encoding_inplace(rho: np.ndarray, features: np.ndarray, n_input: int, n: int,
                            scale: float = np.pi) -> np.ndarray:
    """Apply R_Y(scale * x_i) on input qubit i, for i = 0..n_input-1."""
    for i in range(n_input):
        rho = apply_single_qubit_gate(rho, ry_matrix(scale * float(features[i])), i, n)
    return rho


def initial_state(n: int) -> np.ndarray:
    """Pure |0...0> density matrix."""
    rho = np.zeros((2 ** n, 2 ** n), dtype=complex)
    rho[0, 0] = 1.0
    return rho


# ---------------------------------------------------------------------------
# Z observables - precompute for speed
# ---------------------------------------------------------------------------


def precompute_z_diagonals(n: int) -> np.ndarray:
    """
    Z_i is diagonal in the computational basis, with entries (-1)^{bit_i(k)}.
    Returns an (n, 2^n) array of these diagonals so <Z_i> = sum_k Zdiag[i, k] * rho_kk.
    """
    D = 2 ** n
    basis = np.arange(D)
    diags = np.empty((n, D), dtype=float)
    for i in range(n):
        # Leftmost qubit = 0; little-endian bit index for qubit i is (n-1-i).
        bits = (basis >> (n - 1 - i)) & 1
        diags[i] = 1.0 - 2.0 * bits  # +1 if bit 0, -1 if bit 1
    return diags


def z_expectations_fast(rho: np.ndarray, z_diags: np.ndarray) -> np.ndarray:
    """O(D) computation of all <Z_i>: just a weighted sum over diag(rho)."""
    diag_rho = np.real(np.diag(rho))
    return z_diags @ diag_rho


# ---------------------------------------------------------------------------
# QRC reservoir wrapper
# ---------------------------------------------------------------------------


@dataclass
class QRCConfig:
    n_input: int
    n_hidden: int
    tau: float
    v: float = 1.0
    j_max: float = 1.0
    seed: int = 42
    reuploads: int = 1  # data re-uploading depth within a single timestep

    @property
    def n_total(self) -> int:
        return self.n_input + self.n_hidden


class QuantumReservoir:
    """
    Stateful QRC: call .step(x) for each input vector x and get back the
    Z-expectation feature vector.
    """

    def __init__(self, cfg: QRCConfig):
        self.cfg = cfg
        self.n = cfg.n_total
        self.H, self.J = random_ising_hamiltonian(self.n, cfg.v, cfg.seed, cfg.j_max)
        self.U = evolution_unitary(self.H, cfg.tau)
        self.U_dag = self.U.conj().T
        # For re-uploading we evolve for tau / reuploads between encodings, so the
        # total evolution per timestep is still tau regardless of depth.
        self.reuploads = max(1, cfg.reuploads)
        if self.reuploads > 1:
            self.U_sub = evolution_unitary(self.H, cfg.tau / self.reuploads)
            self.U_sub_dag = self.U_sub.conj().T
        self.z_diags = precompute_z_diagonals(self.n)
        self._input_zero = initial_state(cfg.n_input)
        self.reset()

    def reset(self) -> None:
        self.rho = initial_state(self.n)

    def step(self, x: np.ndarray) -> np.ndarray:
        n_in = self.cfg.n_input
        n = self.n

        # 1. Trace out the input qubits (qubits 0 .. n_input-1).
        rho_hidden = partial_trace(self.rho, list(range(n_in)), n)
        # 2. Reset input register to |0> and tensor with kept hidden state.
        rho_full = np.kron(self._input_zero, rho_hidden)

        if self.reuploads == 1:
            # 3. Encode features (gate by gate, O(D^2) each).
            rho_full = apply_encoding_inplace(rho_full, x, n_in, n)
            # 4. Evolve for the full tau.
            rho_full = self.U @ rho_full @ self.U_dag
        else:
            # Data re-uploading: interleave (encode, evolve tau/k) k times.
            # Increases the Fourier richness of the input-to-feature map at fixed
            # total evolution time tau (Schuld et al. data-reuploading argument).
            for _ in range(self.reuploads):
                rho_full = apply_encoding_inplace(rho_full, x, n_in, n)
                rho_full = self.U_sub @ rho_full @ self.U_sub_dag

        # 5. Read observables (diagonal-only, O(D)).
        feats = z_expectations_fast(rho_full, self.z_diags)
        self.rho = rho_full
        return feats

    def run(self, X: np.ndarray, warmup: int = 0) -> np.ndarray:
        """Run on an entire sequence X of shape (T, n_input). Returns (T, n_total)."""
        out = np.zeros((len(X), self.n))
        for t, x in enumerate(X):
            out[t] = self.step(x)
        if warmup > 0:
            out = out[warmup:]
        return out


def run_ensemble(X: np.ndarray, configs: list[QRCConfig]) -> np.ndarray:
    """
    QR2-style ensemble: run multiple reservoirs (e.g. different tau) and
    concatenate their Z observables column-wise.
    Returns (T, sum_k n_k).
    """
    blocks = []
    for cfg in configs:
        res = QuantumReservoir(cfg)
        blocks.append(res.run(X))
    return np.concatenate(blocks, axis=1)


if __name__ == "__main__":
    # Smoke test: 8-qubit reservoir, 3 input qubits, run for 20 timesteps.
    cfg = QRCConfig(n_input=3, n_hidden=5, tau=10.0, v=1.0, seed=42)
    res = QuantumReservoir(cfg)
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, size=(20, 3))
    feats = res.run(X)
    print("Feature matrix shape:", feats.shape)
    print("First row:", np.round(feats[0], 3))
    print("Last row :", np.round(feats[-1], 3))
    # Check density matrix stays a valid state.
    print("Final trace:", np.real(np.trace(res.rho)))
    print("Final purity:", np.real(np.trace(res.rho @ res.rho)))
