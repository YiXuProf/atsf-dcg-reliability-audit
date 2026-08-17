"""Test-time sensor-degradation robustness evaluation (opt-in ``--degradation``).

The seven degradations below are applied to the **already z-scored** test
windows (``DatasetBundle.X_test``), i.e. *post-normalization*, so all
magnitudes are expressed in units of per-channel standard deviation of the
test split (for z-scored data this is ~1.0 by construction).  This keeps the
degradation strength comparable across datasets and channels and mirrors how
a deployed model would see corrupted, pre-processed sensor streams.

Degradations
------------
1. ``gaussian_noise_snr20`` / ``gaussian_noise_snr10`` : additive white
   Gaussian noise, per-channel sigma set so the channel-wise SNR (w.r.t. the
   per-channel signal power of the test split) is 20 dB / 10 dB.
2. ``drift`` : linear ramp 0 -> 0.5 channel-std across the window, per
   channel, with a random sign per channel.
3. ``bias`` : constant +0.5 channel-std offset on 10 % of channels, chosen at
   random per window (fixed seed).
4. ``stuck`` : 5 % of channels frozen at their window-start value, chosen at
   random per window (fixed seed).
5. ``dropout`` : 10 % of time samples zeroed at random positions (same
   positions across all channels of a window; fixed seed).
6. ``downsample`` : temporal stride-2 subsample, then linear interpolation
   back to the original window length T.

Determinism
-----------
All randomness uses a dedicated ``numpy.random.RandomState`` seeded per
degradation as ``run_seed + SEED_OFFSET + index`` (see
``train._degradation_eval``), so the whole evaluation is fully deterministic
given the run seed and does **not** touch the global numpy/torch RNGs (the
default pipeline output is unchanged when the feature is off).
"""

from __future__ import annotations

import numpy as np

#: fixed evaluation order (also the column order of degradation_table.csv)
DEGRADATION_NAMES = (
    "gaussian_noise_snr20",
    "gaussian_noise_snr10",
    "drift",
    "bias",
    "stuck",
    "dropout",
    "downsample",
)

#: base offset added to the run seed for degradation RNGs
SEED_OFFSET = 10_000


def _channel_std(X: np.ndarray) -> np.ndarray:
    """Per-channel std over batch and time, shape (N,)."""
    return X.std(axis=(0, 2))


def _gaussian_noise(X: np.ndarray, rng: np.random.RandomState,
                    snr_db: float) -> np.ndarray:
    """Additive N(0, sigma^2); sigma per channel set for ``snr_db`` SNR
    w.r.t. the per-channel signal power of the test split."""
    power = (X ** 2).mean(axis=(0, 2))  # (N,)
    sigma = np.sqrt(power / (10.0 ** (snr_db / 10.0)))
    return X + rng.normal(0.0, 1.0, size=X.shape) * sigma[None, :, None]


def _drift(X: np.ndarray, rng: np.random.RandomState,
           magnitude: float = 0.5) -> np.ndarray:
    """Linear ramp 0 -> ``magnitude`` channel-std across the window, per
    channel, random sign per channel."""
    _, N, T = X.shape
    std = _channel_std(X)  # (N,)
    signs = rng.choice((-1.0, 1.0), size=N)
    ramp = np.linspace(0.0, magnitude, T)
    return X + ramp[None, None, :] * (signs * std)[None, :, None]


def _random_subset_cols(rng: np.random.RandomState, n_rows: int, n_cols: int,
                        frac: float) -> np.ndarray:
    """(n_rows, k) column indices: exactly k = round(frac * n_cols) distinct
    columns chosen uniformly per row (deterministic under ``rng``)."""
    k = max(1, int(round(frac * n_cols)))
    order = rng.random((n_rows, n_cols)).argsort(axis=1, kind="stable")
    return order[:, :k]


def _bias(X: np.ndarray, rng: np.random.RandomState, frac: float = 0.1,
          magnitude: float = 0.5) -> np.ndarray:
    """Constant +``magnitude`` channel-std offset on ``frac`` of channels,
    randomly chosen per window."""
    B, N, _ = X.shape
    std = _channel_std(X)  # (N,)
    ch = _random_subset_cols(rng, B, N, frac)  # (B, k)
    out = X.copy()
    rows = np.arange(B)[:, None]
    out[rows, ch, :] += (magnitude * std[ch])[..., None]
    return out


def _stuck(X: np.ndarray, rng: np.random.RandomState,
           frac: float = 0.05) -> np.ndarray:
    """``frac`` of channels frozen at their window-start value, randomly
    chosen per window."""
    B, N, _ = X.shape
    ch = _random_subset_cols(rng, B, N, frac)  # (B, k)
    out = X.copy()
    rows = np.arange(B)[:, None]
    out[rows, ch, :] = out[rows, ch, :1]
    return out


def _dropout(X: np.ndarray, rng: np.random.RandomState,
             frac: float = 0.1) -> np.ndarray:
    """``frac`` of time samples zeroed at random positions per window (same
    positions across all channels of the window)."""
    B, _, T = X.shape
    idx = _random_subset_cols(rng, B, T, frac)  # (B, k)
    out = X.copy()
    out[np.arange(B)[:, None], :, idx] = 0.0
    return out


def _downsample(X: np.ndarray, rng: np.random.RandomState | None = None,
                stride: int = 2) -> np.ndarray:
    """Temporal stride-``stride`` subsample, then linear interpolation back to
    the original length T (vectorised equivalent of per-channel np.interp on
    the uniform subsample grid).  No randomness; ``rng`` accepted for a
    uniform signature."""
    _, _, T = X.shape
    sub = X[:, :, ::stride]
    n_sub = sub.shape[-1]
    pos = np.arange(T, dtype=np.float64) / stride  # fractional index into sub
    i0 = np.floor(pos).astype(np.int64).clip(0, n_sub - 1)
    i1 = (i0 + 1).clip(max=n_sub - 1)
    w = (pos - i0)[None, None, :]
    return (1.0 - w) * sub[:, :, i0] + w * sub[:, :, i1]


def apply_degradation(X: np.ndarray, name: str,
                      rng: np.random.RandomState) -> np.ndarray:
    """Return a degraded copy of the z-scored windows ``X`` (B, N, T).

    ``rng`` must be a dedicated RandomState (one per degradation); ``X`` is
    not modified.  Output shape always equals input shape."""
    X = np.asarray(X, dtype=np.float64)
    if name == "gaussian_noise_snr20":
        return _gaussian_noise(X, rng, 20.0)
    if name == "gaussian_noise_snr10":
        return _gaussian_noise(X, rng, 10.0)
    if name == "drift":
        return _drift(X, rng)
    if name == "bias":
        return _bias(X, rng)
    if name == "stuck":
        return _stuck(X, rng)
    if name == "dropout":
        return _dropout(X, rng)
    if name == "downsample":
        return _downsample(X, rng)
    raise ValueError(f"unknown degradation {name!r}; "
                     f"expected one of {list(DEGRADATION_NAMES)}")
