"""Shared utilities and the experiment configuration contract (SPEC.md).

Everything random flows through ``set_seed``. ``ExpConfig`` is the single
configuration object consumed by ``model.ATSFDCG`` and ``train.train_one``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed python/numpy/torch RNGs and force deterministic cuDNN behaviour."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class ExpConfig:
    """One experimental configuration (see SPEC.md, general conventions)."""

    name: str
    use_temporal: bool = True
    use_spectral: bool = True
    # "adaptive" | "concat" | "none" | "fixed_global" | "fixed_class"
    fusion: str = "adaptive"
    # "dynamic" | "static" | "none" | "sparsemax" | "entmax" | "lstm"
    gating: str = "dynamic"
    spectral_frontend: str = "fft"  # "fft" | "stft" | "sinc"
    r1_balanced: bool = False  # R1: auxiliary heads + OGM gradient modulation
    r2_load_balanced: bool = False  # R2: per-channel/per-timestep alpha + MoE balance loss
    r2_temperature: float = 1.0  # R2: temperature tau of alpha generation
    r2_lambda_balance: float = 0.01  # R2: weight of L_balance in the total loss
    r2_gumbel: bool = False  # R2 variant: two-expert Gumbel-softmax hard routing
    # fusion == "fixed_global" control: constant scalar alpha (both branches)
    fusion_fixed_alpha: float | None = None
    # fusion == "fixed_class" ORACLE control: per-class alpha, length n_classes,
    # indexed by the TRUE label (train and eval alike; diagnostic upper bound)
    fusion_fixed_class_alpha: tuple | None = None
    window: int = 128
    stride: int = 64
    batch_size: int = 64
    lr: float = 1e-3
    epochs: int = 100
    patience: int = 10
    lambda_aux: float = 0.5
    seed: int = 42
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")


def accuracy_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """Top-1 accuracy and macro-F1 as floats."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    acc = float((y_true == y_pred).mean())
    classes = np.unique(np.concatenate([y_true, y_pred]))
    f1s = []
    for c in classes:
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        prec = tp / (tp + fp) if tp + fp > 0 else 0.0
        rec = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0)
    return acc, float(np.mean(f1s))
