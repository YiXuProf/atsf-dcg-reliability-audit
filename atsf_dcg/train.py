"""Training / evaluation loop for one ExpConfig (SPEC.md §train.py, integrator).

Shapes follow the project convention: windows ``(B, N, T)`` -> logits
``(B, K)``; diagnostic tensors ``alpha (B,1,T')`` / ``G (B,C,T')``.

``train_one(cfg, bundle, log_path=None, arch="atsf")``:
- seeds everything via ``utils.set_seed(cfg.seed)``;
- trains ``model.ATSFDCG`` (arch="atsf", default) or
  ``model_tsnet.TSFTimesNet`` (arch="tsnet"; same forward/cache/param-group
  contract) with Adam(cfg.lr) + CrossEntropy, early stopping
  on validation accuracy (``cfg.patience``), best-weight restore, then test;
- remedy R1 (``cfg.r1_balanced``): total loss
  ``L_fused + lambda_aux * (coef_t * L_t + coef_f * L_f)`` where
  ``(coef_t, coef_f)`` come from ``remedies.OGMBalancer`` fed with the
  auxiliary-head accuracies measured on the training batches of the previous
  epoch (initial coefficients 1.0/1.0);
- remedy R2 (``cfg.r2_load_balanced``): adds
  ``cfg.r2_lambda_balance * model.balance_loss()`` to the loss and records
  the per-epoch mean as ``"L_balance"`` in the epoch log;
- after each ``backward()`` records rho = ``grad_norm_ratio`` between
  ``model.temporal_params()`` and ``model.spectral_params()`` (None when a
  branch is disabled) and appends one JSON line per epoch through
  ``diagnostics.DiagnosticsLogger`` when ``log_path`` is given;
- computes on the full test set: H(alpha), alpha temporal variance,
  per-class alpha, S(0.9) gate saturation and a permutation null
  (n_perm=20); quantities without a corresponding tensor are None;
- oracle control ``fusion == "fixed_class"``: the true batch labels are
  forwarded to the model (``model(X, y)``) in the training loop, in
  validation/test evaluation and in the permutation null, since the
  per-class alpha vector is indexed by the true label.

Returns::

    {"config": str, "seed": int, "accuracy": float, "macro_f1": float,
     "best_val_acc": float, "epochs_run": int,
     "diagnostics": [per-epoch record dicts],
     "final": {"rho_last": float|None, "H_alpha": float|None,
               "alpha_tvar": float|None, "S_tau": float|None,
               "per_class_alpha": dict|None, "perm_null": dict|None},
     "model": ATSFDCG}  # only when return_model=True (best weights restored)
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset

from . import degradation as deg
from . import diagnostics as diag
from .model import ATSFDCG
from .remedies import OGMBalancer
from .utils import ExpConfig, accuracy_macro_f1, set_seed
from .data import DatasetBundle

# architecture registry for train_one(arch=...); TSFTimesNet shares the full
# ATSFDCG contract (forward/cache/params), so no other code path branches
ARCHS = ("atsf", "tsnet")


def build_model(arch: str, n_channels: int, n_classes: int,
                cfg: ExpConfig) -> nn.Module:
    """Instantiate the audit model for ``arch`` ("atsf" | "tsnet")."""
    if arch == "atsf":
        return ATSFDCG(n_channels, n_classes, cfg)
    if arch == "tsnet":
        from .model_tsnet import TSFTimesNet  # local: keeps import graph flat
        return TSFTimesNet(n_channels, n_classes, cfg)
    raise ValueError(f"unknown arch {arch!r}; expected one of {ARCHS}")


def _make_loader(X: np.ndarray, y: np.ndarray, batch_size: int,
                 shuffle: bool, device: str) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(np.ascontiguousarray(X)).float(),
        torch.from_numpy(np.ascontiguousarray(y)).long(),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(0) if shuffle else None)


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, device: str,
              collect_cache: bool = False, pass_labels: bool = False):
    """Mean accuracy/F1; optionally also concatenated alpha/G and labels.

    ``pass_labels`` forwards the batch labels to the model (required by the
    ``fixed_class`` oracle fusion; ignored by every other fusion mode)."""
    model.eval()
    preds, trues = [], []
    alphas, gates = [], []
    for X, y in loader:
        X = X.to(device)
        logits, cache = model(X, y.to(device)) if pass_labels else model(X)
        preds.append(logits.argmax(dim=1).cpu().numpy())
        trues.append(y.numpy())
        if collect_cache:
            if cache["alpha"] is not None:
                alphas.append(cache["alpha"].cpu())
            if cache["G"] is not None:
                gates.append(cache["G"].cpu())
    y_true = np.concatenate(trues)
    y_pred = np.concatenate(preds)
    acc, f1 = accuracy_macro_f1(y_true, y_pred)
    alpha = torch.cat(alphas) if alphas else None
    G = torch.cat(gates) if gates else None
    return acc, f1, y_true, y_pred, alpha, G


def _degradation_eval(model: nn.Module, X_test: np.ndarray,
                      y_test: np.ndarray, cfg: ExpConfig, device: str,
                      pass_labels: bool) -> dict[str, float]:
    """Test-set accuracy under each of the 7 sensor degradations (opt-in).

    Degradations are applied to the already z-scored test windows
    (post-normalization, in units of per-channel std — see
    ``degradation.py``).  Each degradation uses a dedicated RandomState
    seeded ``cfg.seed + deg.SEED_OFFSET + index``, so the result is fully
    deterministic given the run seed and leaves the global RNGs untouched.
    """
    out: dict[str, float] = {}
    for i, name in enumerate(deg.DEGRADATION_NAMES):
        rng = np.random.RandomState(cfg.seed + deg.SEED_OFFSET + i)
        Xd = deg.apply_degradation(X_test, name, rng)
        loader = _make_loader(Xd, y_test, cfg.batch_size, shuffle=False,
                              device=device)
        acc_d, _, _, _, _, _ = _evaluate(model, loader, device,
                                         pass_labels=pass_labels)
        out[name] = acc_d
    return out


def train_one(cfg: ExpConfig, bundle: DatasetBundle,
              log_path: str | None = None,
              return_model: bool = False,
              arch: str = "atsf",
              log_epoch_indicators: bool = False,
              degradation: bool = False) -> dict:
    """Train + early-stop + test one configuration (see module docstring).

    With ``return_model=True`` the returned dict additionally contains the
    key ``"model"`` — the network with the best-validation weights restored
    (used by ``eval_dump`` to re-run test inference).  Default ``False``
    keeps the historical return contract unchanged.  ``arch`` selects the
    model class ("atsf" default = ``ATSFDCG``; "tsnet" = ``TSFTimesNet``);
    the default path is byte-identical to the historical behaviour.

    Opt-in features (both default off, default call byte-identical):

    - ``log_epoch_indicators``: after each epoch's validation, run one
      forward pass over a fixed deterministic subset of the validation set
      (first 256 windows, fixed order, non-shuffled) and add ``h_alpha``
      (routing entropy) and ``s_tau`` (gate saturation at tau=0.9) to the
      epoch record — mirroring the final-diagnostics guard logic, so
      configs without fusion/gating get JSON nulls.
    - ``degradation``: after the final test evaluation, evaluate the
      trained model on the test set under the 7 sensor degradations of
      ``degradation.py`` and store ``final["degradation"] =
      {name: accuracy}``.
    """
    set_seed(cfg.seed)
    device = cfg.device
    n_channels = int(bundle.X_train.shape[1])
    n_classes = len(bundle.class_names)

    model = build_model(arch, n_channels, n_classes, cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    ce = nn.CrossEntropyLoss()
    balancer = OGMBalancer() if cfg.r1_balanced else None
    coef_t = coef_f = 1.0

    train_loader = _make_loader(bundle.X_train, bundle.y_train,
                                cfg.batch_size, shuffle=True, device=device)
    val_loader = _make_loader(bundle.X_val, bundle.y_val,
                              cfg.batch_size, shuffle=False, device=device)
    test_loader = _make_loader(bundle.X_test, bundle.y_test,
                               cfg.batch_size, shuffle=False, device=device)
    has_val = len(bundle.y_val) > 0
    # fixed_class is an oracle fusion: the true labels index the per-class
    # alpha vector, train and eval alike (deliberate diagnostic upper bound)
    pass_labels = cfg.fusion == "fixed_class"

    # opt-in per-epoch indicators: fixed deterministic validation subset
    # (first 256 windows, fixed order, never shuffled), built once
    ind_loader: DataLoader | None = None
    if log_epoch_indicators:
        src_X, src_y = (bundle.X_val, bundle.y_val) if has_val \
            else (bundle.X_train, bundle.y_train)
        n_sub = min(256, len(src_y))
        base = TensorDataset(
            torch.from_numpy(np.ascontiguousarray(src_X)).float(),
            torch.from_numpy(np.ascontiguousarray(src_y)).long(),
        )
        ind_loader = DataLoader(Subset(base, list(range(n_sub))),
                                batch_size=cfg.batch_size, shuffle=False)

    has_temporal = len(model.temporal_params()) > 0
    has_spectral = len(model.spectral_params()) > 0
    rho_defined = has_temporal and has_spectral

    logger = diag.DiagnosticsLogger(log_path) if log_path else None
    history: list[dict] = []

    best_val = -1.0
    best_state = copy.deepcopy(model.state_dict())
    bad_epochs = 0
    epochs_run = 0

    try:
        for epoch in range(cfg.epochs):
            model.train()
            epochs_run = epoch + 1
            tot_loss, tot_n = 0.0, 0
            bal_sum, bal_cnt = 0.0, 0  # R2: mean L_balance over training batches
            rhos: list[float] = []
            # R1 auxiliary-head accuracy on training batches (for OGM)
            aux_correct_t = aux_correct_f = aux_n = 0

            for X, y in train_loader:
                X, y = X.to(device), y.to(device)
                opt.zero_grad()
                logits, cache = model(X, y) if pass_labels else model(X)
                loss = ce(logits, y)
                if cfg.r1_balanced:
                    if cache["logits_t"] is not None:
                        loss = loss + cfg.lambda_aux * coef_t * ce(cache["logits_t"], y)
                        aux_correct_t += int((cache["logits_t"].argmax(1) == y).sum())
                    if cache["logits_f"] is not None:
                        loss = loss + cfg.lambda_aux * coef_f * ce(cache["logits_f"], y)
                        aux_correct_f += int((cache["logits_f"].argmax(1) == y).sum())
                    aux_n += int(y.numel())
                if cfg.r2_load_balanced:  # R2: MoE load-balancing term
                    l_balance = model.balance_loss()
                    loss = loss + cfg.r2_lambda_balance * l_balance
                    bal_sum += float(l_balance.detach())
                    bal_cnt += 1
                loss.backward()
                if rho_defined:
                    rhos.append(diag.grad_norm_ratio(model.temporal_params(),
                                                     model.spectral_params()))
                opt.step()
                tot_loss += float(loss) * int(y.numel())
                tot_n += int(y.numel())

            # validation (fall back to train accuracy if no val split)
            if has_val:
                val_acc, _, _, _, _, _ = _evaluate(model, val_loader, device,
                                                   pass_labels=pass_labels)
            else:
                val_acc, _, _, _, _, _ = _evaluate(model, train_loader, device,
                                                   pass_labels=pass_labels)

            rho_epoch = float(np.mean(rhos)) if rhos else None
            record = {
                "train_loss": tot_loss / max(tot_n, 1),
                "val_acc": val_acc,
                "rho": rho_epoch,
            }
            if cfg.r2_load_balanced:  # R2: per-epoch mean L_balance
                record["L_balance"] = bal_sum / max(bal_cnt, 1)
            if cfg.r1_balanced and aux_n:
                acc_t = aux_correct_t / aux_n
                acc_f = aux_correct_f / aux_n
                record["acc_t_aux"] = acc_t
                record["acc_f_aux"] = acc_f
                record["coef_t"] = coef_t
                record["coef_f"] = coef_f
                # coefficients for the NEXT epoch from this epoch's aux accs
                coef_t, coef_f = balancer.coefficients(acc_t, acc_f)

            if log_epoch_indicators:  # opt-in per-epoch routing/gate metrics
                _, _, _, _, alpha_i, G_i = _evaluate(
                    model, ind_loader, device, collect_cache=True,
                    pass_labels=pass_labels)
                # mirrors the final-diagnostics guard logic: null when the
                # config has no fusion alpha / no gate tensor
                record["h_alpha"] = diag.routing_entropy(alpha_i) \
                    if alpha_i is not None else None
                record["s_tau"] = diag.gate_saturation(G_i, tau=0.9) \
                    if G_i is not None else None

            history.append({"epoch": epoch, **record})
            if logger is not None:
                logger.log_epoch(epoch, record)

            if val_acc > best_val:
                best_val = val_acc
                best_state = copy.deepcopy(model.state_dict())
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= cfg.patience:
                    break
    finally:
        if logger is not None:
            logger.close()

    # ---- restore best weights, test --------------------------------------
    model.load_state_dict(best_state)
    acc, f1, y_true, _, alpha, G = _evaluate(model, test_loader, device,
                                             collect_cache=True,
                                             pass_labels=pass_labels)

    rho_last = history[-1]["rho"] if history else None
    final = {
        "rho_last": rho_last,
        "H_alpha": None,
        "alpha_tvar": None,
        "S_tau": None,
        "per_class_alpha": None,
        "perm_null": None,
    }
    if alpha is not None:
        final["H_alpha"] = diag.routing_entropy(alpha)
        final["alpha_tvar"] = diag.alpha_temporal_variance(alpha)
        final["per_class_alpha"] = diag.per_class_alpha(
            alpha, torch.from_numpy(y_true), n_classes)
        X_test = torch.from_numpy(np.ascontiguousarray(bundle.X_test)).float()
        y_test = torch.from_numpy(np.ascontiguousarray(bundle.y_test)).long()
        final["perm_null"] = diag.permutation_null_alpha(
            model, X_test, y_test, n_perm=20)
    if G is not None:
        final["S_tau"] = diag.gate_saturation(G, tau=0.9)
    if degradation:  # opt-in test-time sensor-degradation robustness eval
        final["degradation"] = _degradation_eval(
            model, bundle.X_test, bundle.y_test, cfg, device, pass_labels)

    result = {
        "config": cfg.name,
        "seed": cfg.seed,
        "accuracy": acc,
        "macro_f1": f1,
        "best_val_acc": best_val,
        "epochs_run": epochs_run,
        "diagnostics": history,
        "final": final,
    }
    if return_model:
        result["model"] = model
    return result
