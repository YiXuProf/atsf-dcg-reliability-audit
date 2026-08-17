"""Evaluation-dump CLI: per-window test-set artefacts for figure regeneration.

The paper's Figs. 3–7 were produced under a superseded leaky protocol; this
module re-trains ONE configuration/seed under the run-level leakage-free
split (``data.load_nppad``) and dumps everything the figure script
(``make_fig_runlevel.py``) needs::

    python -m atsf_dcg.eval_dump --data-root /path/to/NPPAD \
        --config full --seed 42 --out-dir output/intermediate/eval_dump
    python -m atsf_dcg.eval_dump --synthetic --smoke --out-dir /tmp/dump
    # train-split alpha means for the fixed-fusion control experiments:
    python -m atsf_dcg.eval_dump --data-root /path/to/NPPAD \
        --config full --seed 42 --split train --out-dir output/intermediate/eval_dump

``--split {test,train,val}`` (default ``test``) selects which bundle arrays
the dump runs over; non-test dumps land in ``<slug>_seed<seed>_<split>/`` so
they never overwrite the test dump.

Outputs under ``--out-dir/<slug>_seed<seed>[_<split>]/``:

- ``predictions.csv`` : window_idx, class_name, y_true, y_pred, correct
- ``alpha.npy``       : (n_windows, T') float32 fusion weight (skipped, with a
                        printed note, when the config produces no alpha)
- ``alpha_means.json``: written whenever ``alpha.npy`` is written —
                        ``{"split": ..., "global": mean over all windows and
                        t', "per_class": {class_name: mean, ...},
                        "n_windows": n}``; per-class means use the TRUE
                        labels of the dumped split (dump the train split for
                        the fixed-fusion controls -> no leakage)
- ``gates.npy``       : (n_windows, C, T') float32 gate matrix G
                        (skipped when gating == "none")
- ``features.npz``    : z (n_windows, 128) float32, labels int64,
                        class_names (str array)
- ``meta.json``       : config/seed, split, recomputed accuracy/macro-F1,
                        window/stride, protocol note

Training reuses ``train.train_one(..., return_model=True)`` (same code path
as ``run_experiments``); inference is batched under ``torch.no_grad()``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import load_nppad, make_synthetic
from .run_experiments import (
    FIXED_FUSION_MODES, _slug, _to_jsonable, build_configs,
    inject_fixed_alphas,
)
from .train import _make_loader, train_one
from .utils import accuracy_macro_f1


@torch.no_grad()
def _collect(model: torch.nn.Module, X_split: np.ndarray, y_split: np.ndarray,
             batch_size: int, device: str) -> dict:
    """Batched eval-mode inference over one ENTIRE split's arrays.

    Returns y_true, y_pred, z (n,128), alpha (n,T'|None), G (n,C,T'|None)
    as numpy arrays, plus the recomputed accuracy/macro-F1.  The true labels
    are forwarded to the model when it runs the ``fixed_class`` oracle
    fusion; every other fusion mode ignores them.
    """
    model.eval()
    loader = _make_loader(X_split, y_split, batch_size,
                          shuffle=False, device=device)
    pass_labels = getattr(model, "fusion", None) == "fixed_class"
    preds, trues, zs, alphas, gates = [], [], [], [], []
    for X, y in loader:
        X = X.to(device)
        logits, cache = model(X, y.to(device)) if pass_labels else model(X)
        preds.append(logits.argmax(dim=1).cpu().numpy())
        trues.append(y.numpy())
        zs.append(cache["z"].cpu().numpy())
        if cache["alpha"] is not None:
            alphas.append(cache["alpha"].cpu().numpy())
        if cache["G"] is not None:
            gates.append(cache["G"].cpu().numpy())
    y_true = np.concatenate(trues)
    y_pred = np.concatenate(preds)
    acc, f1 = accuracy_macro_f1(y_true, y_pred)

    alpha = None
    if alphas:
        alpha = np.concatenate(alphas).astype(np.float32)  # (n, 1|C, T')
        if alpha.ndim == 3 and alpha.shape[1] == 1:
            alpha = alpha[:, 0, :]  # (B,1,T') -> (n,T')
        elif alpha.ndim == 3:  # R2 per-channel alpha: channel mean for figures
            print("[eval_dump] alpha is per-channel (R2); dumping the "
                  "channel mean as (n,T')", flush=True)
            alpha = alpha.mean(axis=1).astype(np.float32)
    G = np.concatenate(gates).astype(np.float32) if gates else None
    return {"y_true": y_true, "y_pred": y_pred, "acc": acc, "f1": f1,
            "z": np.concatenate(zs).astype(np.float32),
            "alpha": alpha, "G": G}


def run(args: argparse.Namespace) -> dict:
    # ---- config from the experiment grid -----------------------------------
    configs = {c.name: c for c in build_configs(smoke=False)}
    if args.config not in configs:
        raise ValueError(f"unknown config '{args.config}'; "
                         f"available: {sorted(configs)}")
    epochs = 2 if args.smoke else args.epochs
    cfg = replace(configs[args.config], seed=args.seed, epochs=epochs,
                  patience=min(configs[args.config].patience, epochs))
    if cfg.fusion in FIXED_FUSION_MODES and not args.fixed_alpha_file:
        raise ValueError(
            f"config '{cfg.name}' uses fusion='{cfg.fusion}' and needs "
            "--fixed-alpha-file PATH (an alpha_means.json written by a "
            "train-split eval dump of an adaptive config)")

    # ---- data --------------------------------------------------------------
    synth_params = None
    if args.synthetic:
        if args.smoke:
            synth_params = dict(n_classes=6, runs_per_class=3,
                                n_channels=16, seed=0)
        else:
            synth_params = dict(seed=0)
        bundle = make_synthetic(**synth_params)
        data_source = "synthetic"
    else:
        bundle = load_nppad(args.data_root, window=args.window,
                            stride=args.stride)
        data_source = str(args.data_root)

    # ---- split selection ---------------------------------------------------
    split_arrays = {"train": (bundle.X_train, bundle.y_train),
                    "val": (bundle.X_val, bundle.y_val),
                    "test": (bundle.X_test, bundle.y_test)}
    X_split, y_split = split_arrays[args.split]
    if args.split == "val" and len(y_split) == 0:
        raise ValueError(
            "--split val requested but this dataset bundle has no validation "
            "split (0 val windows)")
    if cfg.fusion in FIXED_FUSION_MODES:  # inject after bundle load: the
        # bundle's class order is the source of truth for the index mapping
        with open(args.fixed_alpha_file, "r", encoding="utf-8") as fh:
            cfg = inject_fixed_alphas(cfg, json.load(fh),
                                      list(bundle.class_names))
    print(f"[setup] data={data_source} config={cfg.name} seed={args.seed} "
          f"epochs={epochs} device={cfg.device} split={args.split} "
          f"{args.split}_windows={len(y_split)}", flush=True)

    # ---- train one run (same code path as run_experiments) -----------------
    res = train_one(cfg, bundle, log_path=None, return_model=True)
    model = res["model"]
    print(f"[train] acc={res['accuracy']:.4f} macroF1={res['macro_f1']:.4f} "
          f"epochs_run={res['epochs_run']}", flush=True)

    # ---- full split inference ----------------------------------------------
    out = _collect(model, X_split, y_split, cfg.batch_size, cfg.device)
    y_true, y_pred = out["y_true"], out["y_pred"]
    print(f"[cross-check] recomputed ({args.split}) acc={out['acc']:.4f} "
          f"macroF1={out['f1']:.4f} vs train_one acc={res['accuracy']:.4f} "
          f"macroF1={res['macro_f1']:.4f}", flush=True)

    split_tag = "" if args.split == "test" else f"_{args.split}"
    dump_dir = Path(args.out_dir) / f"{_slug(cfg.name)}_seed{args.seed}{split_tag}"
    dump_dir.mkdir(parents=True, exist_ok=True)

    # predictions.csv
    class_names = list(bundle.class_names)
    pd.DataFrame({
        "window_idx": np.arange(len(y_true)),
        "class_name": [class_names[i] for i in y_true],
        "y_true": y_true,
        "y_pred": y_pred,
        "correct": (y_true == y_pred).astype(int),
    }).to_csv(dump_dir / "predictions.csv", index=False)

    # alpha.npy (skipped when the config has no alpha) + alpha_means.json
    if out["alpha"] is not None:
        alpha_arr = out["alpha"]  # (n, T') float32
        np.save(dump_dir / "alpha.npy", alpha_arr)
        # global + per-class means (TRUE labels of the dumped split), used
        # to calibrate the fixed_global / fixed_class control experiments;
        # float64 accumulation for a stable cross-check against alpha.npy
        a64 = alpha_arr.astype(np.float64)
        per_class = {}
        for k, name in enumerate(class_names):
            mask = y_true == k
            if bool(mask.any()):
                per_class[name] = float(a64[mask].mean())
        alpha_means = {
            "split": args.split,
            "global": float(a64.mean()),
            "per_class": per_class,
            "n_windows": int(len(y_true)),
        }
        with open(dump_dir / "alpha_means.json", "w", encoding="utf-8") as fh:
            json.dump(_to_jsonable(alpha_means), fh, indent=2,
                      ensure_ascii=False)
    else:
        print(f"[eval_dump] config '{cfg.name}' has no fusion alpha; "
              "alpha.npy skipped", flush=True)

    # gates.npy (skipped when gating == "none")
    if out["G"] is not None:
        np.save(dump_dir / "gates.npy", out["G"])
    else:
        print(f"[eval_dump] config '{cfg.name}' has gating=none; "
              "gates.npy skipped", flush=True)

    # features.npz
    np.savez(dump_dir / "features.npz", z=out["z"], labels=y_true,
             class_names=np.asarray(class_names, dtype=str))

    # meta.json
    meta = {
        "config": cfg.name,
        "seed": args.seed,
        "data_source": data_source,
        "synthetic_params": synth_params,
        "split": args.split,
        "test_accuracy": out["acc"],
        "test_macro_f1": out["f1"],
        "train_one_accuracy": res["accuracy"],
        "train_one_macro_f1": res["macro_f1"],
        "n_windows": int(len(y_true)),
        "n_test_windows": int(len(y_true)),  # legacy alias (default split)
        "window": args.window,
        "stride": args.stride,
        "protocol": ("run-level leakage-free split (split unit = simulation "
                     "run; z-score statistics from training runs only)"),
        "alpha_shape": list(out["alpha"].shape) if out["alpha"] is not None else None,
        "gates_shape": list(out["G"].shape) if out["G"] is not None else None,
        "z_shape": list(out["z"].shape),
    }
    with open(dump_dir / "meta.json", "w", encoding="utf-8") as fh:
        json.dump(_to_jsonable(meta), fh, indent=2, ensure_ascii=False)

    print(f"[done] eval dump written to {dump_dir}/: predictions.csv, "
          f"{'alpha.npy, alpha_means.json, ' if out['alpha'] is not None else ''}"
          f"{'gates.npy, ' if out['G'] is not None else ''}"
          f"features.npz, meta.json", flush=True)
    return {"dump_dir": str(dump_dir), "meta": meta}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="ATSF-DCG evaluation dump "
                                 "(run-level figure regeneration)")
    src = ap.add_mutually_exclusive_group(required=False)
    src.add_argument("--data-root", type=str, default=None,
                     help="dataset root (default: data/NuclearPowerPlantAccidentData)")
    src.add_argument("--synthetic", action="store_true",
                     help="use make_synthetic instead of NPPAD")
    ap.add_argument("--config", type=str, default="full",
                    help="config name from run_experiments.build_configs()")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", type=str, choices=("test", "train", "val"),
                    default="test",
                    help="which bundle split to dump (default: test); "
                         "use train to calibrate the fixed-fusion controls "
                         "without label leakage")
    ap.add_argument("--fixed-alpha-file", type=str, default=None,
                    help="alpha_means.json for fixed_global/fixed_class "
                         "configs (written by a train-split dump of an "
                         "adaptive config)")
    ap.add_argument("--smoke", action="store_true",
                    help="synthetic micro data, epochs=2 (testing)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--out-dir", type=str, default="output/intermediate/eval_dump")
    args = ap.parse_args(argv)
    if args.smoke:
        args.synthetic = True  # smoke always runs on synthetic micro data
    if not args.synthetic and not args.data_root:
        from .paths import DATA_ROOT, nppad_root
        args.data_root = str(nppad_root())
        if not Path(args.data_root).exists():
            ap.error(
                f"no data at {args.data_root}. Download NPPAD into "
                f"{DATA_ROOT}/ (see data/README.md), or pass --data-root / "
                f"--synthetic")
    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
