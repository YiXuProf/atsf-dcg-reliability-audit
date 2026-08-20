"""Run-level panels for manuscript Figs. 3–7.

Regenerates every figure from an ``atsf_dcg.eval_dump`` artefact directory
produced under the run-level leakage-free protocol::

    python scripts/make_fig03_07_runlevel.py \
        --dump-dir output/intermediate/eval_dump/full_seed42 \
        --out-dir output/figures/fusion

Stems match ``output/paper/figures/``: fig03_…–fig07_….
Every figure is written as PNG (300 dpi) + SVG.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from atsf_dcg.data import load_nppad, make_synthetic  # noqa: E402
from atsf_dcg.paths import FIGURES_ROOT, INTERMEDIATE_ROOT  # noqa: E402

# ---- house style ------------------------------------------------------------
PALETTE = {"light": "#6B8CBB", "dark": "#2E4A62", "gray": "#7A8B99",
           "accent": "#4A6FA5"}


def _style() -> None:
    plt.rcParams.update({
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": PALETTE["dark"],
        "axes.labelcolor": PALETTE["dark"],
        "xtick.color": PALETTE["dark"],
        "ytick.color": PALETTE["dark"],
        "figure.dpi": 300,
    })


def _save(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    for ext in ("png", "svg"):
        fig.savefig(out_dir / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] wrote {out_dir / (stem + '.png')} (+ .svg)", flush=True)


def _class_colors(n: int) -> np.ndarray:
    """n distinct class colours (tab20 for <=20 classes)."""
    cmap = plt.get_cmap("tab20")
    return cmap(np.linspace(0, 1, max(n, 2)))[:n]


def _stratified_subsample(y: np.ndarray, max_n: int, seed: int = 0) -> np.ndarray:
    """Deterministic stratified subsample indices (sorted)."""
    y = np.asarray(y)
    n = len(y)
    if n <= max_n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    per = max(1, max_n // len(classes))
    picked = []
    for c in classes:
        ci = np.where(y == c)[0]
        rng.shuffle(ci)
        picked.append(ci[:per])
    idx = np.concatenate(picked)
    if len(idx) < max_n:  # top up rounding loss from the leftovers
        rest = np.setdiff1d(np.arange(n), idx)
        rng.shuffle(rest)
        idx = np.concatenate([idx, rest[:max_n - len(idx)]])
    return np.sort(idx)


# ---- manuscript Fig. 3: fusion-weight histogram (generator fig2) ------------

def fig1_alpha_histogram(alpha: np.ndarray, out_dir: Path) -> None:
    flat = alpha.reshape(-1)
    mean = float(flat.mean())
    p5, p95 = (float(v) for v in np.percentile(flat, [5, 95]))
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.hist(flat, bins=60, range=(0, 1), color=PALETTE["light"],
            edgecolor="none")
    ax.axvline(0.5, color=PALETTE["gray"], linestyle="--", linewidth=1.0)
    ax.set_xlabel(r"fusion weight $\alpha$ (temporal branch)")
    ax.set_ylabel("count")
    ax.text(0.98, 0.95,
            f"mean = {mean:.3f}\nP5-P95 = [{p5:.3f}, {p95:.3f}]",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=PALETTE["gray"], alpha=0.9))
    _save(fig, out_dir, "fig03_fusion_weight_dist")
    print(f"[fig3] alpha: mean={mean:.4f} P5={p5:.4f} P95={p95:.4f} "
          f"n={flat.size}", flush=True)


# ---- manuscript Fig. 4: per-class fusion weights (generator fig3) -----------

def fig2_perclass_alpha(alpha: np.ndarray, y_true: np.ndarray,
                        class_names: list[str], out_dir: Path) -> None:
    per_window = alpha.mean(axis=1)  # (n,) mean over T' per window
    means, stds = [], []
    for k in range(len(class_names)):
        vals = per_window[y_true == k]
        means.append(float(vals.mean()) if len(vals) else np.nan)
        stds.append(float(vals.std(ddof=1)) if len(vals) > 1 else 0.0)
    order = np.argsort([c.lower() for c in class_names])  # alphabetical
    names = [class_names[i] for i in order]
    m = np.asarray(means)[order]
    s = np.asarray(stds)[order]

    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    ax.bar(np.arange(len(names)), m, yerr=s, capsize=2,
           color=PALETTE["light"], edgecolor=PALETTE["dark"], linewidth=0.5,
           error_kw=dict(ecolor=PALETTE["dark"], linewidth=0.6))
    ax.axhline(0.5, color=PALETTE["gray"], linestyle="--", linewidth=1.0)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel(r"mean $\alpha$")
    ax.set_ylim(0, 1)
    _save(fig, out_dir, "fig04_fusion_weight_by_class")

    finite = m[np.isfinite(m)]
    print(f"[fig4] per-class mean alpha band: "
          f"[{finite.min():.4f}, {finite.max():.4f}] "
          f"(spread={finite.max() - finite.min():.4f})", flush=True)


# ---- manuscript Fig. 5: gate heatmaps (generator fig4) ----------------------

def fig3_gate_heatmaps(gates: np.ndarray, preds: pd.DataFrame,
                       class_names: list[str], out_dir: Path,
                       n_panels: int = 5) -> None:
    """First correctly-classified test window from 5 different classes."""
    y_true = preds["y_true"].to_numpy()
    correct = preds["correct"].to_numpy().astype(bool)
    picks: list[int] = []
    for k in range(len(class_names)):
        if len(picks) >= n_panels:
            break
        idxs = np.where((y_true == k) & correct)[0]
        note = ""
        if len(idxs) == 0:  # no correct window for this class: fall back
            idxs = np.where(y_true == k)[0]
            note = " (no correct window; first window shown)"
        if len(idxs) == 0:
            continue
        picks.append(int(idxs[0]))
        if note:
            print(f"[fig5] class '{class_names[k]}'{note}", flush=True)

    fig, axes = plt.subplots(1, len(picks), figsize=(2.0 * len(picks), 2.2),
                             constrained_layout=True)
    axes = np.atleast_1d(axes)
    im = None
    for ax, wi in zip(axes, picks):
        im = ax.imshow(gates[wi], aspect="auto", origin="lower",
                       cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_title(f"{class_names[y_true[wi]]}", fontsize=8)
        ax.set_xlabel("t'", fontsize=8)
    axes[0].set_ylabel("channel")
    cbar = fig.colorbar(im, ax=list(axes), shrink=0.8)
    cbar.set_label("gate G", fontsize=8)
    _save(fig, out_dir, "fig05_gate_heatmap")

    sats = {class_names[y_true[wi]]: float((gates[wi] > 0.9).mean())
            for wi in picks}
    print("[fig5] gate saturation S(0.9) per panel: "
          + ", ".join(f"{k}={v:.4f}" for k, v in sats.items()), flush=True)


# ---- manuscript Fig. 6: t-SNE raw vs features (generator fig5) --------------

def fig4_tsne(z: np.ndarray, y_true: np.ndarray, class_names: list[str],
              X_test: np.ndarray, out_dir: Path, perplexity: float,
              max_n: int = 2000) -> None:
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score

    idx = _stratified_subsample(y_true, max_n)
    raw = X_test[idx].reshape(len(idx), -1)
    feat = z[idx]
    ys = y_true[idx]
    print(f"[fig6] subsample n={len(idx)} (stratified over "
          f"{len(np.unique(ys))} classes)", flush=True)

    # quantitative upgrade: silhouette in the ORIGINAL high-dim spaces
    sil_raw = float(silhouette_score(raw, ys))
    sil_feat = float(silhouette_score(feat, ys))
    print(f"[fig6] silhouette (high-dim): raw={sil_raw:.4f} -> "
          f"z={sil_feat:.4f} (delta={sil_feat - sil_raw:+.4f})", flush=True)

    perp = float(min(perplexity, (len(idx) - 1) / 3))
    emb_raw = TSNE(n_components=2, perplexity=perp, random_state=0,
                   init="pca", learning_rate="auto").fit_transform(raw)
    emb_feat = TSNE(n_components=2, perplexity=perp, random_state=0,
                    init="pca", learning_rate="auto").fit_transform(feat)

    colors = _class_colors(len(class_names))
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.0))
    for ax, emb, title in ((axes[0], emb_raw, "raw input windows"),
                           (axes[1], emb_feat, "fused feature z")):
        for k, name in enumerate(class_names):
            sel = ys == k
            if not sel.any():
                continue
            ax.scatter(emb[sel, 0], emb[sel, 1], s=3, color=colors[k],
                       label=name, linewidths=0, rasterized=True)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=6,
                   markerscale=3, frameon=False, borderaxespad=0.0)
    fig.subplots_adjust(wspace=0.05)
    _save(fig, out_dir, "fig06_tsne")


# ---- manuscript Fig. 7: confusion matrix (generator fig6) -------------------

def fig5_confusion_matrix(preds: pd.DataFrame, class_names: list[str],
                          meta: dict, out_dir: Path) -> None:
    from sklearn.metrics import confusion_matrix

    K = len(class_names)
    cm = confusion_matrix(preds["y_true"], preds["y_pred"],
                          labels=np.arange(K))
    with np.errstate(invalid="ignore", divide="ignore"):
        cm_norm = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(K))
    ax.set_yticks(np.arange(K))
    ax.set_xticklabels(class_names, rotation=90, fontsize=6)
    ax.set_yticklabels(class_names, fontsize=6)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"config={meta['config']} seed={meta['seed']} "
                 "(run-level leakage-free protocol)", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8, label="row-normalised rate")
    _save(fig, out_dir, "fig07_confusion_matrix")

    acc = float(np.trace(cm) / max(cm.sum(), 1))
    print(f"[fig7] accuracy implied by confusion matrix: {acc:.4f} "
          f"({int(np.trace(cm))}/{int(cm.sum())})", flush=True)


# ---- driver -------------------------------------------------------------------

def _load_raw_test_windows(meta: dict, data_root: str | None) -> np.ndarray:
    """Rebuild the test split the dump was produced from (Fig. 6 raw panel)."""
    if data_root:
        bundle = load_nppad(data_root, window=int(meta["window"]),
                            stride=int(meta["stride"]))
    elif meta.get("synthetic_params") is not None:
        bundle = make_synthetic(**meta["synthetic_params"])
    else:
        raise ValueError("--data-root is required for a non-synthetic dump")
    return bundle.X_test


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Run-level Figs. 3–7 regeneration "
                                 "from an eval_dump artefact directory")
    ap.add_argument("--dump-dir", type=str,
                    default=str(INTERMEDIATE_ROOT / "eval_dump" / "full_seed42"),
                    help="e.g. output/intermediate/eval_dump/full_seed42")
    ap.add_argument("--data-root", type=str, default=None,
                    help="NPPAD tree (only needed for the Fig. 6 raw panel "
                         "when the dump is not synthetic)")
    ap.add_argument("--out-dir", type=str,
                    default=str(FIGURES_ROOT / "fusion"))
    ap.add_argument("--perplexity", type=float, default=30.0,
                    help="t-SNE perplexity (default 30)")
    ap.add_argument("--max-tsne", type=int, default=2000,
                    help="max windows for the t-SNE/silhouette subsample")
    args = ap.parse_args(argv)

    _style()
    dump_dir = Path(args.dump_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(dump_dir / "meta.json", "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    preds = pd.read_csv(dump_dir / "predictions.csv")
    feats = np.load(dump_dir / "features.npz", allow_pickle=False)
    z, y_true = feats["z"], feats["labels"]
    class_names = [str(c) for c in feats["class_names"]]
    print(f"[setup] dump={dump_dir} config={meta['config']} "
          f"seed={meta['seed']} windows={len(y_true)} "
          f"classes={len(class_names)}", flush=True)

    alpha_path = dump_dir / "alpha.npy"
    if alpha_path.is_file():
        alpha = np.load(alpha_path)
        fig1_alpha_histogram(alpha, out_dir)
        fig2_perclass_alpha(alpha, y_true, class_names, out_dir)
    else:
        print("[skip] alpha.npy not in dump (config has no fusion alpha); "
              "fig1/fig2 skipped", flush=True)

    gates_path = dump_dir / "gates.npy"
    if gates_path.is_file():
        fig3_gate_heatmaps(np.load(gates_path), preds, class_names, out_dir)
    else:
        print("[skip] gates.npy not in dump (gating=none); fig3 skipped",
              flush=True)

    X_test = _load_raw_test_windows(meta, args.data_root)
    if len(X_test) != len(y_true):
        raise ValueError(f"rebuilt test split has {len(X_test)} windows but "
                         f"the dump has {len(y_true)}; window/stride or data "
                         "root mismatch")
    fig4_tsne(z, y_true, class_names, X_test, out_dir,
              perplexity=args.perplexity, max_n=args.max_tsne)

    fig5_confusion_matrix(preds, class_names, meta, out_dir)
    print(f"[done] run-level figures written to {out_dir}/", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
