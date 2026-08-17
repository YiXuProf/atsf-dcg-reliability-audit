"""Experiment runner CLI (SPEC.md §run_experiments.py, integrator).

Usage::

    python -m atsf_dcg.run_experiments --data-root /path/to/NPPAD
    python -m atsf_dcg.run_experiments --synthetic --smoke

Runs the 18-config ablation/remedy grid plus 3 reviewer-mandated controls
(``r1_w/o_spectral``, ``full_fixed_global``, ``full_fixed_class``; the two
fixed-fusion controls take their constant alphas from
``--fixed-alpha-file``, an ``alpha_means.json`` written by a train-split
``eval_dump``).  ``--smoke`` restricts to
``full`` + ``full_r2`` + ``full_r4_lstm``, synthetic micro data,
2 seeds x 2 epochs, and writes
under ``--out-dir`` (default ``output/experiments/nppad_atsf_full``)::

- ``results_table.csv``   : config, accuracy "mean+-std", macro-F1 "mean+-std"
                            (feeds manuscript Tables II / III / VI)
- ``remedy_table.csv``    : per-config means of final diagnostic metrics
                            (manuscript Table IV: H(alpha), rho, S(tau), perm-null z)
- ``significance.csv``    : paired t-test + Wilcoxon p-values vs ``full``
                            (feeds Tables II / III / VI)
- ``diagnostics/{slug}_seed{seed}.jsonl`` : per-epoch training diagnostics
- ``protocol_report.json``: DatasetBundle.report + run settings (paper 4.1/4.7)

Cross-dataset / cross-architecture replication (plan v6, Stage 3)::

    python -m atsf_dcg.run_experiments --cell replication --dataset tep \
        --arch tsnet --data-root /path/to/TEP
    python -m atsf_dcg.run_experiments --cell replication --dataset tep \
        --arch tsnet --synthetic --smoke   # sandbox smoke path

``--dataset tep`` loads the MathWorks TEP .mat files via
``data_tep.load_tep`` (exact ``--tep-runs`` train/val/test run counts,
default 98,21,21; stride default 128); ``--dataset paderborn`` loads the
Paderborn bearing .mat tree via ``data_paderborn.load_paderborn`` (4
bearing-code classes; default ``--split-unit run`` is the 60/20/20
run-level split stratified over (bearing, setting) cells; ``--split-unit
bearing`` is the 32-bearing held-out split and writes to
``output/experiments/CellD_Paderborn_ATSF_bearing`` unless ``--out-dir``
is given; stride default 128); ``--arch tsnet`` trains
``model_tsnet.TSFTimesNet``; ``--cell replication`` runs the 7-config
replication grid (+ ``tsnet_vanilla`` under ``--arch tsnet``).  Unless
``--out-dir`` is given, non-default cells write to
``output/experiments/<Cell*>`` so cells never clobber each other;
``--merge-only`` takes the same ``--cell/--dataset/--arch`` flags.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .data import DatasetBundle, load_nppad, make_synthetic
from .data_paderborn import load_paderborn, make_synthetic_paderborn
from .data_tep import load_tep, make_synthetic_tep
from .diagnostics import _to_jsonable
from .train import ARCHS, train_one
from .utils import ExpConfig

DEFAULT_SEEDS = [42, 43, 44, 45, 46]
SMOKE_SEEDS = [42, 43]

FINAL_METRICS = ["H_alpha", "alpha_tvar", "S_tau", "rho_last", "perm_null_z"]

DATASETS = ("nppad", "tep", "paderborn")
CELLS = ("full", "replication")

# the 7 replication-cell configs (plan v6 Stage 3); identical ExpConfig
# semantics to their namesakes in the full grid
REPLICATION_NAMES = ["full", "w/o_spectral", "w/o_temporal", "w/o_fusion",
                     "w/o_gating", "full_r1", "full_r2_gumbel"]


def build_configs(smoke: bool = False, cell: str = "full",
                  arch: str = "atsf") -> list[ExpConfig]:
    """The 18-config grid from SPEC.md + SPEC v2, plus 3 reviewer-mandated
    control configs (smoke: full, full_r2, full_r4_lstm only).

    ``cell="replication"`` instead builds the 7-config cross-dataset /
    cross-architecture replication grid (plan v6 Stage 3): full,
    w/o_spectral, w/o_temporal, w/o_fusion, w/o_gating, full_r1,
    full_r2_gumbel — plus, when ``arch="tsnet"``, an 8th config
    ``tsnet_vanilla`` (use_spectral=False, fusion="none", gating="none":
    the plain TimesNet classifier, Cell D baseline).  Replication smoke:
    full, full_r2_gumbel (+ tsnet_vanilla under arch="tsnet").

    The default call ``build_configs()`` / ``build_configs(smoke)`` returns
    exactly the historical grid (regression-guarded by
    tests/test_cli_replication.py).

    ``full_fixed_global`` / ``full_fixed_class`` are built WITHOUT their
    constant alpha values (lazy injection): ``main()`` fills them from
    ``--fixed-alpha-file`` after the dataset bundle is loaded, so
    ``build_configs`` — and therefore ``--merge-only`` — works without the
    file."""
    if cell == "replication":
        configs = [
            ExpConfig(name="full"),
            ExpConfig(name="w/o_spectral", use_spectral=False),
            ExpConfig(name="w/o_temporal", use_temporal=False),
            ExpConfig(name="w/o_fusion", fusion="none"),
            ExpConfig(name="w/o_gating", gating="none"),
            ExpConfig(name="full_r1", r1_balanced=True),
            ExpConfig(name="full_r2_gumbel", r2_load_balanced=True,
                      r2_gumbel=True),
        ]
        if arch == "tsnet":
            configs.append(ExpConfig(name="tsnet_vanilla",
                                     use_spectral=False, fusion="none",
                                     gating="none"))
        if smoke:
            keep = {"full", "full_r2_gumbel", "tsnet_vanilla"}
            configs = [c for c in configs if c.name in keep]
        return configs
    if cell != "full":
        raise ValueError(f"unknown cell {cell!r}; expected one of {CELLS}")
    configs = [
        ExpConfig(name="full"),
        ExpConfig(name="w/o_spectral", use_spectral=False),
        ExpConfig(name="w/o_temporal", use_temporal=False),
        ExpConfig(name="w/o_fusion", fusion="none"),
        ExpConfig(name="w/o_dynamic_gating", gating="static"),
        ExpConfig(name="w/o_gating", gating="none"),
        ExpConfig(name="full_r1", r1_balanced=True),
        ExpConfig(name="full_r3_stft", spectral_frontend="stft"),
        ExpConfig(name="full_r3_sinc", spectral_frontend="sinc"),
        ExpConfig(name="full_r1_r3", r1_balanced=True, spectral_frontend="stft"),
        # SPEC v2: R2 (load-balanced fusion) and R4 (non-saturating gates)
        ExpConfig(name="full_r2", r2_load_balanced=True),
        ExpConfig(name="full_r2_gumbel", r2_load_balanced=True, r2_gumbel=True),
        ExpConfig(name="full_r4_sparsemax", gating="sparsemax"),
        ExpConfig(name="full_r4_entmax", gating="entmax"),
        ExpConfig(name="full_r4_lstm", gating="lstm"),
        ExpConfig(name="full_r1_r2_r3", r1_balanced=True, r2_load_balanced=True,
                  spectral_frontend="stft"),
        ExpConfig(name="full_all", r1_balanced=True, r2_load_balanced=True,
                  spectral_frontend="stft", gating="lstm"),
        # reviewer-mandated controls (constant alphas injected lazily in
        # main() from --fixed-alpha-file; see inject_fixed_alphas)
        ExpConfig(name="r1_w/o_spectral", use_spectral=False, r1_balanced=True),
        ExpConfig(name="full_fixed_global", fusion="fixed_global"),
        ExpConfig(name="full_fixed_class", fusion="fixed_class"),
    ]
    if smoke:
        configs = [c for c in configs if c.name in ("full", "full_r2", "full_r4_lstm")]
    return configs


def _slug(name: str) -> str:
    """Filesystem-safe slug: 'w/o_dynamic_gating' -> 'wo_dynamic_gating'."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name.replace("w/o", "wo")).strip("_")


FIXED_FUSION_MODES = ("fixed_global", "fixed_class")


def inject_fixed_alphas(cfg: ExpConfig, alpha_means: dict,
                        class_names: list[str]) -> ExpConfig:
    """Copy of ``cfg`` with its constant fusion alpha(s) injected from an
    eval-dump ``alpha_means.json`` dict (``{"global": float, "per_class":
    {class_name: float, ...}, ...}``).

    The bundle's ``class_names`` order (= label ids) is the source of truth
    for the per-class index mapping; a class missing from the file is an
    error.  Non-fixed configs are returned unchanged.
    """
    if cfg.fusion not in FIXED_FUSION_MODES:
        return cfg
    try:
        global_alpha = float(alpha_means["global"])
        per_class = alpha_means["per_class"]
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(
            "malformed alpha-means file (expected keys 'global' and "
            f"'per_class' as written by eval_dump): {e}") from e
    updates: dict = {}
    if cfg.fusion == "fixed_global":
        updates["fusion_fixed_alpha"] = global_alpha
    if cfg.fusion == "fixed_class":
        missing = [n for n in class_names if n not in per_class]
        if missing:
            raise ValueError(
                f"alpha-means per_class is missing classes {missing}; the "
                f"bundle class order (label ids) is {list(class_names)}")
        updates["fusion_fixed_class_alpha"] = tuple(
            float(per_class[n]) for n in class_names)
    return replace(cfg, **updates)


def _mean_std(xs: list[float]) -> tuple[float, float]:
    arr = np.asarray(xs, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0


def _fmt(ms: tuple[float, float]) -> str:
    return f"{ms[0]:.4f}±{ms[1]:.4f}"


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


# ---- multiple-comparison corrections (review M1) ---------------------------

def _holm_adjust(pvals: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values (family = all comparisons passed in)."""
    m = len(pvals)
    order = np.argsort(pvals, kind="stable")
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * pvals[idx])
        running = max(running, val)
        adj[idx] = running
    return adj


def _bh_adjust(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg (FDR) adjusted p-values."""
    m = len(pvals)
    order = np.argsort(pvals, kind="stable")[::-1]
    adj = np.empty(m)
    running = 1.0
    for rev, idx in enumerate(order):
        rank = m - rev
        running = min(running, pvals[idx] * m / rank)
        adj[idx] = min(1.0, running)
    return adj


def _paired_stats(cfg_vals: np.ndarray, full_vals: np.ndarray
                  ) -> tuple[float, float, float, float, float, float]:
    """On paired differences (config - full): mean diff, 95% CI, Cohen's d_z,
    paired t p, Wilcoxon p."""
    diff = cfg_vals - full_vals
    n = len(diff)
    mean_d = float(diff.mean())
    sd = float(diff.std(ddof=1)) if n > 1 else float("nan")
    if n > 1 and sd > 0:
        tcrit = float(stats.t.ppf(0.975, n - 1))
        half = tcrit * sd / math.sqrt(n)
        d_z = mean_d / sd
    else:
        half = float("nan")
        d_z = float("nan")
    try:
        t_p = float(stats.ttest_rel(cfg_vals, full_vals).pvalue)
    except Exception:
        t_p = float("nan")
    try:
        w_p = float(stats.wilcoxon(cfg_vals, full_vals).pvalue)
    except Exception:
        w_p = float("nan")
    return mean_d, mean_d - half, mean_d + half, d_z, t_p, w_p


def _shapiro_p(diff: np.ndarray) -> float:
    """Shapiro-Wilk normality diagnostic on paired differences (needs n>=3)."""
    if len(diff) < 3:
        return float("nan")
    try:
        return float(stats.shapiro(diff).pvalue)
    except Exception:
        return float("nan")


def _write_tables(out_dir: Path, configs: list[ExpConfig],
                  results: dict[str, list[dict]]) -> None:
    """Rebuild results_table.csv / remedy_table.csv / significance.csv from a
    results dict ``{config_name: [{"accuracy","macro_f1","final",...}]}``.
    Used both after a fresh run grid and in ``--merge-only`` mode."""
    # ---- results_table.csv (manuscript Tables II / III / VI) ---------------
    rows = []
    for cfg in configs:
        rs = results.get(cfg.name, [])
        if not rs:
            continue
        rows.append({
            "config": cfg.name,
            "accuracy": _fmt(_mean_std([r["accuracy"] for r in rs])),
            "macro_f1": _fmt(_mean_std([r["macro_f1"] for r in rs])),
        })
    pd.DataFrame(rows).to_csv(out_dir / "results_table.csv", index=False)

    # ---- remedy_table.csv (manuscript Table IV: diagnostic means) ---------
    rows = []
    for cfg in configs:
        rs = results.get(cfg.name, [])
        if not rs:
            continue
        row = {"config": cfg.name,
               "accuracy_mean": _mean_std([r["accuracy"] for r in rs])[0]}
        for m in FINAL_METRICS:
            vals = []
            for r in rs:
                f = r["final"]
                v = (f["perm_null"] or {}).get("z") if m == "perm_null_z" else f.get(m)
                if v is not None:
                    vals.append(float(v))
            row[m] = float(np.mean(vals)) if vals else np.nan
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "remedy_table.csv", index=False)

    # ---- significance.csv (vs full, paired by seed; accuracy + macro-F1,
    #      effect sizes, CIs, normality diagnostic, Holm/BH correction) ------
    full_by_seed = {r.get("seed"): (r["accuracy"], r["macro_f1"])
                    for r in results.get("full", [])}
    rows = []
    for cfg in configs:
        if cfg.name == "full":
            continue
        rs = results.get(cfg.name, [])
        if not rs:
            continue
        cfg_by_seed = {r.get("seed"): (r["accuracy"], r["macro_f1"]) for r in rs}
        if all(s is not None for s in list(full_by_seed) + list(cfg_by_seed)):
            common = sorted(set(full_by_seed) & set(cfg_by_seed))
            if len(common) < min(len(full_by_seed), len(cfg_by_seed)):
                print(f"[warn] {cfg.name}: seed sets differ from full "
                      f"(common n={len(common)})", flush=True)
            full_pairs = full_by_seed
        else:  # legacy entries without seed: positional pairing, min length
            common = list(range(min(len(rs), len(results.get("full", [])))))
            cfg_by_seed = {s: (rs[s]["accuracy"], rs[s]["macro_f1"])
                           for s in common}
            full_pairs = {s: (results["full"][s]["accuracy"],
                              results["full"][s]["macro_f1"])
                          for s in common}
        acc = np.asarray([cfg_by_seed[s][0] for s in common], dtype=np.float64)
        facc = np.asarray([full_pairs[s][0] for s in common], dtype=np.float64)
        f1 = np.asarray([cfg_by_seed[s][1] for s in common], dtype=np.float64)
        ff1 = np.asarray([full_pairs[s][1] for s in common], dtype=np.float64)
        a = _paired_stats(acc, facc)
        f = _paired_stats(f1, ff1)
        rows.append({
            "config": cfg.name, "n_paired": int(len(acc)),
            "acc_diff_mean": a[0], "acc_ci95_lo": a[1], "acc_ci95_hi": a[2],
            "cohens_d_acc": a[3], "t_paired_p": a[4], "wilcoxon_p": a[5],
            "shapiro_p_acc_diff": _shapiro_p(acc - facc),
            "f1_diff_mean": f[0], "f1_ci95_lo": f[1], "f1_ci95_hi": f[2],
            "cohens_d_f1": f[3], "f1_t_paired_p": f[4], "f1_wilcoxon_p": f[5],
        })
    # family-wise corrections across ALL comparisons (review M1)
    for col, prefix in (("t_paired_p", "t"), ("wilcoxon_p", "wilcoxon"),
                        ("f1_t_paired_p", "f1_t"), ("f1_wilcoxon_p", "f1_wilcoxon")):
        pv = np.asarray([r[col] for r in rows], dtype=np.float64)
        ok = np.isfinite(pv)
        holm = np.full(len(pv), np.nan)
        bh = np.full(len(pv), np.nan)
        if ok.any():
            holm[ok] = _holm_adjust(pv[ok])
            bh[ok] = _bh_adjust(pv[ok])
        for r, h, b in zip(rows, holm, bh):
            r[f"{prefix}_holm_p"] = float(h) if np.isfinite(h) else float("nan")
            r[f"{prefix}_bh_p"] = float(b) if np.isfinite(b) else float("nan")
    pd.DataFrame(rows).to_csv(out_dir / "significance.csv", index=False)

    _write_degradation_table(out_dir, configs, results)


def _write_degradation_table(out_dir: Path, configs: list[ExpConfig],
                             results: dict[str, list[dict]]) -> None:
    """degradation_table.csv (opt-in ``--degradation``): rows = degradation
    name + a leading ``clean`` row, columns = config, cells = mean+-std
    accuracy across seeds; the final column ``delta_vs_clean_pp`` is the
    accuracy drop vs clean in percentage points, averaged over configs
    (0.0 on the clean row).

    Only written when at least one run carries a ``final.degradation``
    dict; with no degradation data nothing is written, keeping merge
    output byte-identical to the historical behaviour."""
    from .degradation import DEGRADATION_NAMES  # local: keeps import graph flat

    per_cfg: dict[str, list[tuple[float, dict]]] = {}
    for cfg in configs:
        entries = []
        for r in results.get(cfg.name, []):
            d = (r.get("final") or {}).get("degradation")
            if d:
                entries.append((float(r["accuracy"]),
                                {k: float(v) for k, v in d.items()}))
        if entries:
            per_cfg[cfg.name] = entries
    if not per_cfg:
        return

    names = [n for n in DEGRADATION_NAMES
             if any(n in d for es in per_cfg.values() for _, d in es)]
    names += sorted({k for es in per_cfg.values() for _, d in es
                     for k in d} - set(names))

    cfg_names = list(per_cfg)
    rows = []
    clean_means = {c: _mean_std([a for a, _ in es])[0]
                   for c, es in per_cfg.items()}
    for name in ["clean", *names]:
        row: dict = {"degradation": name}
        deltas = []
        for c in cfg_names:
            if name == "clean":
                ms = _mean_std([a for a, _ in per_cfg[c]])
            else:
                ms = _mean_std([d[name] for _, d in per_cfg[c]
                                if name in d])
                deltas.append(ms[0] - clean_means[c])
            row[c] = _fmt(ms)
        row["delta_vs_clean_pp"] = 0.0 if name == "clean" else round(
            float(np.mean(deltas)) * 100.0, 2)
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "degradation_table.csv", index=False)


def _write_perm_null_summary(out_dir: Path, configs: list[ExpConfig],
                             results: dict[str, list[dict]]) -> None:
    """perm_null_summary.csv: observed Var_t(alpha) vs the permutation-null
    noise threshold (review M3: F2's criterion threshold was never reported).
    Threshold = null_mean + 2*null_std, averaged over seeds."""
    rows = []
    for cfg in configs:
        pns = [r["final"].get("perm_null") for r in results.get(cfg.name, [])]
        pns = [p for p in pns if p]
        if not pns:
            continue
        zs = [p["z"] for p in pns if p.get("z") is not None]  # None = degenerate
        rows.append({
            "config": cfg.name,
            "observed_var": float(np.mean([p["observed_var"] for p in pns])),
            "null_mean": float(np.mean([p["null_mean"] for p in pns])),
            "null_std": float(np.mean([p["null_std"] for p in pns])),
            "threshold_null_plus_2sd": float(np.mean(
                [p["null_mean"] + 2.0 * p["null_std"] for p in pns])),
            "z_mean": float(np.mean(zs)) if zs else float("nan"),
        })
    pd.DataFrame(rows).to_csv(out_dir / "perm_null_summary.csv", index=False)


def _write_rho_curve(out_dir: Path, diag_dir: Path,
                     slug2name: dict[str, str]) -> None:
    """rho_curve.csv: per-epoch gradient-norm ratio from the diagnostics
    JSONL epoch rows (review M3: rho must be shown sustained over training,
    not only as a terminal value)."""
    pat = re.compile(r"^(?P<slug>.+)_seed(?P<seed>\d+)\.jsonl$")
    rows = []
    for path in sorted(diag_dir.glob("*_seed*.jsonl")):
        m = pat.match(path.name)
        if not m:
            continue
        name = slug2name.get(m.group("slug"))
        if name is None:
            continue
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "epoch" in rec and rec.get("rho") is not None:
                    rows.append({"config": name, "seed": int(m.group("seed")),
                                 "epoch": int(rec["epoch"]),
                                 "rho": float(rec["rho"])})
    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "rho_curve.csv", index=False)


def merge_only(args: argparse.Namespace) -> dict:
    """No training: scan ``out_dir/diagnostics/*_seed*.jsonl``, recover each
    run's final metrics from its ``{"final": ...}`` row, and rebuild all
    tables. Lets you append extra seeds and re-aggregate without rerunning."""
    out_dir = Path(args.out_dir)
    diag_dir = out_dir / "diagnostics"
    if not diag_dir.is_dir():
        raise FileNotFoundError(f"no diagnostics directory: {diag_dir}")
    configs = build_configs(smoke=False, cell=args.cell, arch=args.arch)
    slug2name = {_slug(c.name): c.name for c in configs}
    results: dict[str, list[dict]] = {c.name: [] for c in configs}
    pat = re.compile(r"^(?P<slug>.+)_seed(?P<seed>\d+)\.jsonl$")
    n_files = n_rows = 0
    for path in sorted(diag_dir.glob("*_seed*.jsonl")):
        m = pat.match(path.name)
        if not m:
            continue
        name = slug2name.get(m.group("slug"))
        if name is None:
            print(f"[merge] skip unknown slug: {path.name}", flush=True)
            continue
        n_files += 1
        final = None
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "final" in rec:
                    final = rec["final"]
        if final is None:
            print(f"[merge] no final row in {path.name} (incomplete run?) "
                  f"-- skipped", flush=True)
            continue
        f = dict(final)
        entry = {"seed": int(m.group("seed")),
                 "accuracy": float(f.pop("accuracy")),
                 "macro_f1": float(f.pop("macro_f1")),
                 "epochs_run": f.pop("epochs_run", None),
                 "final": f}
        results[name].append(entry)
        n_rows += 1
    for name in results:
        results[name].sort(key=lambda r: r["seed"])
    seeds_found = sorted({r["seed"] for rs in results.values() for r in rs})
    empty = [c.name for c in configs if not results[c.name]]
    if empty:
        print(f"[merge] configs with no runs (excluded from tables): {empty}",
              flush=True)
    print(f"[merge] merged {n_rows}/{n_files} runs; seeds={seeds_found}",
          flush=True)
    _write_tables(out_dir, configs, results)
    _write_perm_null_summary(out_dir, configs, results)
    _write_rho_curve(out_dir, diag_dir, slug2name)

    # refresh protocol_report.json (keep dataset_report, update seed list)
    proto_path = out_dir / "protocol_report.json"
    protocol: dict = {}
    if proto_path.exists():
        with open(proto_path, "r", encoding="utf-8") as fh:
            protocol = json.load(fh)
    protocol["seeds"] = seeds_found
    protocol["merged_from_diagnostics"] = True
    with open(proto_path, "w", encoding="utf-8") as fh:
        json.dump(_to_jsonable(protocol), fh, indent=2, ensure_ascii=False)

    # per-class run/window counts for the paper's protocol table (review m9)
    pc = (protocol.get("dataset_report") or {}).get("per_class") or {}
    if pc:
        wr = set(protocol["dataset_report"].get("within_run_split_classes") or [])
        pc_rows = [{"class": k, "within_run_split": k in wr,
                    "train_runs": v["train_runs"], "val_runs": v["val_runs"],
                    "test_runs": v["test_runs"],
                    "train_windows": v["train_windows"],
                    "val_windows": v["val_windows"],
                    "test_windows": v["test_windows"]}
                   for k, v in pc.items()]
        pd.DataFrame(pc_rows).to_csv(out_dir / "per_class_table.csv", index=False)

    print(f"[done] merged tables written to {out_dir}/: results_table.csv, "
          f"remedy_table.csv, significance.csv, protocol_report.json",
          flush=True)
    return {"results": results, "out_dir": str(out_dir)}


def run(args: argparse.Namespace) -> dict:
    out_dir = Path(args.out_dir)
    diag_dir = out_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    seeds = SMOKE_SEEDS if args.smoke else list(args.seeds)
    configs = build_configs(smoke=args.smoke, cell=args.cell, arch=args.arch)
    if args.configs:
        keep = set(args.configs)
        unknown = keep - {c.name for c in configs}
        if unknown:
            raise ValueError(f"unknown config name(s): {sorted(unknown)}; "
                             f"available: {[c.name for c in configs]}")
        configs = [c for c in configs if c.name in keep]

    # ---- data ------------------------------------------------------------
    # stride default differs per dataset (64 NPPAD overlap / 128 TEP &
    # Paderborn non-overlapping); an explicit --stride always wins
    stride = args.stride if args.stride is not None else (
        128 if args.dataset in ("tep", "paderborn") else 64)
    if args.synthetic:
        if args.smoke:
            if args.dataset == "tep":
                bundle = make_synthetic_tep(n_classes=6, runs_per_class=3,
                                            n_channels=16, seed=0)
            elif args.dataset == "paderborn":
                split_unit = getattr(args, "split_unit", "run")
                bundle = make_synthetic_paderborn(n_classes=6,
                                                  runs_per_class=3,
                                                  n_samples=512, seed=0,
                                                  split_unit=split_unit)
            else:
                bundle = make_synthetic(n_classes=6, runs_per_class=3,
                                        n_channels=16, seed=0)
        else:
            bundle = (make_synthetic_tep(seed=0) if args.dataset == "tep"
                      else make_synthetic_paderborn(
                          seed=0,
                          split_unit=getattr(args, "split_unit", "run"))
                      if args.dataset == "paderborn"
                      else make_synthetic(seed=0))
        data_source = "synthetic" if args.dataset == "nppad" \
            else f"synthetic_{args.dataset}"
    elif args.dataset == "tep":
        bundle = load_tep(args.data_root, window=args.window, stride=stride,
                          runs_per_class=args.tep_runs, split_seed=42)
        data_source = f"tep:{args.data_root}"
    elif args.dataset == "paderborn":
        bundle = load_paderborn(args.data_root, window=args.window,
                                stride=stride, split_seed=42,
                                split_unit=getattr(args, "split_unit", "run"))
        data_source = f"paderborn:{args.data_root}"
    else:
        bundle = load_nppad(args.data_root, window=args.window, stride=stride)
        data_source = str(args.data_root)

    # ---- fixed-fusion controls: inject constant alphas AFTER the bundle --
    # is loaded (the bundle's class order defines the per-class index mapping)
    fixed = [c for c in configs if c.fusion in FIXED_FUSION_MODES]
    if fixed:
        if not args.fixed_alpha_file:
            raise ValueError(
                f"--fixed-alpha-file PATH (an alpha_means.json from a "
                f"train-split eval_dump) is required: selected configs "
                f"{[c.name for c in fixed]} use fusion "
                f"{sorted({c.fusion for c in fixed})}")
        with open(args.fixed_alpha_file, "r", encoding="utf-8") as fh:
            alpha_means = json.load(fh)
        configs = [inject_fixed_alphas(c, alpha_means, list(bundle.class_names))
                   for c in configs]
        print(f"[setup] injected fixed alphas from {args.fixed_alpha_file} "
              f"(split={alpha_means.get('split')}, "
              f"global={alpha_means.get('global')})", flush=True)
    elif args.fixed_alpha_file:
        print("[warn] --fixed-alpha-file given but no fixed_global/"
              "fixed_class config is selected -- ignored", flush=True)

    epochs = 2 if args.smoke else args.epochs
    print(f"[setup] data={data_source} seeds={seeds} cell={args.cell} "
          f"dataset={args.dataset} arch={args.arch} "
          f"configs={[c.name for c in configs]} epochs={epochs} "
          f"device={'cuda' if __import__('torch').cuda.is_available() else 'cpu'}",
          flush=True)

    # ---- run grid ----------------------------------------------------------
    results: dict[str, list[dict]] = {c.name: [] for c in configs}
    n_runs = len(configs) * len(seeds)
    done = 0
    for seed in seeds:
        for cfg in configs:
            done += 1
            cfg_run = replace(cfg, seed=seed, epochs=epochs,
                              patience=min(cfg.patience, epochs))
            log_path = str(diag_dir / f"{_slug(cfg.name)}_seed{seed}.jsonl")
            print(f"[{done}/{n_runs}] config={cfg.name} seed={seed} -> "
                  f"{log_path}", flush=True)
            t0 = time.time()
            res = train_one(cfg_run, bundle, log_path=log_path,
                            arch=args.arch,
                            log_epoch_indicators=args.log_epoch_indicators,
                            degradation=args.degradation)
            res["seed"] = seed  # enables seed-paired significance tests
            results[cfg.name].append(res)
            if log_path:  # persist final metrics so partial reruns can be merged
                with open(log_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"final": _to_jsonable({
                        "accuracy": res["accuracy"], "macro_f1": res["macro_f1"],
                        "epochs_run": res["epochs_run"], **res["final"]})}) + "\n")
            final = res["final"]
            print(f"    acc={res['accuracy']:.4f} macroF1={res['macro_f1']:.4f} "
                  f"epochs_run={res['epochs_run']} "
                  f"H_alpha={final['H_alpha']} rho_last={final['rho_last']} "
                  f"({time.time() - t0:.1f}s)", flush=True)

    _write_tables(out_dir, configs, results)

    # ---- protocol_report.json ----------------------------------------------
    protocol = {
        "data_source": data_source,
        "dataset": args.dataset,
        "arch": args.arch,
        "cell": args.cell,
        "split_unit": getattr(args, "split_unit", "run"),
        "smoke": bool(args.smoke),
        "seeds": seeds,
        "epochs": epochs,
        "configs": [c.name for c in configs],
        "dataset_report": bundle.report,
    }
    with open(out_dir / "protocol_report.json", "w", encoding="utf-8") as fh:
        json.dump(_to_jsonable(protocol), fh, indent=2, ensure_ascii=False)

    print(f"[done] outputs written to {out_dir}/: results_table.csv, "
          f"remedy_table.csv, significance.csv, diagnostics/*.jsonl, "
          f"protocol_report.json", flush=True)
    return {"results": results, "bundle": bundle, "out_dir": str(out_dir)}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="ATSF-DCG experiment runner")
    src = ap.add_mutually_exclusive_group(required=False)
    src.add_argument("--data-root", type=str, default=None,
                     help="dataset root (default: data/ under this repo; "
                          "see data/README.md)")
    src.add_argument("--synthetic", action="store_true",
                     help="use make_synthetic instead of NPPAD")
    ap.add_argument("--merge-only", action="store_true",
                    help="no training: rebuild the three CSV tables from "
                         "diagnostics/*_seed*.jsonl final rows (use after "
                         "appending extra seeds)")
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--smoke", action="store_true",
                    help="synthetic micro data, 2 seeds x 2 epochs, "
                         "configs full + full_r2 + full_r4_lstm only")
    ap.add_argument("--configs", type=str, nargs="+", default=None,
                    help="subset of config names to run (default: all)")
    ap.add_argument("--fixed-alpha-file", type=str, default=None,
                    help="alpha_means.json (written by eval_dump, e.g. a "
                         "train-split dump of config 'full'); required iff a "
                         "fixed_global/fixed_class config is selected")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--stride", type=int, default=None,
                    help="window stride (default: 64 for nppad, 128 for "
                         "tep/paderborn)")
    ap.add_argument("--dataset", choices=DATASETS, default="nppad",
                    help="nppad (default), tep (TEP files under data/TEP; "
                         "see data/README.md) or paderborn "
                         "(Paderborn bearing .mat tree under data/Paderborn; "
                         "see data/README.md)")
    ap.add_argument("--split-unit", choices=("run", "bearing"), default="run",
                    help="Paderborn only: run (default, current Cell D) or "
                         "bearing (held-out bearings). Default out-dir for "
                         "bearing is "
                         "output/experiments/CellD_Paderborn_ATSF_bearing")
    ap.add_argument("--arch", choices=ARCHS, default="atsf",
                    help="atsf (default, ATSFDCG) or tsnet (TSFTimesNet: "
                         "TimesNet temporal branch + same spectral/fusion/"
                         "gating machinery)")
    ap.add_argument("--cell", choices=CELLS, default="full",
                    help="full (default: 21-config grid) or replication "
                         "(7-config replication grid, +tsnet_vanilla under "
                         "--arch tsnet)")
    ap.add_argument("--tep-runs", type=str, default="98,21,21",
                    help="TEP only: exact train,val,test runs per class "
                         "(default 98,21,21 of the 500 available)")
    ap.add_argument("--log-epoch-indicators", action="store_true",
                    help="opt-in: after each epoch's validation, log "
                         "h_alpha (routing entropy) and s_tau (gate "
                         "saturation at tau=0.9) on a fixed deterministic "
                         "first-256-window validation subset into the epoch "
                         "record (nulls for configs without fusion/gating)")
    ap.add_argument("--degradation", action="store_true",
                    help="opt-in: after the final test evaluation, evaluate "
                         "the trained model on the test set under 7 sensor "
                         "degradations (applied post-normalization, in "
                         "per-channel-std units; seeded per run seed) and "
                         "store final['degradation']; merge then also "
                         "writes degradation_table.csv")
    ap.add_argument("--out-dir", type=str, default=None,
                    help="output directory (default: "
                         "output/experiments/<mapped cell folder>)")
    args = ap.parse_args(argv)
    if args.smoke:
        args.synthetic = True  # smoke always runs on synthetic micro data
    try:
        args.tep_runs = tuple(int(v) for v in str(args.tep_runs).split(","))
        if len(args.tep_runs) != 3:
            raise ValueError
    except ValueError:
        ap.error(f"--tep-runs must be 'train,val,test' integers, got "
                 f"{args.tep_runs!r}")
    if args.split_unit == "bearing" and args.dataset != "paderborn":
        ap.error("--split-unit bearing requires --dataset paderborn")
    if args.out_dir is None:
        from .paths import default_out_dir
        if args.split_unit == "bearing":
            args.out_dir = str(Path("output") / "experiments"
                               / "CellD_Paderborn_ATSF_bearing")
        else:
            args.out_dir = default_out_dir(args.cell, args.dataset, args.arch)
    if args.merge_only:
        merge_only(args)
        return
    if not args.synthetic and not args.data_root:
        from .paths import DATA_ROOT, default_data_root
        args.data_root = str(default_data_root(args.dataset))
        if not Path(args.data_root).exists():
            ap.error(
                f"no data at {args.data_root}. Download {args.dataset} into "
                f"{DATA_ROOT}/ (see data/README.md), or pass --data-root / "
                f"--synthetic")
    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
