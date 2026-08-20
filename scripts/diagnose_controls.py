"""Custom paired diagnostics for the two reviewer control experiments.

Standalone: reads output/experiments/nppad_atsf_full/diagnostics/*.jsonl
(per-seed final metrics), runs the planned contrasts that significance.csv
does NOT cover (significance.csv only tests everything vs "full").

Planned contrasts
  Experiment A (F1 decisive test, R1-repaired state):
    A1  r1_w/o_spectral  vs full_r1      (acc, F1)  <- THE decisive test
    A2  r1_w/o_spectral  vs w/o_spectral (acc)      (R1 conditioning change?)
  Experiment B (F2 fixed-mean controls):
    B1  full_fixed_global vs full            (acc, F1)
    B2  full_fixed_class  vs full            (acc, F1)
    B3  full_fixed_class  vs full_fixed_global (acc)  (class-diff value)
    B4  full_fixed_global vs w/o_fusion      (acc)  (own mean vs uniform 0.5)
  TOST equivalence tests (bound eps, default 0.5 pp):
    full_fixed_global vs full, w/o_fusion vs full, full_r1 vs full

Usage (from the repository root)::
    python scripts/diagnose_controls.py
    python scripts/diagnose_controls.py --eps 0.005
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atsf_dcg.paths import EXPERIMENTS_ROOT, TABLES_ROOT  # noqa: E402

try:  # preferred: reuse the pipeline's own slug (run from the project root)
    from atsf_dcg.run_experiments import _slug
except ImportError:  # fallback: byte-identical local replica
    import re as _re

    def _slug(name: str) -> str:
        return _re.sub(r"[^A-Za-z0-9_.-]+", "_",
                       name.replace("w/o", "wo")).strip("_")


# ------------------------------------------------------------------ jsonl read
def load_per_seed(diag_dir: Path) -> dict[str, dict[int, dict]]:
    """{config_name: {seed: {"accuracy":..,"macro_f1":..}}} from diag jsonl.

    Row schema matches run_experiments.merge_only: each line is a dict;
    the last line containing a "final" key holds the terminal metrics as
    row["final"]["accuracy"] / row["final"]["macro_f1"]. The seed is NOT
    stored in the row -- it is parsed from the filename
    ``{slug}_seed{seed}.jsonl``.
    """
    import re
    pat = re.compile(r"^(?P<slug>.+)_seed(?P<seed>\d+)\.jsonl$")
    out: dict[str, dict[int, dict]] = {}
    for p in sorted(diag_dir.glob("*_seed*.jsonl")):
        m = pat.match(p.name)
        if not m:
            continue
        seed = int(m.group("seed"))
        final = None
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if "final" in row:
                    final = row["final"]
        if final is None:
            print(f"[warn] no final row in {p.name} (incomplete run?) -- skipped")
            continue
        out.setdefault(p.name, {})[seed] = {
            "accuracy": float(final["accuracy"]),
            "macro_f1": float(final["macro_f1"]),
        }
    # map file slug -> config name via build_configs slug table
    try:
        from atsf_dcg.run_experiments import build_configs
        slug2name = {_slug(c.name): c.name for c in build_configs()}
    except Exception:
        # offline fallback: the 20 known config names
        names = ["full", "w/o_spectral", "w/o_temporal", "w/o_fusion",
                 "w/o_dynamic_gating", "w/o_gating", "full_r1", "full_r3_stft",
                 "full_r3_sinc", "full_r1_r3", "full_r2", "full_r2_gumbel",
                 "full_r4_sparsemax", "full_r4_entmax", "full_r4_lstm",
                 "full_r1_r2_r3", "full_all", "r1_w/o_spectral",
                 "full_fixed_global", "full_fixed_class"]
        slug2name = {_slug(n): n for n in names}
    named: dict[str, dict[int, dict]] = {}
    for fname, seeds in out.items():
        slug = fname.rsplit("_seed", 1)[0]
        name = slug2name.get(slug, slug)
        named.setdefault(name, {}).update(seeds)
    return named


# ------------------------------------------------------------------ statistics
def paired(a: np.ndarray, b: np.ndarray) -> dict:
    """Paired stats for a - b (per-seed)."""
    d = a - b
    n = d.size
    md = float(d.mean())
    sd = float(d.std(ddof=1))
    se = sd / np.sqrt(n)
    ci = stats.t.ppf(0.975, n - 1) * se
    t_p = float(stats.ttest_rel(a, b).pvalue)
    try:
        w_p = float(stats.wilcoxon(d).pvalue)
    except ValueError:  # all-zero differences
        w_p = float("nan")
    dz = md / sd if sd > 0 else float("nan")
    shapiro = float(stats.shapiro(d).pvalue) if n >= 3 else float("nan")
    return {"n": n, "mean_diff": md, "ci95_lo": md - ci, "ci95_hi": md + ci,
            "cohens_dz": dz, "t_p": t_p, "wilcoxon_p": w_p, "shapiro_p": shapiro}


def tost(a: np.ndarray, b: np.ndarray, eps: float) -> dict:
    """TOST equivalence test on paired diffs, bound +-eps (same units as data)."""
    d = a - b
    n = d.size
    md, sd = float(d.mean()), float(d.std(ddof=1))
    se = sd / np.sqrt(n)
    t1 = (md + eps) / se          # H0: md <= -eps
    t2 = (eps - md) / se          # H0: md >= +eps
    p1 = 1.0 - stats.t.cdf(t1, n - 1)
    p2 = 1.0 - stats.t.cdf(t2, n - 1)
    p = max(p1, p2)
    return {"eps": eps, "mean_diff": md, "tost_p": p,
            "equivalent_at_eps": bool(p < 0.05)}


def holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = [None] * m
    runmax = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * pvals[idx])
        runmax = max(runmax, val)
        adj[idx] = runmax
    return adj


# ---------------------------------------------------------------------- main
CONTRASTS = [
    ("A1", "r1_w/o_spectral", "full_r1", ("accuracy", "macro_f1")),
    ("A2", "r1_w/o_spectral", "w/o_spectral", ("accuracy",)),
    ("B1", "full_fixed_global", "full", ("accuracy", "macro_f1")),
    ("B2", "full_fixed_class", "full", ("accuracy", "macro_f1")),
    ("B3", "full_fixed_class", "full_fixed_global", ("accuracy",)),
    ("B4", "full_fixed_global", "w/o_fusion", ("accuracy",)),
]
TOST_PAIRS = [("full_fixed_global", "full"),
              ("w/o_fusion", "full"),
              ("full_r1", "full")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag-dir",
                    default=str(EXPERIMENTS_ROOT / "nppad_atsf_full" / "diagnostics"))
    ap.add_argument("--eps", type=float, default=0.005,
                    help="TOST equivalence bound in accuracy units (default 0.005 = 0.5pp)")
    ap.add_argument("--out",
                    default=str(TABLES_ROOT / "reviewer_controls" / "custom_paired.csv"))
    args = ap.parse_args()

    data = load_per_seed(Path(args.diag_dir))
    print(f"[setup] configs found: {len(data)}; seeds per config: "
          f"{sorted({len(v) for v in data.values()})}")

    rows = []
    for tag, cfg_a, cfg_b, metrics in CONTRASTS:
        for metric in metrics:
            if cfg_a not in data or cfg_b not in data:
                print(f"[skip] {tag} {metric}: missing {cfg_a} or {cfg_b}")
                continue
            seeds = sorted(set(data[cfg_a]) & set(data[cfg_b]))
            a = np.array([data[cfg_a][s][metric] for s in seeds])
            b = np.array([data[cfg_b][s][metric] for s in seeds])
            st = paired(a, b)
            rows.append({"contrast": tag, "metric": metric,
                         "a": cfg_a, "b": cfg_b,
                         "a_mean": float(a.mean()), "b_mean": float(b.mean()),
                         **st})
            print(f"[{tag}] {metric:8s} {cfg_a:18s} vs {cfg_b:18s} "
                  f"diff={st['mean_diff']*100:+.2f}pp "
                  f"CI95=[{st['ci95_lo']*100:+.2f},{st['ci95_hi']*100:+.2f}] "
                  f"d={st['cohens_dz']:+.2f} "
                  f"t_p={st['t_p']:.2e} w_p={st['wilcoxon_p']:.2e}")

    # Holm within the planned-contrast family (acc family / F1 family)
    df = pd.DataFrame(rows)
    for metric in ("accuracy", "macro_f1"):
        sub = df[df.metric == metric]
        if len(sub):
            df.loc[sub.index, "holm_p_t"] = holm(sub["t_p"].tolist())
            df.loc[sub.index, "holm_p_w"] = holm(
                [p if np.isfinite(p) else 1.0 for p in sub["wilcoxon_p"]])

    print("\n[tost] equivalence tests (eps = {:.1f} pp)".format(args.eps * 100))
    tost_rows = []
    for cfg_a, cfg_b in TOST_PAIRS:
        if cfg_a not in data or cfg_b not in data:
            continue
        seeds = sorted(set(data[cfg_a]) & set(data[cfg_b]))
        a = np.array([data[cfg_a][s]["accuracy"] for s in seeds])
        b = np.array([data[cfg_b][s]["accuracy"] for s in seeds])
        r = tost(a, b, args.eps)
        tost_rows.append({"a": cfg_a, "b": cfg_b, **r})
        verdict = ("EQUIVALENT within eps" if r["equivalent_at_eps"]
                   else "NOT shown equivalent (inconclusive)")
        print(f"  {cfg_a:18s} vs {cfg_b:18s} diff={r['mean_diff']*100:+.2f}pp "
              f"tost_p={r['tost_p']:.4f} -> {verdict}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    pd.DataFrame(tost_rows).to_csv(out.with_name("custom_tost.csv"), index=False)
    print(f"\n[done] wrote {out} and {out.with_name('custom_tost.csv')}")


if __name__ == "__main__":
    main()
