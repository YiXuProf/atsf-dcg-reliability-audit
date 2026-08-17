"""CLI wiring tests for the replication cell (plan v6 Stage 2c/3).

- ``build_configs(cell="replication", arch=...)`` builds the 7 (+1 under
  arch="tsnet") expected config names, in order.
- REGRESSION GUARD: the DEFAULT ``build_configs()`` / ``build_configs(
  smoke=True)`` output is byte-identical (repr) to the pre-change baseline
  captured from the unmodified code — names, order and every ExpConfig
  field (the baseline strings below were generated on master before the
  replication wiring; both machines are CPU, so device='cpu').
- out-dir resolution: default invocation ->
  ``output/experiments/nppad_atsf_full``; other (cell, dataset, arch)
  triples map to ``output/experiments/<Cell*>``; explicit --out-dir always wins.
- end-to-end: ``--cell replication --dataset tep --arch tsnet --synthetic``
  trains the selected configs (incl. tsnet_vanilla), writes diagnostics +
  tables + protocol_report with dataset/arch/cell fields, and
  ``--merge-only`` (same flags) rebuilds the tables per cell.

Run from the project root:  ``python tests/test_cli_replication.py``
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from atsf_dcg.run_experiments import build_configs  # noqa: E402

REPLICATION_ATSF = ["full", "w/o_spectral", "w/o_temporal", "w/o_fusion",
                    "w/o_gating", "full_r1", "full_r2_gumbel"]
REPLICATION_TSNET = REPLICATION_ATSF + ["tsnet_vanilla"]

# --- pre-change baseline reprs (master, before the replication wiring) ----
_DEFAULT_REPR_PATH = Path(__file__).with_name("baseline_build_configs.txt")
_SMOKE_REPR_PATH = Path(__file__).with_name("baseline_build_configs_smoke.txt")


def _config_builder_checks() -> None:
    names_a = [c.name for c in build_configs(cell="replication",
                                             arch="atsf")]
    names_t = [c.name for c in build_configs(cell="replication",
                                             arch="tsnet")]
    assert names_a == REPLICATION_ATSF, names_a
    assert names_t == REPLICATION_TSNET, names_t
    vanilla = build_configs(cell="replication", arch="tsnet")[-1]
    assert vanilla.use_spectral is False and vanilla.use_temporal is True
    assert vanilla.fusion == "none" and vanilla.gating == "none"
    assert vanilla.r1_balanced is False and vanilla.r2_load_balanced is False
    # replication ExpConfig semantics match their full-grid namesakes
    full_grid = {c.name: c for c in build_configs()}
    for c in build_configs(cell="replication", arch="atsf"):
        ref = full_grid[c.name]
        assert repr(c) == repr(ref), (c.name, repr(c), repr(ref))
    # smoke subset of the replication cell
    smoke_t = [c.name for c in build_configs(smoke=True, cell="replication",
                                             arch="tsnet")]
    assert smoke_t == ["full", "full_r2_gumbel", "tsnet_vanilla"], smoke_t
    smoke_a = [c.name for c in build_configs(smoke=True, cell="replication",
                                             arch="atsf")]
    assert smoke_a == ["full", "full_r2_gumbel"], smoke_a
    print("[test] OK: replication cell builds 7 (+tsnet_vanilla) configs; "
          "ExpConfig semantics match the full grid.")


def _regression_guard() -> None:
    default_now = repr(build_configs())
    smoke_now = repr(build_configs(smoke=True))
    default_base = _DEFAULT_REPR_PATH.read_text()
    smoke_base = _SMOKE_REPR_PATH.read_text()
    assert default_now == default_base, (
        "default build_configs() output CHANGED (byte-compare failed); "
        "the replication wiring must not touch the default grid")
    assert smoke_now == smoke_base, "default build_configs(smoke=True) changed"
    print("[test] OK: regression guard — default build_configs() byte-"
          "identical to pre-change baseline (names, order, all fields).")


def _run(cmd: list[str], cwd: Path = ROOT, env_extra: dict | None = None
         ) -> subprocess.CompletedProcess:
    print(f"[test] running: {' '.join(cmd)}")
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                          env=env)
    sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"command failed ({proc.returncode}): {' '.join(cmd)}")
    return proc


def _end_to_end() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="atsf_dcg_replication_"))
    env = {"PYTHONPATH": str(ROOT)}

    # ---- default out-dir resolution: results_<cell>_<dataset>_<arch> ----
    _run([sys.executable, "-m", "atsf_dcg.run_experiments",
          "--cell", "replication", "--dataset", "tep", "--arch", "tsnet",
          "--synthetic", "--epochs", "1", "--seeds", "42",
          "--configs", "full", "tsnet_vanilla"],
         cwd=tmp, env_extra=env)
    out_dir = tmp / "output" / "experiments" / "CellC_TEP_TimesNet"
    assert out_dir.is_dir(), f"expected default out dir {out_dir}"
    diag = out_dir / "diagnostics"
    for slug in ("full", "tsnet_vanilla"):
        f = diag / f"{slug}_seed42.jsonl"
        assert f.is_file(), f"missing {f}"
        finals = [json.loads(l)["final"] for l in
                  f.read_text().strip().splitlines()
                  if "final" in json.loads(l)]
        assert len(finals) == 1
        assert 0.0 <= float(finals[0]["accuracy"]) <= 1.0
    rt = pd.read_csv(out_dir / "results_table.csv")
    assert list(rt["config"]) == ["full", "tsnet_vanilla"], list(rt["config"])
    proto = json.loads((out_dir / "protocol_report.json").read_text())
    assert proto["dataset"] == "tep" and proto["arch"] == "tsnet"
    assert proto["cell"] == "replication"
    assert proto["dataset_report"]["dataset"] == "TEP"
    # tsnet_vanilla is single-branch: no alpha diagnostics
    vf = [json.loads(l)["final"] for l in
          (diag / "tsnet_vanilla_seed42.jsonl").read_text().strip()
          .splitlines() if "final" in json.loads(l)][0]
    assert vf["H_alpha"] is None and vf["perm_null"] is None
    print("[test] OK: end-to-end replication run (tep x tsnet, synthetic) "
          f"-> {out_dir.name}/ with diagnostics + tables + protocol report.")

    # ---- merge-only per cell (same flags, no training) --------------------
    _run([sys.executable, "-m", "atsf_dcg.run_experiments", "--merge-only",
          "--cell", "replication", "--dataset", "tep", "--arch", "tsnet"],
         cwd=tmp, env_extra=env)
    rt_m = pd.read_csv(out_dir / "results_table.csv")
    assert list(rt_m["config"]) == ["full", "tsnet_vanilla"]
    for col in ("accuracy", "macro_f1"):
        assert (rt_m[col] == rt[col]).all(), "merge-only table drift"
    proto_m = json.loads((out_dir / "protocol_report.json").read_text())
    assert proto_m["merged_from_diagnostics"] is True
    print("[test] OK: --merge-only works per cell (replication/tep/tsnet).")

    # ---- explicit --out-dir always wins; default invocation -> results ----
    out2 = tmp / "custom"
    _run([sys.executable, "-m", "atsf_dcg.run_experiments",
          "--cell", "replication", "--dataset", "tep", "--arch", "tsnet",
          "--synthetic", "--epochs", "1", "--seeds", "42",
          "--configs", "tsnet_vanilla", "--out-dir", str(out2)],
         cwd=tmp, env_extra=env)
    assert (out2 / "results_table.csv").is_file()
    assert not (tmp / "results").exists(), \
        "no plain results/ dir should be created by these invocations"
    # bad --tep-runs rejected
    proc = subprocess.run(
        [sys.executable, "-m", "atsf_dcg.run_experiments", "--synthetic",
         "--tep-runs", "98,21", "--configs", "full"],
        cwd=tmp, capture_output=True, text=True, env={**os.environ, **env})
    assert proc.returncode != 0 and "--tep-runs" in proc.stderr
    print("[test] OK: explicit --out-dir wins; --tep-runs validated.")


def main() -> None:
    t0 = time.time()
    _config_builder_checks()
    _regression_guard()
    _end_to_end()
    print(f"[test_cli_replication] ALL OK ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
