#!/usr/bin/env python3
"""Download + extract the Paderborn KAt bearing dataset (32 bearings).

Idempotent: already-extracted bearings are skipped, partial downloads resume
from scratch only for the interrupted bearing. Safe to re-run any number of
times until it prints ALL DONE.

Usage:
    python3 download_paderborn.py              # → <repo>/data/Paderborn
    python3 download_paderborn.py /path/to/Paderborn
    python3 download_paderborn.py --keep-rar

Requires one of: unrar / unar / 7z   (apt install -y unrar)
Source: https://groups.uni-paderborn.de/kat/BearingDataCenter/
Dataset: Lessmeier et al. 2016, doi:10.36001/phme.2016.v3i1.1577 (CC BY-NC 4.0)
"""
import argparse
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

BASE_URL = "https://groups.uni-paderborn.de/kat/BearingDataCenter/{code}.rar"

BEARINGS = [
    # 6 healthy
    "K001", "K002", "K003", "K004", "K005", "K006",
    # 12 outer-ring damage (KA01/03/05-09 artificial; KA04/15/16/22/30 real)
    "KA01", "KA03", "KA04", "KA05", "KA06", "KA07", "KA08", "KA09",
    "KA15", "KA16", "KA22", "KA30",
    # 11 inner-ring damage (KI01/03/05/07/08 artificial; KI04/14/16/17/18/21 real)
    "KI01", "KI03", "KI04", "KI05", "KI07", "KI08",
    "KI14", "KI16", "KI17", "KI18", "KI21",
    # 3 compound (both rings, real)
    "KB23", "KB24", "KB27",
]

EXPECTED_RUNS_PER_BEARING = 80  # 4 settings x 20 runs


def find_extractor() -> list[str]:
    """Return the extraction command template, or die with instructions."""
    if shutil.which("unrar"):
        return ["unrar", "x", "-o+", "{rar}", "{dst}/"]
    if shutil.which("unar"):
        return ["unar", "-o", "{dst}", "{rar}"]
    if shutil.which("7z"):
        return ["7z", "x", "-y", f"-o{'{dst}'}", "{rar}"]
    sys.exit("ERROR: no extractor found. Install one: "
             "apt install -y unrar   (or: unar, p7zip-full)")


def download(url: str, dst: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    tmp = dst.with_suffix(".part")
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1024 * 1024)
    tmp.rename(dst)


def mat_count(bdir: Path) -> int:
    return len(list(bdir.rglob("*.mat")))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default=None,
                    help="target directory (default: <repo>/data/Paderborn)")
    ap.add_argument("--keep-rar", action="store_true",
                    help="keep .rar archives after extraction (default: delete)")
    ap.add_argument("--no-download", action="store_true",
                    help="only extract .rar files already present; never download "
                         "(use after uploading locally-downloaded archives)")
    args = ap.parse_args()

    root = Path(args.root) if args.root else (
        Path(__file__).resolve().parents[1] / "data" / "Paderborn")
    root.mkdir(parents=True, exist_ok=True)
    extract_cmd = find_extractor()

    ok, failed = [], []
    for i, code in enumerate(BEARINGS, 1):
        bdir = root / code
        n = mat_count(bdir) if bdir.is_dir() else 0
        if n == EXPECTED_RUNS_PER_BEARING:
            print(f"[{i:2d}/32] {code}: already extracted ({n} .mat) — skip")
            ok.append(code)
            continue

        rar = root / f"{code}.rar"
        try:
            if not rar.exists():
                if args.no_download:
                    raise RuntimeError("rar not present and --no-download set")
                print(f"[{i:2d}/32] {code}: downloading ...", flush=True)
                download(BASE_URL.format(code=code), rar)
            size_mb = rar.stat().st_size / 1e6
            if size_mb < 100:  # archives are 152-178 MB; smaller = truncated
                raise RuntimeError(f"suspicious size {size_mb:.1f} MB")

            print(f"[{i:2d}/32] {code}: extracting ({size_mb:.0f} MB) ...",
                  flush=True)
            # archives already contain a top-level {code}/ folder, so extract
            # into root (extracting into bdir would give {code}/{code}/ nested)
            cmd = [c.format(rar=str(rar), dst=str(root)) for c in extract_cmd]
            subprocess.run(cmd, check=True, capture_output=True)

            n = mat_count(bdir)
            if n != EXPECTED_RUNS_PER_BEARING:
                raise RuntimeError(f"expected {EXPECTED_RUNS_PER_BEARING} .mat, "
                                   f"got {n}")
            if not args.keep_rar:
                rar.unlink()
            ok.append(code)
            print(f"[{i:2d}/32] {code}: OK ({n} .mat)", flush=True)
        except Exception as e:  # noqa: BLE001 - report and continue with next
            print(f"[{i:2d}/32] {code}: FAILED — {e}", flush=True)
            failed.append(code)

    total = sum(mat_count(root / c) for c in BEARINGS if (root / c).is_dir())
    print(f"\n=== summary: {len(ok)}/32 bearings OK, {total} .mat files "
          f"(expect 2560) ===")
    if failed:
        print("FAILED bearings (re-run this script to retry):",
              ", ".join(failed))
        sys.exit(1)
    print("ALL DONE")


if __name__ == "__main__":
    main()
