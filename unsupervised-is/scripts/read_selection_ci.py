#!/usr/bin/env python3
"""Aggregate selection outputs and compute 95% CI for time and reduction.

Searches `resources/outsel/**/saida_*.json` and prints and saves summaries.
Writes `selection_summary.json` and `selection_summary.csv` under `resources/outsel`.
Uses Student's t for small samples when `scipy` is available; falls back to z=1.96 otherwise.
"""
from pathlib import Path
import json
import os
import math
import statistics
import csv
import warnings

ROOT = Path(__file__).resolve().parent.parent
OUTSEL_ROOT = ROOT / "resources" / "outsel"
OUT_JSON = OUTSEL_ROOT / "selection_summary.json"
OUT_CSV = OUTSEL_ROOT / "selection_summary.csv"

# ─── Projeto 1: unsupervised-is ──────────────────────────────────────────────

try:
    from scipy.stats import t as student_t
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


def critical_value_95(n):
    if n <= 1:
        return None
    alpha = 0.05
    if HAVE_SCIPY:
        return float(student_t.ppf(1 - alpha / 2, df=n - 1))
    if n >= 30:
        return 1.96
    warnings.warn("scipy not available: falling back to z=1.96 for CI (small-n t unavailable)")
    return 1.96


def ci95(values):
    n = len(values)
    mean = statistics.mean(values)
    std = statistics.stdev(values) if n > 1 else 0.0
    crit = critical_value_95(n)
    if crit is None:
        margin = 0.0
    else:
        margin = crit * (std / math.sqrt(n)) if n > 1 else 0.0
    return mean, margin, n


def main():
    if not OUTSEL_ROOT.exists():
        raise SystemExit(f"Path not found: {OUTSEL_ROOT}")

    entries = {}
    # find all saida_*.json files under outsel (supports multiple layouts)
    for fp in sorted(OUTSEL_ROOT.rglob("saida_*.json")):
        if not fp.is_file():
            continue
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"Failed to read {fp}: {e}")
            continue

        dataset_name = fp.parent.name
        entries.setdefault(dataset_name, {})

        method = data.get("method") or fp.stem.replace("saida_", "")
        info = data.get("info", {})
        times = info.get("time_for_reduce") or []
        redu = info.get("reducion") or []

        if len(times) == 0 and "time_mean" in info:
            times = [info.get("time_mean")]
        if len(redu) == 0 and "reduction_mean" in info:
            redu = [info.get("reduction_mean")]

        if not times and not redu:
            print(f"No time/reduction data in {fp}")
            continue

        t_mean, t_err, tn = ci95(times) if times else (None, None, 0)
        r_mean, r_err, rn = ci95(redu) if redu else (None, None, 0)

        entries[dataset_name][method] = {
            "time": {"mean": t_mean, "error": t_err, "n": tn},
            "reduction": {"mean": r_mean, "error": r_err, "n": rn},
        }

    # Print to stdout
    for dataset in sorted(entries.keys()):
        print(f"{dataset}:")
        methods = entries[dataset]
        for method in sorted(methods.keys(), key=lambda m: (methods[m]["time"]["mean"] if methods[m]["time"]["mean"] is not None else -1)):
            tm = methods[method]["time"]
            rm = methods[method]["reduction"]
            t_str = f"{tm['mean']:.3f} ±{tm['error']:.3f} (n={tm['n']})" if tm["mean"] is not None else "n/a"
            r_str = f"{rm['mean']:.4f} ±{rm['error']:.4f} (n={rm['n']})" if rm["mean"] is not None else "n/a"
            print(f"    - {method}: time={t_str}  reduction={r_str}")
        print()

    # Save JSON
    OUTSEL_ROOT.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as jf:
        json.dump(entries, jf, indent=2)

    # Save CSV
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as cf:
        writer = csv.writer(cf)
        writer.writerow(["dataset", "method", "time_mean", "time_error", "time_n", "reduction_mean", "reduction_error", "reduction_n"])
        for dataset in sorted(entries.keys()):
            for method in sorted(entries[dataset].keys()):
                tm = entries[dataset][method]["time"]
                rm = entries[dataset][method]["reduction"]
                writer.writerow([
                    dataset,
                    method,
                    tm["mean"], tm["error"], tm["n"],
                    rm["mean"], rm["error"], rm["n"],
                ])

    print(f"Wrote: {OUT_JSON}")
    print(f"Wrote: {OUT_CSV}")


if __name__ == "__main__":
    main()
