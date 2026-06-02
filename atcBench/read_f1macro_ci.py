#!/usr/bin/env python3
import glob
import json
import math
import statistics
import os
from pathlib import Path

PATTERN = "resources/output/roberta/**/measures.fold_*.json"


def main() -> None:
    paths = sorted(glob.glob(PATTERN, recursive=True))
    if not paths:
        raise SystemExit(f"No files found for pattern: {PATTERN}")

    # Aggregate by dataset and method. Directories are named "dataset_method".
    dataset_groups: dict[str, dict[str, list[float]]] = {}
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "macro" not in data:
            raise SystemExit(f"Missing 'macro' field in: {path}")

        rel = os.path.relpath(os.path.dirname(path), "resources/output/roberta")
        top = Path(rel).parts[0] if rel != "." and Path(rel).parts else os.path.basename(os.path.dirname(path))

        if "_" in top:
            dataset, method = top.split("_", 1)
        else:
            dataset, method = top, ""

        dataset_groups.setdefault(dataset, {}).setdefault(method, []).append(float(data["macro"]))

    # Print formatted output per dataset
    for dataset in sorted(dataset_groups.keys()):
        print(f"{dataset}:")
        methods = dataset_groups[dataset]
        for method in sorted(methods.keys(), 
                             key=lambda mean: statistics.mean(methods[mean]) if methods[mean] else float("-inf"), 
                             reverse=True):
            values = methods[method]
            n = len(values)
            mean = statistics.mean(values)
            std = statistics.stdev(values) if n > 1 else 0.0
            margin = 1.96 * (std / math.sqrt(n)) if n > 1 else 0.0
            ci_low = mean - margin
            ci_high = mean + margin
            method_label = method if method else "(unknown)"
            print(f"    - {method_label}: {mean:.6f} 95%_CI=[{ci_low:.6f}, {ci_high:.6f}]")
        print()


if __name__ == "__main__":
    main()
