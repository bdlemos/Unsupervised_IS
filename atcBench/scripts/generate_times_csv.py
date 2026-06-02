#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import os
import re
from pathlib import Path
from collections import defaultdict

PATTERN_DEFAULT = "resources/output/*/**/measures.fold_*.json"

def find_top_dir(parts):
    # Find the first ancestor folder that looks like dataset_method (contains an underscore)
    for p in parts[::-1]:
        if "_" in p and p.lower() != "outputs":
            return p
    # fallback to the parent folder name
    return parts[-1] if parts else ""

def load_selection_csv(path: Path) -> dict:
    """Load selection CSV and return mapping (dataset, method) -> (reduction string, time_mean float).
    """
    mapping = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return mapping

        ds_field = next((fn for fn in reader.fieldnames if 'dataset' in fn.lower()), None)
        method_field = next((fn for fn in reader.fieldnames if 'method' in fn.lower()), None)
        red_field = next((fn for fn in reader.fieldnames if 'reduct' in fn.lower() or 'rate' in fn.lower() or 'select' in fn.lower()), None)
        time_field = next((fn for fn in reader.fieldnames if 'time_mean' in fn.lower()), None)
        name_field = next((fn for fn in reader.fieldnames if fn.lower() in ('name', 'dataset_method', 'id')), None)

        for row in reader:
            ds = ''
            mt = ''
            red = ''
            time_val = 0.0

            if ds_field and method_field:
                ds = (row.get(ds_field) or '').strip()
                mt = (row.get(method_field) or '').strip()
            elif name_field:
                name = (row.get(name_field) or '').strip()
                parts = name.split('_', 1)
                ds = parts[0] if parts else ''
                mt = parts[1] if len(parts) > 1 else ''
            
            if red_field:
                red = (row.get(red_field) or '').strip()
            if time_field:
                try:
                    time_val = float(row.get(time_field))
                except ValueError:
                    time_val = 0.0

            if not ds and not mt:
                continue
            if (mt or '').lower() == 'none':
                mt = ''
            mapping[(ds, mt)] = (red, time_val)

    return mapping

def main():
    p = argparse.ArgumentParser(description="Generate CSV of total time (IS + model) from outputs")
    p.add_argument("--pattern", default=None, help="glob pattern to find measures (default: resources/output/*/**/measures.fold_*.json)")
    p.add_argument("--selection-csv", default="/home/bernardo/projects/unsupervised-is/resources/outsel/selection_summary.csv", help="CSV with selection reduction and time values")
    p.add_argument("-o", "--output", default="times_from_outputs.csv", help="output CSV path")
    args = p.parse_args()
    pattern = args.pattern if args.pattern else PATTERN_DEFAULT

    paths = sorted(glob.glob(pattern, recursive=True))
    if not paths:
        raise SystemExit(f"No files found for pattern: {pattern}")

    # data[(dataset,method,model)][fold] = time_train
    data = defaultdict(dict)
    fold_set = set()

    fold_re = re.compile(r"fold_(\d+)\.json$")

    for path in paths:
        m = fold_re.search(path)
        if not m:
            continue
        fold = int(m.group(1))
        fold_set.add(fold)

        rel = os.path.relpath(os.path.dirname(path), "resources/output")
        parts = rel.split(os.sep) if rel != "." else [os.path.basename(os.path.dirname(path))]
        model = parts[0] if parts else ""
        top = find_top_dir(parts)

        if "_" in top:
            dataset, method = top.rsplit("_", 1)
        else:
            dataset, method = top, ""

        with open(path, "r", encoding="utf-8") as f:
            j = json.load(f)

        if "time_train" not in j:
            continue

        try:
            val = float(j["time_train"])
        except Exception:
            continue

        data[(dataset, method, model)][fold] = val

    if not data:
        raise SystemExit("No valid time_train values found in matched files")

    folds = sorted(fold_set)

    # same format, but columns named time0...timeN
    time_cols = [f"time{f}" for f in folds]
    header = ["dataset", "method", "reduction", "class"] + time_cols + ["time_avg_ic"]

    out_path = Path(args.output)
    selection_map = {}
    sel_path = Path(args.selection_csv)
    if sel_path.exists():
        try:
            selection_map = load_selection_csv(sel_path)
            print(f"Loaded selection CSV with {len(selection_map)} entries from: {sel_path}")
        except Exception as exc:
            print(f"Warning: failed to load selection CSV: {exc}")
    else:
        print(f"Selection CSV not found: {sel_path}")

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        keys = sorted(data.keys(), key=lambda x: (x[0] or "", x[1] or "", x[2] or ""))
        for dataset, method, model in keys:
            sel_data = selection_map.get((dataset, method)) or selection_map.get((dataset, method or "None")) or ("", 0.0)
            reduction = sel_data[0]
            time_is = sel_data[1]

            print(f"Processing: dataset={dataset}, method={method}, model={model}, IS_time={time_is:.2f}")
            row = [dataset, method, reduction, model]
            folds_map = data[(dataset, method, model)]
            
            for fnum in folds:
                if fnum in folds_map:
                    total_time = folds_map[fnum] + time_is
                    row.append(f"{total_time:.2f}")
                else:
                    row.append("")
            
            row.append("")
            writer.writerow(row)

    print(f"Wrote CSV to: {out_path}")

if __name__ == "__main__":
    main()
