"""
Generate a CSV file with fold macro values from measures JSON files in outputs.
python scripts/generate_results_csv.py --pattern "resources/output/*/**/measures.fold_*.json" -o "Resultados-IS - effect-fold.fromresources.csv"
"""

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
    """Load selection CSV and return mapping (dataset, method) -> reduction string.

    Heuristics: looks for columns named like 'dataset' and 'method', and a reduction-like
    column containing 'reduct' or 'rate' or 'select'. If not found, tries 'name' column
    containing 'dataset_method'.
    """
    mapping = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return mapping

        fieldnames = [fn.lower() for fn in reader.fieldnames]
        ds_field = next((fn for fn in reader.fieldnames if 'dataset' in fn.lower()), None)
        method_field = next((fn for fn in reader.fieldnames if 'method' in fn.lower()), None)
        red_field = next((fn for fn in reader.fieldnames if 'reduct' in fn.lower() or 'rate' in fn.lower() or 'select' in fn.lower()), None)
        name_field = next((fn for fn in reader.fieldnames if fn.lower() in ('name', 'dataset_method', 'id')), None)

        for row in reader:
            ds = ''
            mt = ''
            red = ''
            if ds_field and method_field and red_field:
                ds = (row.get(ds_field) or '').strip()
                mt = (row.get(method_field) or '').strip()
                red = (row.get(red_field) or '').strip()
            elif name_field and red_field:
                name = (row.get(name_field) or '').strip()
                parts = name.split('_', 1)
                ds = parts[0] if parts else ''
                mt = parts[1] if len(parts) > 1 else ''
                red = (row.get(red_field) or '').strip()
            else:
                # best-effort: find any reduction-like column and any column with underscore
                for col in reader.fieldnames:
                    if ('reduct' in col.lower() or 'rate' in col.lower() or 'select' in col.lower()) and not red:
                        red = (row.get(col) or '').strip()
                for col in reader.fieldnames:
                    val = (row.get(col) or '').strip()
                    if '_' in val and not ds:
                        parts = val.split('_', 1)
                        ds = parts[0]
                        mt = parts[1] if len(parts) > 1 else ''
                        break

            if not ds and not mt:
                continue
            if (mt or '').lower() == 'none':
                mt = ''
            mapping[(ds, mt)] = red

    return mapping


def main():
    p = argparse.ArgumentParser(description="Generate CSV of fold macros from outputs")
    p.add_argument("--pattern", default=None, help="glob pattern to find measures (default: resources/output/*/**/measures.fold_*.json)")
    default_results_dir = os.environ.get("RESULTS_DIR", str(Path(__file__).resolve().parent.parent.parent / "results"))
    p.add_argument("--selection-csv", default=os.path.join(default_results_dir, "instance_selection", "selection_summary.csv"), help="CSV with selection reduction values to include (optional)")
    p.add_argument("--model", default=None, help="filter results to a specific model subdirectory (e.g. modernbert, roberta)")
    p.add_argument("-o", "--output", default="results_from_outputs.csv", help="output CSV path")
    args = p.parse_args()
    pattern = args.pattern if args.pattern else PATTERN_DEFAULT

    paths = sorted(glob.glob(pattern, recursive=True))
    if not paths:
        raise SystemExit(f"No files found for pattern: {pattern}")

    # data[(dataset,method,model)][fold] = value
    data = defaultdict(dict)
    fold_set = set()

    fold_re = re.compile(r"fold_(\d+)\.json$")

    for path in paths:
        m = fold_re.search(path)
        if not m:
            continue
        fold = int(m.group(1))
        fold_set.add(fold)

        # determine top-level folder name that contains dataset and method
        base_dir = pattern.split("/*/**/")[0] if "/*/**/" in pattern else "resources/output"
        rel = os.path.relpath(os.path.dirname(path), base_dir)
        parts = rel.split(os.sep) if rel != "." else [os.path.basename(os.path.dirname(path))]
        model = parts[0] if parts else ""

        # skip if --model filter is set and this path belongs to a different model
        if args.model and model != args.model:
            continue
        top = find_top_dir(parts)

        if "_" in top:
            dataset, method = top.rsplit("_", 1)
        else:
            dataset, method = top, ""

        with open(path, "r", encoding="utf-8") as f:
            j = json.load(f)

        if "macro" not in j:
            # skip files without macro
            continue

        try:
            val = float(j["macro"])
        except Exception:
            continue

        data[(dataset, method, model)][fold] = val

    if not data:
        raise SystemExit("No valid macro values found in matched files")

    folds = sorted(fold_set)
    max_fold = max(folds)

    # header: dataset,method,reduction,class,mac0...macN,macro_avg_ic
    mac_cols = [f"mac{f}" for f in folds]
    header = ["dataset", "method", "reduction", "class"] + mac_cols + ["macro_avg_ic"]

    out_path = Path(args.output)
    # try to load selection reductions CSV (optional)
    selection_map = {}
    sel_path = Path(args.selection_csv)
    if sel_path.exists():
        try:
            selection_map = load_selection_csv(sel_path)
            print(f"Loaded selection CSV with {len(selection_map)} entries from: {sel_path}")
        except Exception as exc:
            print(f"Warning: failed to load selection CSV: {exc}")
    else:
        print(f"Selection CSV not found (skipping reductions): {sel_path}")

    with out_path.open("w", encoding="utf-8", newline="") as f:
        # write CSV manually to avoid extra quoting issues in spreadsheets
        writer = csv.writer(f)
        writer.writerow(header)

        # sort by dataset then method then model
        keys = sorted(data.keys(), key=lambda x: (x[0] or "", x[1] or "", x[2] or ""))
        for dataset, method, model in keys:
            # fill reduction from selection CSV when available
            reduction = selection_map.get((dataset, method)) or selection_map.get((dataset, method or "None")) or ""
            print(f"Processing: dataset={dataset}, method={method}, model={model}, reduction={reduction}")
            row = [dataset, method, reduction, model]
            folds_map = data[(dataset, method, model)]
            for fnum in folds:
                if fnum in folds_map:
                    row.append(f"{folds_map[fnum]:.10f}")
                else:
                    row.append("")
            # leave macro_avg_ic empty (to be calculated in spreadsheet)
            row.append("")
            writer.writerow(row)

    print(f"Wrote CSV to: {out_path}")


if __name__ == "__main__":
    main()
