from src.utils.general import print_in_file
import numpy as np
import os
import socket
import subprocess
from datetime import datetime
import json


def to_jsonable(obj):
    """Convert numpy scalars/arrays (possibly nested in dicts/lists) to plain JSON types."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    return repr(obj)


def code_version():
    """Commit of the code that ran (GIT_COMMIT env var wins, e.g. inside Docker without .git)."""
    if os.environ.get("GIT_COMMIT"):
        return {"commit": os.environ["GIT_COMMIT"], "dirty": None}
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        commit = subprocess.check_output(["git", "-C", here, "rev-parse", "HEAD"],
                                         stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "-C", here, "status", "--porcelain", "--untracked-files=no"],
                                             stderr=subprocess.DEVNULL, text=True).strip())
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def save_selection_params(args, folds_record):
    """Write split_<folds>_<method>_params.json: every value that decided the selection, per fold."""
    reductions = [f["reduction"] for f in folds_record]
    times = [f["time_seconds"] for f in folds_record]
    data = {
        "schema_version": 1,
        "dataset": args.dataset,
        "method": args.method,
        "method_tag": args.method_tag,
        "seed": args.seed,
        "inputrep": args.inputrep,
        "n_folds": args.folds,
        "code": code_version(),
        "machine": socket.gethostname(),
        "started_at": args.start_time,
        "finished_at": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
        "summary": {
            "reduction_mean": float(np.mean(reductions)),
            "reduction_std": float(np.std(reductions)),
            "time_mean_seconds": float(np.mean(times)),
        },
        "folds": folds_record,
    }
    filename = f"{args.outputdir}/split_{args.folds}_{args.method_tag}_params.json"
    with open(filename, "w") as outfile:
        json.dump(to_jsonable(data), outfile, indent=2)
    print(f"Selection parameters saved to {filename}")


def save_results(args, info):
    # saving results in file
    string = f"""times: {info['time_for_reduce']}\n
				reducion: {info['reducion']}\n
				time mean: {np.mean(info['time_for_reduce'])}\n
				reducion mean: {np.mean(info['reducion'])}\n
				reduced_len_mean: {np.mean(info['reduced_len'])}\n
				original_len_mean: {np.mean(info['original_len'])}"""

    print_in_file(string, args.filename)

    info["time_mean"] = np.mean(info['time_for_reduce'])
    info["reduction_mean"] = np.mean(info['reducion'])
    info["reduced_len_mean"] = np.mean(info['reduced_len'])
    info["original_len_mean"] = np.mean(info['original_len'])

    args.end_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

    # Saving results in json
    data = {
        "hiperparams": str(args),
        "dataset": args.dataset,
        "method": args.method_tag,
        "info": info,
        "machine": socket.gethostname(),
    }

    with open(args.filename+".json", 'w') as outfile:
        json.dump(to_jsonable(data), outfile, indent=4)
