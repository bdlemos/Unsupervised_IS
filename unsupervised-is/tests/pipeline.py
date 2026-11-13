"""Toy dataset in the pipeline layout and a driver for run_generateSplit.main()."""
import gzip
import pickle
import sys

import pandas as pd
from sklearn.datasets import dump_svmlight_file
from sklearn.model_selection import StratifiedKFold

from synthetic import make_blobs

FOLDS = 3


def make_dataset(root):
    """root/datain/toy/{jina-v5/train<f>.gz,test<f>.gz, splits/split_<FOLDS>.pkl} as the pipeline expects."""
    X, y = make_blobs(600, 8, 0)
    rep = root / "datain" / "toy" / "jina-v5"
    rep.mkdir(parents=True)
    (root / "datain" / "toy" / "splits").mkdir()
    rows = []
    for f, (tr, te) in enumerate(StratifiedKFold(FOLDS, shuffle=True, random_state=0).split(X, y)):
        for name, idx in (("train", tr), ("test", te)):
            with gzip.open(rep / f"{name}{f}.gz", "wb") as fh:
                dump_svmlight_file(X[idx], y[idx], fh)
        rows.append({"train_idxs": list(tr), "test_idxs": list(te)})
    splits = pd.DataFrame(rows)
    with open(root / "datain" / "toy" / "splits" / f"split_{FOLDS}.pkl", "wb") as fh:
        pickle.dump(splits, fh)
    return splits


def run(monkeypatch, root, method, *extra):
    import run_generateSplit
    monkeypatch.setattr(sys, "argv", [
        "run_generateSplit.py", "-d", "toy", "-m", method, "--folds", str(FOLDS),
        "--inputrep", "jina-v5", "--datain", str(root / "datain"), "--out", str(root / "out"), *extra,
    ])
    run_generateSplit.main()
    return root / "out" / "selection" / "toy"


def load_split(outdir, tag):
    with open(outdir / f"split_{FOLDS}_{tag}.pkl", "rb") as fh:
        return pickle.load(fh)
