import gzip
import json
import pickle
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import dump_svmlight_file
from sklearn.model_selection import StratifiedKFold

from synthetic import make_blobs

FOLDS = 3


@pytest.fixture
def dataset(tmp_path):
    """datain/<ds>/{jina-v5/train<f>.gz,test<f>.gz, splits/split_<FOLDS>.pkl} as the pipeline expects."""
    X, y = make_blobs(600, 8, 0)
    rep = tmp_path / "datain" / "toy" / "jina-v5"
    rep.mkdir(parents=True)
    (tmp_path / "datain" / "toy" / "splits").mkdir()
    rows = []
    for f, (tr, te) in enumerate(StratifiedKFold(FOLDS, shuffle=True, random_state=0).split(X, y)):
        for name, idx in (("train", tr), ("test", te)):
            with gzip.open(rep / f"{name}{f}.gz", "wb") as fh:
                dump_svmlight_file(X[idx], y[idx], fh)
        rows.append({"train_idxs": list(tr), "test_idxs": list(te)})
    with open(tmp_path / "datain" / "toy" / "splits" / f"split_{FOLDS}.pkl", "wb") as fh:
        pickle.dump(pd.DataFrame(rows), fh)
    return tmp_path


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


def test_params_json_records_every_fold(fake_ae, dataset, monkeypatch):
    outdir = run(monkeypatch, dataset, "entropy-sublinear-ae-is")
    data = json.load(open(outdir / f"split_{FOLDS}_entropy-sublinear-ae-is_params.json"))
    assert data["method"] == data["method_tag"] == "entropy-sublinear-ae-is"
    assert data["seed"] is None
    assert len(data["folds"]) == FOLDS
    for f, rec in enumerate(data["folds"]):
        assert rec["fold"] == f
        assert rec["params"]["random_state"] == 13
        assert rec["params"]["mode"] == "original"
        sel = rec["selection"]
        assert sel["n_clusters"] == int(np.sqrt(rec["n_train"]))
        assert 0.05 <= sel["target_reduction"] <= 0.5
        assert rec["n_kept"] == sel["n_kept"]
    assert data["summary"]["reduction_mean"] == pytest.approx(np.mean([r["reduction"] for r in data["folds"]]))


def test_seed_gives_independent_reproducible_repetitions(fake_ae, dataset, monkeypatch):
    method = "ablation-cluster-sublinear-random"
    outdir = run(monkeypatch, dataset, method, "--seed", "1")
    first = load_split(outdir, f"{method}-seed1")
    run(monkeypatch, dataset, method, "--seed", "1")
    again = load_split(outdir, f"{method}-seed1")
    run(monkeypatch, dataset, method, "--seed", "2")
    other = load_split(outdir, f"{method}-seed2")

    for f in range(FOLDS):
        assert list(first.loc[f].train_idxs) == list(again.loc[f].train_idxs)
        assert set(first.loc[f].train_idxs) != set(other.loc[f].train_idxs)

    params = json.load(open(outdir / f"split_{FOLDS}_{method}-seed2_params.json"))
    states = [r["params"]["random_state"] for r in params["folds"]]
    assert params["seed"] == 2 and len(set(states)) == FOLDS and 13 not in states


def test_unknown_method_fails_loudly(fake_ae, dataset, monkeypatch):
    with pytest.raises(ValueError, match="Unknown selection method"):
        run(monkeypatch, dataset, "no-such-method")
