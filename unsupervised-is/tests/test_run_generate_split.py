import json

import numpy as np
import pytest

from pipeline import FOLDS, load_split, run


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


def test_seed_suffix_in_method_name_equals_seed_flag(fake_ae, dataset, monkeypatch):
    outdir = run(monkeypatch, dataset, "fx-r20-cells", "--seed", "4")
    by_flag = load_split(outdir, "fx-r20-cells-seed4")
    (outdir / f"split_{FOLDS}_fx-r20-cells-seed4.pkl").unlink()
    run(monkeypatch, dataset, "fx-r20-cells-seed4")
    by_name = load_split(outdir, "fx-r20-cells-seed4")
    for f in range(FOLDS):
        assert list(by_flag.loc[f].train_idxs) == list(by_name.loc[f].train_idxs)
    params = json.load(open(outdir / f"split_{FOLDS}_fx-r20-cells-seed4_params.json"))
    assert params["method"] == "fx-r20-cells" and params["seed"] == 4


def test_unknown_method_fails_loudly(fake_ae, dataset, monkeypatch):
    with pytest.raises(ValueError, match="Unknown selection method"):
        run(monkeypatch, dataset, "no-such-method")
