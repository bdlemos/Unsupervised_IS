"""Random selection at the per-fold count of a reference method (fx-match<ref>-<selection>)."""
import json
import pickle

import numpy as np
import pandas as pd
import pytest

import run_generateSplit
from pipeline import FOLDS, load_split, make_dataset, run
from src.iSel.factorial_is import FactorialIS
from synthetic import make_blobs

METHOD = "fx-matchbiois-global"
# Kept fraction of the reference in each fold: different per fold, so a run that used one
# global rate instead of the per-fold count would fail
REFERENCE_KEEP = (0.61, 0.70, 0.55)


def write_reference(root, splits, reference="biois"):
    """A reference selection in the jina-results layout, keeping REFERENCE_KEEP of each fold."""
    rng = np.random.RandomState(7)
    rows = []
    for f in range(FOLDS):
        train = np.asarray(splits.loc[f].train_idxs)
        kept = rng.choice(train, size=int(len(train) * REFERENCE_KEEP[f]), replace=False)
        rows.append({"train_idxs": list(kept), "test_idxs": list(splits.loc[f].test_idxs)})
    folder = root / "reference" / "toy"
    folder.mkdir(parents=True)
    with open(folder / f"split_{FOLDS}_{reference}.pkl", "wb") as fh:
        pickle.dump(pd.DataFrame(rows), fh)
    return [len(r["train_idxs"]) for r in rows]


@pytest.fixture
def matched(tmp_path):
    """Toy dataset plus a biois reference; returns (root, base splits, per-fold reference kept counts)."""
    splits = make_dataset(tmp_path)
    return tmp_path, splits, write_reference(tmp_path, splits)


def run_matched(monkeypatch, root, method=METHOD):
    return run(monkeypatch, root, method, "--match-dir", str(root / "reference"))


def kept_per_fold(outdir, tag):
    split = load_split(outdir, tag)
    return [sorted(split.loc[f].train_idxs) for f in range(FOLDS)]


class TestFactorialMatchedRate:
    @pytest.mark.parametrize("keep_count", [1, 450, 899, 900])
    def test_global_keeps_exactly_keep_count(self, keep_count):
        X, y = make_blobs(900, 16, 1)

        selector = FactorialIS(rate="matched", selection="global", keep_count=keep_count).fit(X, y)

        assert len(selector.sample_indices_) == keep_count

    # Cell selections keep at least one instance per cell (sublinear quotas), so the count is
    # exact only when it is at least the number of cells (30 here)
    @pytest.mark.parametrize("keep_count", [450, 600])
    @pytest.mark.parametrize("selection", ["cells", "re"])
    def test_cell_selections_keep_exactly_keep_count(self, fake_ae, selection, keep_count):
        X, y = make_blobs(900, 16, 1)

        selector = FactorialIS(rate="matched", selection=selection, keep_count=keep_count).fit(X, y)

        assert len(selector.sample_indices_) == keep_count

    def test_records_matched_rate_rule(self):
        X, y = make_blobs(900, 16, 1)

        selector = FactorialIS(rate="matched", selection="global", keep_count=600).fit(X, y)

        assert selector.selection_info_["rate_rule"] == "matched"

    def test_target_reduction_is_the_removed_fraction(self):
        X, y = make_blobs(900, 16, 1)

        selector = FactorialIS(rate="matched", selection="global", keep_count=600).fit(X, y)

        assert selector.selection_info_["target_reduction"] == pytest.approx(300 / 900)

    def test_global_matched_skips_tessellation(self):
        X, y = make_blobs(900, 16, 1)

        selector = FactorialIS(rate="matched", selection="global", keep_count=600).fit(X, y)

        assert "n_clusters" not in selector.selection_info_

    @pytest.mark.parametrize("keep_count", [
        None,   # matched without a count
        0,
        901,    # more than the 900 instances
    ])
    def test_rejects_missing_or_out_of_range_count(self, keep_count):
        X, y = make_blobs(900, 16, 1)

        with pytest.raises(ValueError):
            FactorialIS(rate="matched", selection="global", keep_count=keep_count).fit(X, y)


class TestMatchedRegistry:
    @pytest.mark.parametrize(("method", "reference"), [
        ("fx-matchbiois-global", "biois"),
        ("fx-matchbiois-cells", "biois"),
        ("fx-matche2sc-re", "e2sc"),
        ("fx-r20-global", None),
        ("fx-entropy-re", None),
        ("fx-matchbiois", None),           # no selection
        ("fx-matchbiois-knn", None),       # unknown selection
        ("biois", None),
    ])
    def test_matched_reference_parses_method_name(self, method, reference):
        assert run_generateSplit.matched_reference(method) == reference

    @pytest.mark.parametrize("selection", ["global", "cells", "re"])
    def test_get_selector_builds_matched_factorial_cell(self, selection):
        selector = run_generateSplit.get_selector(f"fx-matchbiois-{selection}")

        assert (selector.rate, selector.selection) == ("matched", selection)


class TestMatchedSelectionRun:
    @pytest.mark.parametrize("fold", range(FOLDS))
    def test_removed_count_equals_reference_removed_count(self, matched, monkeypatch, fold):
        root, splits, reference_kept = matched
        n_train = len(splits.loc[fold].train_idxs)

        outdir = run_matched(monkeypatch, root, f"{METHOD}-seed1")

        removed = n_train - len(load_split(outdir, f"{METHOD}-seed1").loc[fold].train_idxs)
        assert removed == n_train - reference_kept[fold]

    @pytest.mark.parametrize("fold", range(FOLDS))
    def test_params_json_records_target_count(self, matched, monkeypatch, fold):
        root, _, reference_kept = matched

        outdir = run_matched(monkeypatch, root, f"{METHOD}-seed1")

        match = json.load(open(outdir / f"split_{FOLDS}_{METHOD}-seed1_params.json"))["folds"][fold]["match"]
        assert (match["reference"], match["target_n_kept"]) == ("biois", reference_kept[fold])

    def test_same_seed_reproduces_selection(self, matched, monkeypatch):
        root, _, _ = matched
        first = kept_per_fold(run_matched(monkeypatch, root, f"{METHOD}-seed1"), f"{METHOD}-seed1")

        again = kept_per_fold(run_matched(monkeypatch, root, f"{METHOD}-seed1"), f"{METHOD}-seed1")

        assert first == again

    @pytest.mark.parametrize("fold", range(FOLDS))
    def test_different_seeds_select_different_instances(self, matched, monkeypatch, fold):
        root, _, _ = matched
        seed1 = kept_per_fold(run_matched(monkeypatch, root, f"{METHOD}-seed1"), f"{METHOD}-seed1")

        seed2 = kept_per_fold(run_matched(monkeypatch, root, f"{METHOD}-seed2"), f"{METHOD}-seed2")

        assert seed1[fold] != seed2[fold]

    def test_selection_differs_from_reference(self, matched, monkeypatch):
        root, _, _ = matched
        reference = pd.read_pickle(root / "reference" / "toy" / f"split_{FOLDS}_biois.pkl")

        selected = kept_per_fold(run_matched(monkeypatch, root, f"{METHOD}-seed1"), f"{METHOD}-seed1")

        assert selected[0] != sorted(reference.loc[0].train_idxs)


class TestMatchedReferenceErrors:
    def test_missing_reference_file_names_expected_path(self, dataset, monkeypatch):
        (dataset / "reference").mkdir()
        expected = str(dataset / "reference" / "toy" / f"split_{FOLDS}_biois.pkl")

        with pytest.raises(FileNotFoundError, match=expected):
            run_matched(monkeypatch, dataset)

    def test_missing_match_dir_fails(self, dataset, monkeypatch):
        monkeypatch.delenv("MATCH_SELECTION_DIR", raising=False)

        with pytest.raises(FileNotFoundError, match="MATCH_SELECTION_DIR"):
            run(monkeypatch, dataset, METHOD)

    def test_reference_from_other_folds_fails(self, tmp_path, monkeypatch):
        splits = make_dataset(tmp_path)
        shifted = splits.copy()
        shifted["train_idxs"] = list(splits["test_idxs"])     # indices outside each fold's train set
        write_reference(tmp_path, shifted)

        with pytest.raises(ValueError, match="other folds"):
            run_matched(monkeypatch, tmp_path)
