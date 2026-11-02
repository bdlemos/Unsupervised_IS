import numpy as np
import pytest

from synthetic import make_blobs
from src.iSel import ablation_methods
from src.iSel.autoencoder_is import AutoencoderIS
from src.iSel.entropy_sublinear_ae_is import EntropySublinearAEIS
from src.iSel.factorial_is import FactorialIS
from src.iSel.sublinear_ae_is import SublinearAEIS


def _kept(selector, n=900, d=16, seed=1):
    X, y = make_blobs(n, d, seed)
    selector.fit(X, y)
    return np.sort(np.asarray(selector.sample_indices_))


@pytest.mark.parametrize("n,d,seed", [(300, 8, 0), (900, 16, 1)])
@pytest.mark.parametrize("cell,reference", [
    (dict(rate="entropy", selection="re"), lambda: EntropySublinearAEIS(random_state=13)),
    (dict(rate="entropy", selection="cells"), lambda: ablation_methods.AblationClusterSublinearRandom(random_state=13)),
    (dict(rate="entropy", selection="global"), lambda: ablation_methods.AblationRandomESAERate(random_state=13)),
    (dict(rate=0.25, selection="re"), lambda: SublinearAEIS(target_reduction=0.25, n_clusters=-1, random_state=13)),
])
def test_cells_reproduce_existing_methods(fake_ae, cell, reference, n, d, seed):
    """Existing ESAE / ablation / SAE results stay comparable with the factorial cells."""
    assert np.array_equal(_kept(FactorialIS(**cell, random_state=13), n, d, seed), _kept(reference(), n, d, seed))


@pytest.mark.parametrize("selection", ["cells", "global"])
def test_only_re_cells_train_the_autoencoder(monkeypatch, selection):
    def fail(*a, **k):
        raise AssertionError("autoencoder trained for a cell that does not use it")
    monkeypatch.setattr(AutoencoderIS, "fit", fail)
    sel = FactorialIS(rate=0.2, selection=selection)
    _kept(sel)
    assert sel.selection_info_["time_autoencoder"] == 0.0


def test_fixed_global_cell_skips_tessellation():
    sel = FactorialIS(rate=0.2, selection="global")
    _kept(sel)
    assert "n_clusters" not in sel.selection_info_
    assert sel.selection_info_["time_tessellation"] == 0.0


@pytest.mark.parametrize("rate", [0.20, 0.213])
@pytest.mark.parametrize("selection", ["re", "cells", "global"])
def test_fixed_rate_is_applied(fake_ae, rate, selection):
    sel = FactorialIS(rate=rate, selection=selection)
    _kept(sel)
    info = sel.selection_info_
    assert info["rate_rule"] == "fixed" and info["target_reduction"] == rate
    assert abs(info["reduction"] - rate) < 0.01


def test_cells_share_tessellation_and_budgets(fake_ae):
    """Same rate and random_state: re and cells keep the same count in every cell, so only *which* differs."""
    re_cell, cells = FactorialIS(rate=0.3, selection="re"), FactorialIS(rate=0.3, selection="cells")
    _kept(re_cell)
    _kept(cells)
    assert np.array_equal(re_cell.labels_, cells.labels_)
    k = len(re_cell.cluster_sizes_)
    assert np.array_equal(np.bincount(re_cell.labels_[re_cell.mask], minlength=k),
                          np.bincount(cells.labels_[cells.mask], minlength=k))
    assert not np.array_equal(re_cell.mask, cells.mask)


def test_entropy_rate_matches_across_selections(fake_ae):
    rates = set()
    for selection in ("re", "cells", "global"):
        sel = FactorialIS(rate="entropy", selection=selection)
        _kept(sel)
        rates.add(sel.target_reduction_)
    assert len(rates) == 1


@pytest.mark.parametrize("bad", [dict(selection="knn"), dict(rate=1.5), dict(rate="median")])
def test_invalid_cells_fail(bad):
    with pytest.raises(ValueError):
        _kept(FactorialIS(**bad))


# ── registry ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name,rate,selection", [
    ("fx-entropy-re", "entropy", "re"),
    ("fx-entropy-cells", "entropy", "cells"),
    ("fx-r20-global", 0.20, "global"),
    ("fx-r21.3-cells", 0.213, "cells"),
])
def test_registry_parses_factorial_names(name, rate, selection):
    import run_generateSplit
    sel = run_generateSplit.get_selector(name)
    assert isinstance(sel, FactorialIS)
    assert sel.rate == pytest.approx(rate) if rate != "entropy" else sel.rate == "entropy"
    assert sel.selection == selection


@pytest.mark.parametrize("name", ["fx-r20", "fx-entropy-knn", "fx-20-re"])
def test_registry_rejects_malformed_factorial_names(name):
    import run_generateSplit
    assert run_generateSplit.get_selector(name) is None
