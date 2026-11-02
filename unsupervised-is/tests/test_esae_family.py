import json
import os

import numpy as np
import pytest

from synthetic import make_blobs, fake_reconstruction_errors
from src.main.python.iSel import esae_core, ablation_methods
from src.main.python.iSel.entropy_sublinear_ae_is import EntropySublinearAEIS
from src.main.python.iSel.sublinear_ae_is import SublinearAEIS
from src.main.python.utils.save_results import to_jsonable

GOLDEN = os.path.join(os.path.dirname(__file__), "data", "esae_golden.json")


# ── esae_core ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("k", [None, -1, 0, 99999, 10**6])
def test_sqrt_n_sentinels(k):
    assert esae_core.resolve_n_clusters(k, 900) == 30


def test_explicit_k_is_clamped():
    assert esae_core.resolve_n_clusters(200, 1000) == 200
    assert esae_core.resolve_n_clusters(200, 300) == 150
    assert esae_core.resolve_n_clusters(200, 300, max_divisor=10) == 30
    assert esae_core.resolve_n_clusters(1, 300) == 2


def test_entropy_uniform_and_concentrated():
    assert esae_core.normalized_entropy(np.array([10, 10, 10, 10])) == pytest.approx(1.0)
    assert esae_core.normalized_entropy(np.array([97, 1, 1, 1])) < 0.2
    assert esae_core.normalized_entropy(np.array([5, 0, 0])) == 1.0


def test_entropy_rate_formula_and_clamp():
    assert esae_core.entropy_rate(0.98, 0.5, 15) == pytest.approx(0.5 * (1 - 0.98 ** 15))
    assert esae_core.entropy_rate(1.0, 0.5, 15) == 0.05
    assert esae_core.entropy_rate(0.0, 0.5, 15) == 0.5


def test_sublinear_quotas_hit_target_and_respect_cell_sizes():
    sizes = np.array([400, 100, 25, 4, 1])
    lam, quotas = esae_core.sublinear_quotas(sizes, 0.5, 300)
    assert quotas.sum() >= 300
    assert np.all(quotas <= sizes)
    # sublinear: the largest cell loses the largest fraction
    assert quotas[0] / sizes[0] < quotas[2] / sizes[2]


def test_removal_weights_direction():
    scores = np.array([0.1, 1.0, 10.0])
    assert np.argmax(esae_core.removal_weights(scores, "original")) == 0
    assert np.argmax(esae_core.removal_weights(scores, "cleaning")) == 2
    with pytest.raises(ValueError):
        esae_core.removal_weights(scores, "bogus")


def test_remove_within_cells_meets_quotas_exactly():
    rng = np.random.RandomState(0)
    labels = rng.randint(0, 5, size=200)
    sizes = np.bincount(labels, minlength=5)
    quotas = sizes // 2
    for weights in (None, rng.rand(200) + 0.1):
        mask = esae_core.remove_within_cells(labels, quotas, np.random.RandomState(1), weights)
        assert np.array_equal(np.bincount(labels[mask], minlength=5), quotas)


# ── ESAE-IS ─────────────────────────────────────────────────────────────────

def _fit(selector, n=900, d=16, seed=1):
    X, y = make_blobs(n, d, seed)
    selector.fit(X, y)
    return X, y, selector


def test_esae_matches_paper_code(fake_ae):
    """Bit-identical to the pre-2026-10-10 implementation used for the paper results."""
    golden = json.load(open(GOLDEN))["cases"]
    for key, case in golden.items():
        n, d, seed = map(int, key.split("_"))
        _, _, sel = _fit(EntropySublinearAEIS(random_state=13), n, d, seed)
        assert sel.entropy_ == pytest.approx(case["entropy"], abs=1e-12)
        assert sel.target_reduction_ == pytest.approx(case["target_reduction"], abs=1e-12)
        assert sorted(sel.sample_indices_.tolist()) == case["kept"], key


def test_esae_default_uses_sqrt_n_cells(fake_ae):
    _, _, sel = _fit(EntropySublinearAEIS(random_state=13), n=900)
    assert sel.selection_info_["n_clusters"] == 30
    assert len(sel.cluster_sizes_) == 30


def _within_cell_re(X, labels):
    """Reconstruction error centered on its cell mean (the weighting only acts inside cells)."""
    re = fake_reconstruction_errors(X)
    cell_mean = np.bincount(labels, weights=re) / np.maximum(np.bincount(labels), 1)
    return re - cell_mean[labels]


def test_esae_original_mode_removes_low_error_first(fake_ae):
    X, _, sel = _fit(EntropySublinearAEIS(random_state=13))
    re = _within_cell_re(X, sel.labels_)
    assert re[sel.mask].mean() > re[~sel.mask].mean()
    _, _, clean = _fit(EntropySublinearAEIS(random_state=13, mode="cleaning"))
    assert re[clean.mask].mean() < re[~clean.mask].mean()


def test_esae_selection_record(fake_ae):
    _, _, sel = _fit(EntropySublinearAEIS(random_state=13))
    rec = sel.selection_record()
    info = rec["selection"]
    assert rec["params"]["r_max"] == 0.5 and rec["params"]["alpha"] == 15.0
    assert rec["params"]["mode"] == "original"
    assert info["target_reduction"] == pytest.approx(esae_core.entropy_rate(info["entropy"], 0.5, 15.0))
    assert info["n_kept"] == len(sel.sample_indices_)
    assert info["reduction"] == pytest.approx(1 - info["n_kept"] / info["n_samples"])
    assert abs(info["reduction"] - info["target_reduction"]) < 0.01
    for key in ("sublinear_lambda", "n_cells_pruned", "re_median", "time_autoencoder"):
        assert key in info
    json.dumps(to_jsonable(rec))


def test_fit_does_not_mutate_params(fake_ae):
    sel = SublinearAEIS(target_reduction=0.4, n_clusters=-1, random_state=13, beta=2.0)
    before = sel.get_params()
    _fit(sel)
    assert sel.get_params() == before


def test_esae_does_not_touch_global_rng(fake_ae):
    np.random.seed(123)
    expected = np.random.rand()
    np.random.seed(123)
    sel = EntropySublinearAEIS(random_state=13)
    sel._reconstruction_errors = lambda X, y: fake_reconstruction_errors(X)
    _fit(sel)
    assert np.random.rand() == expected


# ── Ablations ───────────────────────────────────────────────────────────────

ABLATIONS = [
    ablation_methods.AblationRandomESAERate,
    ablation_methods.AblationClusterUniform,
    ablation_methods.AblationClusterSublinearRandom,
    ablation_methods.AblationAEESAERate,
    ablation_methods.AblationClusterUniformAE,
]


@pytest.mark.parametrize("cls", ABLATIONS)
def test_ablations_share_the_esae_rate(fake_ae, cls):
    _, _, esae = _fit(EntropySublinearAEIS(random_state=13))
    _, _, abl = _fit(cls(random_state=13))
    assert abl.selection_info_["target_reduction"] == pytest.approx(esae.target_reduction_)
    assert abl.selection_info_["n_clusters"] == esae.selection_info_["n_clusters"]
    assert abs(abl.reduction_ - esae.reduction_) < 0.02


def test_sublinear_random_ablation_keeps_esae_cell_budgets(fake_ae):
    """ESAE without the AE: same cells, same per-cell counts — only which instances differ."""
    _, _, esae = _fit(EntropySublinearAEIS(random_state=13))
    _, _, abl = _fit(ablation_methods.AblationClusterSublinearRandom(random_state=13))
    k = len(esae.cluster_quotas_)
    assert np.array_equal(np.bincount(esae.labels_[esae.mask], minlength=k),
                          np.bincount(esae.labels_[abl.mask], minlength=k))
