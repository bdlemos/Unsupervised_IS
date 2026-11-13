"""Selectable training regime of the ModernBERT classifier (current vs legacy, see src/model/slm.py)."""
import os
import sys

import pytest
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.slm import (  # noqa: E402
    LegacyValLossStopping,
    SLMClassifier,
    ValF1EarlyStopping,
    build_optimizer,
)

MODEL_CONF = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "conf", "model")
LR = 5e-5


def model_config(name):
    return OmegaConf.load(os.path.join(MODEL_CONF, f"{name}.yaml"))


def stop_epoch(stopper, losses=None, f1s=None):
    """Feed one value per epoch; the 1-based epoch at which the stopper asks to stop, or None."""
    n = len(losses if losses is not None else f1s)
    for epoch in range(n):
        stopper.update(losses[epoch] if losses else 0.0, f1s[epoch] if f1s else 0.0)
        if stopper.should_stop:
            return epoch + 1
    return None


class TinyModel(torch.nn.Module):
    """Named like a transformer: a dense weight, a bias and a norm weight."""

    def __init__(self):
        super().__init__()
        self.dense = torch.nn.Linear(2, 2)
        self.norm = torch.nn.LayerNorm(2)


def step_with_zero_gradient(regime):
    """One optimizer step on a weight of ones whose gradient is zero; returns the weight after it."""
    model = TinyModel()
    torch.nn.init.ones_(model.dense.weight)
    optimizer = build_optimizer(model.named_parameters(), LR, 0.01, regime)
    for p in model.parameters():
        p.grad = torch.zeros_like(p)
    optimizer.step()
    return model.dense.weight.detach()


class TestModelConfigSelection:
    @pytest.mark.parametrize(("config", "regime"), [
        ("modernbert", "current"),          # HEAD default stays the current regime
        ("modernbert_legacy", "legacy"),
    ])
    def test_config_selects_regime(self, config, regime):
        assert SLMClassifier(model_config(config), "trec").training_regime == regime

    @pytest.mark.parametrize(("config", "metric"), [
        ("modernbert", "val_macro_f1"),
        ("modernbert_legacy", "val_loss_legacy"),
    ])
    def test_config_sets_early_stop_metric_recorded_in_measures(self, config, metric):
        assert SLMClassifier(model_config(config), "trec").early_stop_metric == metric

    @pytest.mark.parametrize(("key", "value"), [
        # conf/model/modernbert.yaml at ef6dfb9, the last commit with the legacy setup
        ("max_len", 256),
        ("batch_size", 32),
        ("lr", 5e-5),
        ("num_max_epochs", 10),
        ("patience", 3),
        ("weight_decay_rate", 0.01),
        ("min_val_epoch_impro_delta", 1e-4),
        ("max_grad_norm", 1.0),
        ("warmup_ratio", 0.06),
    ])
    def test_legacy_config_keeps_ef6dfb9_training_args(self, key, value):
        assert model_config("modernbert_legacy").training_args[key] == pytest.approx(value)

    @pytest.mark.parametrize("field", ["model_name", "model_tag", "type", "text_representation"])
    def test_legacy_config_uses_same_model_as_default(self, field):
        assert model_config("modernbert_legacy")[field] == model_config("modernbert")[field]

    def test_unknown_regime_is_rejected(self):
        config = model_config("modernbert")
        config.training_args.regime = "adamw-ish"

        with pytest.raises(ValueError, match="regime"):
            SLMClassifier(config, "trec")


class TestBuildOptimizer:
    @pytest.mark.parametrize(("regime", "optimizer"), [
        ("legacy", torch.optim.Adam),
        ("current", torch.optim.AdamW),
    ])
    def test_regime_selects_optimizer_class(self, regime, optimizer):
        assert type(build_optimizer(TinyModel().named_parameters(), LR, 0.01, regime)) is optimizer

    def test_legacy_has_no_effective_weight_decay(self):
        """A zero gradient leaves the weight unchanged: the 0.01 'weight_decay_rate' is ignored."""
        assert torch.equal(step_with_zero_gradient("legacy"), torch.ones(2, 2))

    def test_current_decays_weights(self):
        assert step_with_zero_gradient("current") == pytest.approx(torch.full((2, 2), 1 - LR * 0.01))

    def test_unknown_regime_is_rejected(self):
        with pytest.raises(ValueError, match="regime"):
            build_optimizer(TinyModel().named_parameters(), LR, 0.01, "sgd")


class TestLegacyValLossStopping:
    @pytest.mark.parametrize(("losses", "epoch"), [
        pytest.param([1.0, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99], 6, id="4th-non-improving-epoch"),
        # epochs 2, 5, 6 and 9 do not improve: stops at 9 although epochs 7-8 improved,
        # as in fold 2 of the legacy trec/biois log (patience 0/3 .. 3/3, stop at epoch 9)
        pytest.param([1.0, 1.1, 0.9, 0.8, 0.85, 0.85, 0.7, 0.6, 0.65, 0.5], 9, id="patience-never-reset"),
        pytest.param([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1], None, id="always-improving"),
        # an improvement smaller than min_delta (1e-4) counts as non-improving
        pytest.param([1.0, 0.99995, 0.99994, 0.99993, 0.99992], 5, id="below-min-delta"),
    ])
    def test_stops_at_fourth_non_improving_epoch_overall(self, losses, epoch):
        assert stop_epoch(LegacyValLossStopping(3, 1e-4), losses=losses) == epoch

    def test_ignores_validation_f1(self):
        f1s = [0.9, 0.1, 0.1, 0.1, 0.1, 0.1]

        assert stop_epoch(LegacyValLossStopping(3, 1e-4), losses=[1.0, 0.9, 0.8, 0.7, 0.6, 0.5], f1s=f1s) is None

    def test_does_not_restore_best_weights(self):
        assert LegacyValLossStopping.restore_best is False


class TestValF1EarlyStopping:
    @pytest.mark.parametrize(("f1s", "epoch"), [
        pytest.param([0.5, 0.6, 0.6, 0.6], 4, id="two-consecutive"),
        # non-improving at 2, 4, 6, 7: the counter resets on 3 and 5
        pytest.param([0.5, 0.4, 0.6, 0.5, 0.7, 0.6, 0.5], 7, id="reset-on-improvement"),
    ])
    def test_stops_after_consecutive_non_improving_epochs(self, f1s, epoch):
        assert stop_epoch(ValF1EarlyStopping(2, 1e-4), f1s=f1s) == epoch

    def test_restores_best_weights(self):
        assert ValF1EarlyStopping.restore_best is True
