import time
from typing import Any, Optional

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import f1_score
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from transformers.utils import logging as hf_logging
from tqdm import tqdm

from src.model.slmdatahandle import prepare_inference_datasets, prepare_training_datasets, prep_data
from src.utils.misc import Classes, Documents

# Every fold reloads the checkpoint, and transformers then prints a "Loading weights" bar and a
# LOAD REPORT warning (decoder.bias UNEXPECTED, classifier.* MISSING), which is expected when a
# classification head is put on a masked-LM checkpoint. Keep errors only.
hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()

# Training regimes, chosen by model.training_args.regime (see conf/model/modernbert_legacy.yaml):
#   current  AdamW (weight decay 0.01 off bias/norm), early stop on validation macro-F1 with
#            patience counted in consecutive epochs, best-F1 weights restored before predicting
#   legacy   the setup of every run up to 2026-11-02 (code at ef6dfb9; e.g. the biO-IS column of
#            the main table): torch Adam with the group key 'weight_decay_rate', which Adam
#            ignores, so the effective weight decay is 0; early stop on validation loss with a
#            patience counter that is never reset; predicts with the LAST epoch's weights
REGIMES = ("current", "legacy")


def build_optimizer(named_parameters, lr: float, weight_decay_rate: float, regime: str = "current") -> Adam:
    """Optimizer for full fine-tuning under `regime` (current: AdamW; legacy: Adam, no effective decay)."""
    if regime not in REGIMES:
        raise ValueError(f"regime must be one of {REGIMES}, got {regime!r}")
    params = list(named_parameters)
    if regime == "legacy":
        # Verbatim from ef6dfb9: torch reads 'weight_decay', so 'weight_decay_rate' is stored in the
        # group but unused and every group gets Adam's default weight_decay=0.
        no_decay = ['bias', 'gamma', 'beta']
        return Adam([
            {'params': [p for n, p in params if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': weight_decay_rate},
            {'params': [p for n, p in params if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0},
        ], lr=lr)
    no_decay = ['bias', 'norm', 'LayerNorm']
    return AdamW([
        {'params': [p for n, p in params if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay_rate},
        {'params': [p for n, p in params if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0},
    ], lr=lr)


class ValF1EarlyStopping:
    """Current regime: stop after `max_patience` consecutive epochs without +min_delta val macro-F1."""

    metric = "val_macro_f1"
    restore_best = True

    def __init__(self, max_patience: int, min_delta: float) -> None:
        self.max_patience = max_patience
        self.min_delta = min_delta
        self.best: Optional[float] = None
        self.patience = 0
        self.should_stop = False

    def update(self, val_loss: float, val_f1: float) -> bool:
        """Record one epoch; True when it is the new best (its weights are the ones to keep)."""
        if self.best is None or val_f1 > self.best + self.min_delta:
            self.best = val_f1
            self.patience = 0
            return True
        self.patience += 1
        self.should_stop = self.patience >= self.max_patience
        return False

    def describe_improvement(self) -> str:
        return f"val macro-F1 improved → {self.best:.4f} (saved)"


class LegacyValLossStopping:
    """Legacy regime (ef6dfb9), bug for bug: the patience counter is cumulative (never reset on an
    improvement) and stops at the (max_patience + 1)-th non-improving epoch overall; nothing is
    restored, so the model predicts with the last epoch's weights."""

    metric = "val_loss_legacy"
    restore_best = False

    def __init__(self, max_patience: int, min_delta: float) -> None:
        self.max_patience = max_patience
        self.min_delta = min_delta
        self.best: Optional[float] = None
        self.patience = 0
        self.should_stop = False

    def update(self, val_loss: float, val_f1: float) -> bool:
        if self.best is None or val_loss + self.min_delta < self.best:
            self.best = val_loss
            return True
        if self.patience == self.max_patience:
            self.should_stop = True
        else:
            self.patience += 1
        return False

    def describe_improvement(self) -> str:
        return f"val loss improved → {self.best:.4E} (not restored: legacy regime)"


EARLY_STOPPING = {"current": ValF1EarlyStopping, "legacy": LegacyValLossStopping}


def make_early_stopping(regime: str, max_patience: int, min_delta: float):
    if regime not in REGIMES:
        raise ValueError(f"regime must be one of {REGIMES}, got {regime!r}")
    return EARLY_STOPPING[regime](max_patience, min_delta)


class SLMClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, model_config: dict[str, Any], dataset: str) -> None:
        self.model_config = model_config
        self.dataset = dataset
        self.model_name: str = model_config['model_name']
        self.model_tag: str = model_config['model_tag']
        self.max_len: int = model_config['training_args']['max_len']
        self.learning_rate: float = model_config['training_args']['lr']
        self.batch_size: int = model_config['training_args']['batch_size']
        self.num_max_epochs: int = model_config['training_args']['num_max_epochs']
        self.max_patience: int = model_config['training_args']['patience']
        self.weight_decay_rate: float = model_config['training_args']['weight_decay_rate']
        self.min_val_epoch_impro_delta: float = model_config['training_args']['min_val_epoch_impro_delta']
        self.max_grad_norm: float = model_config['training_args']['max_grad_norm']
        self.warmup_ratio: float = model_config['training_args'].get('warmup_ratio', 0.06)
        self.training_regime: str = model_config['training_args'].get('regime', 'current')
        if self.training_regime not in REGIMES:
            raise ValueError(f"model.training_args.regime must be one of {REGIMES}, got {self.training_regime!r}")
        self.early_stop_metric: str = EARLY_STOPPING[self.training_regime].metric

        self.full_finetuning: bool = True
        self.epoch_id: int = 0
        self.logging_dir: str = f"logs/{self.model_tag}/{self.dataset}/"
        print(f"[init] logging dir: {self.logging_dir}")

    def load_model(
        self,
        model_name: str,
        num_training_labels: int,
    ) -> tuple[AutoModelForSequenceClassification, AutoTokenizer]:
        print(f"[load] loading model '{model_name}' ({num_training_labels} labels)...")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_training_labels,
            torch_dtype="auto",
            device_map="auto",
        )

        print(f"[load] loading tokenizer (max_len={self.max_len})...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            do_lower_case=False,
            max_length=self.max_len,
        )

        if self.model_tag in ('roberta', 'bart'):
            tokenizer.add_prefix_space = True

        print("[load] model and tokenizer ready.")
        return model, tokenizer

    def fit(
        self,
        X_train: Documents,
        y_train: Classes,
        X_val: Optional[Documents] = None,
        y_val: Optional[Classes] = None,
    ) -> "SLMClassifier":
        """Fine-tuning of the pre-trained model.

        Parameters
        ----------
        X_train : Documents
        y_train : Classes
        X_val : Documents, optional
        y_val : Classes, optional
        """
        self.device = torch.device('cuda:0')

        X_train, y_train, X_val, y_val = prep_data(X_train, y_train, X_val, y_val)
        self.num_classes: int = len(set(y_train))

        self._time_to_train: float = time.time()
        self.model, self.tokenizer = self.load_model(self.model_name, num_training_labels=self.num_classes)
        self.model.to(self.device)

        self.training_loss: list[float] = []
        self.validation_loss: list[float] = []
        self.validation_f1: list[float] = []
        stopper = make_early_stopping(self.training_regime, self.max_patience, self.min_val_epoch_impro_delta)
        best_weights: Optional[dict[str, torch.Tensor]] = None
        self.best_epoch: int = 0

        data_loader_train, data_loader_val = prepare_training_datasets(
            X_train, y_train, X_val, y_val, self.tokenizer, self.max_len, self.batch_size
        )

        optimizer = self._set_optimizer()

        total_steps = len(data_loader_train) * self.num_max_epochs
        warmup_steps = int(total_steps * self.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        print(f"\n[fit] model={self.model_name} | classes={self.num_classes} | epochs={self.num_max_epochs} | patience={self.max_patience} | regime={self.training_regime}")
        print(f"[fit] lr={self.learning_rate:.1e} | warmup={warmup_steps} steps ({self.warmup_ratio*100:.0f}% of {total_steps})")
        print("-" * 70)

        while self.epoch_id < self.num_max_epochs:
            self.epoch_id += 1
            print(f"\n[epoch {self.epoch_id}/{self.num_max_epochs}]")

            # --- train ---
            self.model.train()
            tr_loss, nb_tr_steps = 0.0, 0

            for batch in data_loader_train:
                batch = {k: v.type(torch.long).to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss, _ = outputs[:2]

                loss.backward()
                tr_loss += loss.item()
                nb_tr_steps += 1

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_train_loss = tr_loss / nb_tr_steps
            self.training_loss.append(epoch_train_loss)

            # --- validate ---
            self.model.eval()
            vl_loss, nb_vl_steps = 0.0, 0
            val_pred, val_true = [], []

            for batch in data_loader_val:
                batch = {k: v.type(torch.long).to(self.device) for k, v in batch.items()}
                with torch.no_grad():
                    outputs = self.model(**batch)
                    loss, logits = outputs[:2]
                vl_loss += loss.item()
                nb_vl_steps += 1
                val_pred.append(logits.argmax(dim=-1).cpu().numpy())
                val_true.append(batch['labels'].cpu().numpy())

            dev_loss = vl_loss / nb_vl_steps
            dev_f1 = f1_score(np.concatenate(val_true), np.concatenate(val_pred), average='macro')
            self.validation_loss.append(dev_loss)
            self.validation_f1.append(dev_f1)

            current_lr = scheduler.get_last_lr()[0]
            print(f"  train loss : {epoch_train_loss:.4E}")
            print(f"  val loss   : {dev_loss:.4E}  |  val macro-F1: {dev_f1:.4f}  |  lr: {current_lr:.2e}")

            # --- checkpoint / patience (current: on validation macro-F1, since val loss rises from
            #     epoch ~2 through overconfidence while F1 still improves, so selecting on loss picked
            #     under-trained epoch-1/2 models; legacy: see LegacyValLossStopping) ---
            if stopper.update(dev_loss, dev_f1):
                if stopper.restore_best:
                    self.best_epoch = self.epoch_id
                    best_weights = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                print(f"  checkpoint : {stopper.describe_improvement()}")
            else:
                print(f"  patience   : {stopper.patience}/{self.max_patience}")
                if stopper.should_stop:
                    print(f"\n[fit] early stopping triggered at epoch {self.epoch_id}.")
                    break

        print("-" * 70)
        if best_weights is not None:
            self.model.load_state_dict(best_weights)
            print(f"[fit] best weights restored from epoch {self.best_epoch} (val macro-F1: {stopper.best:.4f})")
        elif not stopper.restore_best:
            # legacy: the model predicts with the weights of the last epoch trained
            self.best_epoch = self.epoch_id
            print(f"[fit] legacy regime: predicting with the last epoch's weights (epoch {self.epoch_id})")

        self._time_to_train = time.time() - self._time_to_train
        print(f"[fit] training complete in {self._time_to_train:.1f}s")
        return self

    def softmax(self, x: np.ndarray) -> np.ndarray:
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    def predict_proba(self, X: Documents) -> np.ndarray:
        """Class probability prediction for new documents.

        Parameters
        ----------
        X : Documents

        Returns
        -------
        proba : np.ndarray of shape (n_samples, n_classes)
        """
        self._time_to_predict: float = time.time()
        data_loader_test: DataLoader = prepare_inference_datasets(X, self.tokenizer, self.max_len, self.batch_size)

        self.model.eval()
        torch_logits: list[np.ndarray] = []

        for batch in data_loader_test:
            batch = {k: v.type(torch.long).to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)
                logits = outputs[0]
            torch_logits.append(logits.detach().cpu().numpy())

        proba = self.softmax(np.concatenate(torch_logits))
        self._time_to_predict = time.time() - self._time_to_predict
        return proba

    def representation(self, X: Documents) -> list[list[float]]:
        """Extract document representations from the fitted model.

        Parameters
        ----------
        X : Documents

        Returns
        -------
        rep_list : list[list[float]]
        """
        data_loader_test: DataLoader = prepare_inference_datasets(X, self.tokenizer, self.max_len, self.batch_size)

        self.model.eval()
        rep_list: list[list[float]] = []

        for batch in tqdm(data_loader_test, desc="[representation]"):
            batch = {k: v.type(torch.long).to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                if self.model_tag == 'bert':
                    outputs = self.model.bert(**batch)['pooler_output']
                elif self.model_tag == 'roberta':
                    outputs = self.model.roberta(**batch)['last_hidden_state']
                elif self.model_tag == 'bart':
                    outputs = self.model(**batch).encoder_last_hidden_state

            outputs = outputs.cpu().detach().numpy().tolist()
            for out in outputs:
                if self.model_tag in ('roberta', 'bart'):
                    out = np.mean(out, axis=0).tolist()
                rep_list.append(out)

        return rep_list

    def score(self, X: Documents, y: Classes, sample_weight: Optional[np.ndarray] = None) -> None:
        pass

    def _set_optimizer(self) -> Adam:
        """Optimizer of the training regime (see build_optimizer); head-only when not full fine-tuning."""
        if self.full_finetuning:
            return build_optimizer(self.model.named_parameters(), self.learning_rate,
                                   self.weight_decay_rate, self.training_regime)
        param_optimizer = list(self.model.classifier.named_parameters())
        optimizer_grouped_parameters = [{'params': [p for _, p in param_optimizer]}]
        return AdamW(optimizer_grouped_parameters, lr=self.learning_rate)
