import io
import time
from typing import Any, Optional

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm

from src.model.slmdatahandle import prepare_inference_datasets, prepare_training_datasets, prep_data
from src.utils.misc import Classes, Documents


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
        patience: int = 0
        best_loss: Optional[float] = None
        best_weights: Optional[bytes] = None

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

        print(f"\n[fit] model={self.model_name} | classes={self.num_classes} | epochs={self.num_max_epochs} | patience={self.max_patience}")
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

            for batch in data_loader_val:
                batch = {k: v.type(torch.long).to(self.device) for k, v in batch.items()}
                with torch.no_grad():
                    outputs = self.model(**batch)
                    loss, _ = outputs[:2]
                vl_loss += loss.item()
                nb_vl_steps += 1

            dev_loss = vl_loss / nb_vl_steps
            self.validation_loss.append(dev_loss)

            current_lr = scheduler.get_last_lr()[0]
            print(f"  train loss : {epoch_train_loss:.4E}")
            print(f"  val loss   : {dev_loss:.4E}  |  lr: {current_lr:.2e}")

            # --- checkpoint / patience ---
            if best_loss is None or dev_loss + self.min_val_epoch_impro_delta < best_loss:
                best_loss = dev_loss
                # buffer = io.BytesIO()
                # torch.save(self.model.state_dict(), buffer)
                # best_weights = buffer.getvalue()
                print(f"  checkpoint : val loss improved → {best_loss:.4E} (saved)")
            else:
                print(f"  patience   : {patience}/{self.max_patience}")
                if patience == self.max_patience:
                    print(f"\n[fit] early stopping triggered at epoch {self.epoch_id}.")
                    break
                patience += 1

        # Desativado até segunda ordem.
        # print("-" * 70)
        # if best_weights is not None:
        #     buffer = io.BytesIO(best_weights)
        #     self.model.load_state_dict(torch.load(buffer, map_location=self.device))
        #     print(f"[fit] best weights restored (val loss: {best_loss:.4E})")

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
        """Build the Adam optimizer with optional weight decay separation."""
        if self.full_finetuning:
            param_optimizer = list(self.model.named_parameters())
            no_decay = ['bias', 'gamma', 'beta']
            optimizer_grouped_parameters = [
                {
                    'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                    'weight_decay_rate': self.weight_decay_rate,
                },
                {
                    'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                    'weight_decay_rate': 0.0,
                },
            ]
        else:
            param_optimizer = list(self.model.classifier.named_parameters())
            optimizer_grouped_parameters = [{'params': [p for _, p in param_optimizer]}]

        return Adam(optimizer_grouped_parameters, lr=self.learning_rate)