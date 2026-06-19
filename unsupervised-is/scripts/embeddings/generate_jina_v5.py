import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import gzip
import time
import argparse
import numpy as np
import torch
from pathlib import Path
from sklearn.datasets import dump_svmlight_file
from transformers import AutoModel


import torch

DATASETS_BASE = Path("/data/bernardolemos/datasets")
MODEL_ID = "jinaai/jina-embeddings-v5-text-small"
REPR_DIR = "jina-v5"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
TRUNCATE_DIM = 256
DTYPE = torch.bfloat16


def load_texts(dataset_path: Path) -> list[str]:
    with open(dataset_path / "texts.txt", "r", encoding="utf-8", errors="ignore") as f:
        return [line.rstrip("\n") for line in f]


def load_split(split_csv: Path) -> list[tuple[list[int], list[int]]]:
    splits = []
    with open(split_csv, encoding="utf-8", errors="ignore") as f:
        for line in f:
            train_part, test_part = line.strip().split(";")
            train_idx = list(map(int, train_part.split()))
            test_idx = list(map(int, test_part.split()))
            splits.append((train_idx, test_idx))
    return splits


def encode_texts(model, texts: list[str]) -> np.ndarray:
    all_embeddings = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        with torch.no_grad():
            emb = model.encode(
                batch,
                task="clustering",
                truncate_dim=TRUNCATE_DIM,
            )
            if hasattr(emb, "detach"):
                emb = emb.detach().cpu().to(torch.float32).numpy()
            else:
                emb = np.asarray(emb, dtype=np.float32)
            all_embeddings.append(emb)
    return np.vstack(all_embeddings)


def save_svmlight_gz(X: np.ndarray, labels: np.ndarray, out_path: Path):
    buf = []
    for label, row in zip(labels, X):
        pairs = " ".join(f"{i+1}:{v:.16g}" for i, v in enumerate(row))
        buf.append(f"{int(label)} {pairs}\n")
    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        f.writelines(buf)


def process_dataset(dataset_path: Path, model, n_folds: int):
    split_csv = dataset_path / "splits" / f"split_{n_folds}.csv"
    if not split_csv.exists():
        return

    out_dir = dataset_path / REPR_DIR
    out_dir.mkdir(exist_ok=True)

    texts = load_texts(dataset_path)
    scores_path = dataset_path / "score.txt"
    with open(scores_path, "r", encoding="utf-8", errors="ignore") as f:
        labels = np.array([line.strip() for line in f], dtype=np.int32)

    splits = load_split(split_csv)

    fold_times = []
    encode_times = []

    for fold, (train_idx, test_idx) in enumerate(splits):
        print(f"  [{dataset_path.name}] fold={fold} ...")
        train_texts = [texts[i] for i in train_idx]
        test_texts = [texts[i] for i in test_idx]
        train_labels = labels[train_idx]
        test_labels = labels[test_idx]

        t0 = time.time()
        train_emb = encode_texts(model, train_texts)
        test_emb = encode_texts(model, test_texts)
        encode_time = time.time() - t0
        encode_times.append(encode_time)

        t1 = time.time()
        save_svmlight_gz(train_emb, train_labels, out_dir / f"train{fold}.gz")
        save_svmlight_gz(test_emb, test_labels, out_dir / f"test{fold}.gz")
        fold_times.append(time.time() - t0)

    times_path = out_dir / "times.csv"
    encode_row = " ".join(f"{t}" for t in encode_times)
    total_row = " ".join(f"{t}" for t in fold_times)
    with open(times_path, "w") as f:
        f.write(encode_row + "\n")
        f.write(total_row + "\n")

    print(f"  [{dataset_path.name}] fold={n_folds} done — avg encode {np.mean(encode_times):.2f}s/fold")


def load_model() -> AutoModel:
    model = AutoModel.from_pretrained(
        MODEL_ID,
        dtype=DTYPE,
        trust_remote_code=True,
    )
    model.to(DEVICE)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Dataset names to process. Defaults to all.")
    parser.add_argument("--folds", nargs="*", type=int, default=[5, 10],
                        help="Which split sizes to generate (default: 5 10).")
    args = parser.parse_args()

    datasets = args.datasets or sorted(
        d.name for d in DATASETS_BASE.iterdir() if d.is_dir()
    )

    print(f"Loading model {MODEL_ID} ...")
    model = load_model()

    for ds_name in datasets:
        ds_path = DATASETS_BASE / ds_name
        if not ds_path.is_dir():
            print(f"[SKIP] {ds_name} — not found")
            continue
        print(f"Processing {ds_name} ...")
        for n_folds in args.folds:
            process_dataset(ds_path, model, n_folds)

    print("Done.")


if __name__ == "__main__":
    main()
