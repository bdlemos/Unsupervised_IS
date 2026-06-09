"""Download datasets from Zenodo and organize them into datasets/<name>/.

Expected final layout (matches src/data/loader.py):
    datasets/<name>/
        texts.txt
        score.txt
        splits/
            split_10.pkl
            split_5.pkl
        tfidf/
            train0.gz  test0.gz
            train1.gz  test1.gz
            ...

Usage:
    python download_datasets.py              # downloads all datasets
    python download_datasets.py webkb        # downloads only webkb
"""

import argparse
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import List, Dict

# ── Dataset registry ──────────────────────────────────────────────────────────
# Each entry:
#   name       : directory name under datasets/
#   zenodo_id  : Zenodo record ID (integer)
#   zip_file   : name of the zip file in the Zenodo record


# OHSUMED (Topic)
# - boundary complexo, texto mais técnico

# WebKB (Topic)
# - pequeno, desbalanceado

# Twitter (Topic)
# - alta variabilidade dos modelos no desempenho, ruido muito alto e textos mais curtos

# MPQA (Sentiment)
# - classificação binária, desbalanceado e pequeno

# SST1 (Sentiment)
# - baixa performance, mais classes, grande potencial

# Reuters90 (Topic)
# * muitas classes, extremamente desbalanceado, performance baixa

DATASETS: List[Dict] = [
    {
        "name": "acm",
        "zenodo_id": "7555249",
        "zip_file": "acm.zip",
    },
    {
        "name": "twitter",
        "zenodo_id": "7554707",
        "zip_file": "twitter.zip",
    },
    {
        "name": "webkb",
        "zenodo_id": "7555368",
        "zip_file": "webkb.zip",
    },
    {
        "name": "reuters90",
        "zenodo_id": "7555298",
        "zip_file": "reut90.zip",
    },
    {
        "name": "mpqa",
        "zenodo_id": "7555268",
        "zip_file": "mpqa.zip",
    },
    {
        "name": "sst1",
        "zenodo_id": "7555319",
        "zip_file": "sst1.zip",
    },
    {
        "name": "ohsumed",
        "zenodo_id": "7555276",
        "zip_file": "ohsumed.zip",
    },
    {
        "name": "yelp_reviews",
        "zenodo_id": "7555396",
        "zip_file": "yelp_reviews_2L.zip",
    },
    {
        "name": "sst2",
        "zenodo_id": "7555310",
        "zip_file": "sst2.zip",
    },
    {
        "name": "dblp",
        "zenodo_id": "7555264",
        "zip_file": "dblp.zip",
    },
    {
        "name": "books",
        "zenodo_id": "7555256",
        "zip_file": "books.zip",
    },
    {
        "name": "20ng",
        "zenodo_id": "7555237",
        "zip_file": "20ng.zip",
    },
    {
        "name": "wos11967",
        "zenodo_id": "7555385",
        "zip_file": "wos11967.zip",
    },
    {
        "name": "trec",
        "zenodo_id": "7555342",
        "zip_file": "trec.zip",
    },
    {
        "name": "wos5736",
        "zenodo_id": "7555379",
        "zip_file": "wos5736.zip",
    },
    {
        "name": "pang_movie",
        "zenodo_id": "7555283",
        "zip_file": "pang_movie_2L.zip",
    },
    {
        "name": "movie_review",
        "zenodo_id": "7555273",
        "zip_file": "mr.zip",
    },
    {
        "name": "vader_movie",
        "zenodo_id": "7555354",
        "zip_file": "vader_movie_2L.zip",
    },
    {
        "name": "subj",
        "zenodo_id": "7555339",
        "zip_file": "subj.zip",
    },
    {
        "name": "agnews",
        "zenodo_id": "7555424",
        "zip_file": "agnews.zip",
    },
    {
        "name": "yelp_2013",
        "zenodo_id": "7555898",
        "zip_file": "yelp_2013.zip",
    },
    {
        "name": "medline",
        "zenodo_id": "7555820",
        "zip_file": "medline.zip",
    },
]

# ── Helpers ───────────────────────────────────────────────────────────────────
ZENODO_FILE_URL = "https://zenodo.org/records/{record_id}/files/{filename}?download=1"


def _progress(block_count: int, block_size: int, total: int) -> None:
    downloaded = block_count * block_size
    if total > 0:
        pct = min(100, downloaded * 100 // total)
        bar = "#" * (pct // 2) + "-" * (50 - pct // 2)
        print(f"\r  [{bar}] {pct:3d}%  ({downloaded // 1024 // 1024} MB)", end="", flush=True)


def download_file(url: str, dest: Path) -> None:
    print(f"  Downloading {dest.name} ...")
    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print()  # newline after progress bar


def reorganize(dataset_dir: Path) -> None:
    """Move split_*.pkl → splits/ and ensure tfidf/ exists if present."""
    splits_dir = dataset_dir / "splits"
    splits_dir.mkdir(exist_ok=True)

    for pkl in dataset_dir.glob("split_*.pkl"):
        target = splits_dir / pkl.name
        print(f"  Moving {pkl.name} → splits/")
        shutil.move(str(pkl), target)

    tfidf_dir = dataset_dir / "tfidf"
    if not tfidf_dir.exists():
        # Check if tfidf files are at root level
        gz_files = list(dataset_dir.glob("*.gz"))
        if gz_files:
            tfidf_dir.mkdir(exist_ok=True)
            for gz in gz_files:
                print(f"  Moving {gz.name} → tfidf/")
                shutil.move(str(gz), tfidf_dir / gz.name)


def download_dataset(entry: dict, base_dir: Path) -> None:
    name = entry["name"]
    record_id = entry["zenodo_id"]
    zip_name = entry["zip_file"]

    dataset_dir = base_dir / name
    if dataset_dir.exists():
        print(f"Dataset {name} already exists, skipping download")
        return

    dataset_dir.mkdir(parents=True, exist_ok=True)

    zip_path = dataset_dir / zip_name
    url = ZENODO_FILE_URL.format(record_id=record_id, filename=zip_name)

    print(f"\n{'='*60}")
    print(f"  Dataset : {name}")
    print(f"  Record  : https://zenodo.org/records/{record_id}")
    print(f"  Target  : {dataset_dir}")
    print(f"{'='*60}")

    # Download
    if zip_path.exists():
        print(f"  Zip already exists, skipping download: {zip_path}")
    else:
        download_file(url, zip_path)

    # Extract
    print(f"  Extracting {zip_name} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        # If zip has a single top-level directory, flatten it
        top_dirs = {p.split("/")[0] for p in zf.namelist() if p.strip("/")}
        members = zf.namelist()

        if len(top_dirs) == 1 and all(m.startswith(next(iter(top_dirs))) for m in members):
            # Strip top-level directory when extracting
            top = next(iter(top_dirs)) + "/"
            for member in members:
                relative = member[len(top):]
                if not relative:
                    continue
                target = dataset_dir / relative
                if member.endswith("/"):
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(member) as src, open(target, "wb") as dst:
                        shutil.copyfileobj(src, dst)
        else:
            zf.extractall(dataset_dir)

    # Remove zip to save space
    zip_path.unlink()
    print(f"  Removed {zip_name}")

    # Reorganize to match loader expectations
    reorganize(dataset_dir)

    print(f"  Done. Files in {dataset_dir}:")
    for p in sorted(dataset_dir.rglob("*"))[:20]:
        print(f"    {p.relative_to(dataset_dir)}")
    if sum(1 for _ in dataset_dir.rglob("*")) > 20:
        print("    ...")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Download datasets from Zenodo.")
    parser.add_argument(
        "datasets",
        nargs="*",
        help="Names of datasets to download (default: all). E.g.: webkb reuters",
    )
    parser.add_argument(
        "--data-dir",
        default="resources/datasets",
        help="Base directory for datasets (default: resources/datasets/)",
    )
    args = parser.parse_args()

    base_dir = Path(args.data_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    registry = {d["name"]: d for d in DATASETS}

    targets = args.datasets if args.datasets else list(registry.keys())

    unknown = [t for t in targets if t not in registry]
    if unknown:
        print(f"ERROR: unknown dataset(s): {', '.join(unknown)}")
        print(f"Available: {', '.join(registry)}")
        sys.exit(1)

    for name in targets:
        download_dataset(registry[name], base_dir)

    print(f"\nAll done. Datasets saved to '{base_dir}/'.")


if __name__ == "__main__":
    main()