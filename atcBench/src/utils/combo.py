"""Parsing of the `data=<dataset>_<method>` Hydra override used by run.sh."""

# Dataset names may contain "_" (yelp_reviews); method names never do, so the
# combo is split at the longest known dataset prefix and the rest is the method.
DATASETS = (
    "20ng", "acm", "agnews", "books", "dblp", "medline", "movie_review", "mpqa",
    "ohsumed", "pang_movie", "reuters90", "sst1", "sst2", "subj", "trec", "twitter",
    "vader_movie", "webkb", "wos11967", "wos5736", "yelp_2013", "yelp_reviews",
)

# Datasets evaluated with 5 folds instead of 10
FIVE_FOLD_DATASETS = {"agnews"}


def split_combo(value):
    """'yelp_reviews_fx-r20-cells-seed3' -> ('yelp_reviews', 'fx-r20-cells-seed3'); None if unknown."""
    for dataset in sorted(DATASETS, key=len, reverse=True):
        prefix = f"{dataset}_"
        if value.startswith(prefix) and len(value) > len(prefix):
            return dataset, value[len(prefix):]
    return None


def rewrite_combo_data_override(argv):
    """Rewrite `data=<dataset>_<method>` into the template data config plus field overrides."""
    rewritten = [argv[0]]
    for arg in argv[1:]:
        combo = split_combo(arg.split("=", 1)[1]) if arg.startswith("data=") else None
        if combo is None:
            rewritten.append(arg)
            continue
        dataset, method = combo
        rewritten.extend([
            "data=template",
            f"data.dataset_name={dataset}",
            f"data.is_method={method}",
        ])
        if dataset in FIVE_FOLD_DATASETS:
            rewritten.append("data.nfolds=5")
    return rewritten
