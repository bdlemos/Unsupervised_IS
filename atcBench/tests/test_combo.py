import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from src.utils.combo import rewrite_combo_data_override, split_combo


@pytest.mark.parametrize("value,expected", [
    ("yelp_reviews_fx-r21.3-cells-seed3", ("yelp_reviews", "fx-r21.3-cells-seed3")),
    ("vader_movie_entropy-sublinear-ae-is", ("vader_movie", "entropy-sublinear-ae-is")),
    ("sst1_ablation-cluster-sublinear-random", ("sst1", "ablation-cluster-sublinear-random")),
    ("wos11967_random-rate-40-seed2", ("wos11967", "random-rate-40-seed2")),
    ("unknownds_no-is", None),
    ("sst2_", None),
])
def test_split_combo(value, expected):
    assert split_combo(value) == expected


def test_rewrite_and_agnews_folds():
    argv = ["main.py", "data=agnews_fx-entropy-re", "model=modernbert"]
    assert rewrite_combo_data_override(argv) == [
        "main.py", "data=template", "data.dataset_name=agnews",
        "data.is_method=fx-entropy-re", "data.nfolds=5", "model=modernbert",
    ]


def test_unknown_combo_is_left_untouched():
    argv = ["main.py", "data=template", "model=lr"]
    assert rewrite_combo_data_override(argv) == argv
