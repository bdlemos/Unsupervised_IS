from src.main.python.utils.general import get_data, get_splits, checkpoint_splits, translate_train_idxinfold
from src.main.python.utils.arguments import arguments
from src.main.python.utils.save_results import save_results
import argparse
from datetime import datetime
import time
import numpy as np
import gc
import io
import os
import pandas as pd
from collections import Counter
from src.main.python.iSel import perplexity_is, biois, autoencoder_is, iforest_is, gmm_is, cluster_is
from src.main.python.iSel import random_is, no_is, adaptive_is, adaptive_v2_is, adaptive_cluster_is

import socket

import logging
import logging.config

logger = logging.getLogger(__name__)

beta = 0.60
theta = 0.2
low_percentile = 40
high_percentile = 70

def get_selector(method: str):

    #Baselines
    if method == 'no-is':   return no_is.NoIS()
    if method == 'biois':   return biois.BIOIS(beta=0.25, theta=0.50)
    if method == 'random-is': return random_is.RandomIS(selection_rate=0.5, random_state=28101994)

    #unsupervised
    if method == 'perplexity-is': return perplexity_is.PerplexityIS(n_topics=10, low_percentile=low_percentile, high_percentile=high_percentile, beta=beta, theta=theta)
    if method == 'pca-autoencoder-is': return PCAutoencoder_is.PCAutoencoderIS(
                                                                        n_epochs=20, batch_size=64, bottleneck_ratio=0.05,
                                                                        beta=beta, theta=theta, low_percentile=low_percentile, high_percentile=high_percentile
                                                                    )
    if method == 'autoencoder-is': return autoencoder_is.AutoencoderIS(
                                                                        n_epochs=20, batch_size=64, bottleneck_ratio=0.05,
                                                                        beta=beta, theta=theta, low_percentile=low_percentile, high_percentile=high_percentile
                                                                    )
    if method == 'gmm-is': return gmm_is.GMMIS(low_percentile=low_percentile, high_percentile=high_percentile, beta=beta, theta=theta)

    # Adaptive variants — hyperparameters estimated automatically from the
    # score distribution (imbalance-aware). Three flavours, one per base scorer.
    if method == 'adaptive-perplexity': return adaptive_is.AdaptiveIS(base_method='autoencoder')

    # AdaptiveV2 — quartile-density strategy
    if method == 'adaptive-v2-perplexity': return adaptive_v2_is.AdaptiveV2IS(base_method='autoencoder')

    # Adaptive cluster-based IS — clusters are formed in the original feature space, and then the adaptive strategy is applied within each cluster.
    if method == 'adaptive-cluster-is': return adaptive_cluster_is.AdaptiveClusterIS(base_method='autoencoder', n_clusters=30)

    return None


def get_selection(X, y, fold, args, info):

    method = args.method

    total = Counter(y)
    logger.debug(f"Total instances number: {total}")

    selector = get_selector(method)

    selector.fit(X, y)

    if hasattr(selector, 'low_percentile'):
        info['low_percentile'] = selector.low_percentile
    if hasattr(selector, 'high_percentile'):
        info['high_percentile'] = selector.high_percentile

    if hasattr(selector, 'beta'):
        info['beta'] = selector.beta
    if hasattr(selector, 'theta'):
        info['theta'] = selector.theta

    logger.info("Result: ", Counter(y[selector.sample_indices_]))

    return selector.sample_indices_


def main():

    gc.collect()

    args, info = arguments()
    logger.info(str(args))

    print(f"{args.splitdir}/split_{args.folds}.pkl")
    splits_df = get_splits(f"{args.splitdir}/split_{args.folds}.pkl")

    splits_to_save = {c: [] for c in splits_df.columns if c.endswith("idxs")}

    for f in range(args.folds):
    # for f in range(1):

        logger.info("Fold {}".format(f))
        print("Fold {}".format(f))

        splits_to_save['test_idxs'].append(splits_df.loc[f].test_idxs)


        X_train, y_train, _, _, _ = get_data(args.inputdir, f)
        t = len(y_train)

        ti = time.time()

        idxs_docs = get_selection(X_train, y_train, f, args, info)

        s = len(y_train[idxs_docs])
        r = (t-s)/t

        info['time_for_reduce'].append(time.time() - ti)
        print(info['time_for_reduce'])
        info['original_len'].append(t)
        info['reduced_len'].append(s)
        info['reducion'].append(r)

        splits_to_save['train_idxs'].append(idxs_docs)

    logger.info(f"time: {np.mean(info['time_for_reduce'])}")
    logger.info(f"time std: {np.std(info['time_for_reduce'])}")
    logger.info(f"reducion: {np.mean(info['reducion'])}")
    logger.info(f"reducion std: {np.std(info['reducion'])}")

    splits_to_save_df = pd.DataFrame(data=splits_to_save)

    filename = f"{args.outputdir}/split_{args.folds}_{args.method}_idxinfold.pkl"

    checkpoint_splits(
        splits_df=splits_to_save_df,
        filename = filename
    )

    splits_to_save_df_traslated = translate_train_idxinfold(
        splits_to_save_df, splits_df)

    checkpoint_splits(
        splits_df=splits_to_save_df_traslated,
        filename=filename.replace("_idxinfold", "")
    )

    if args.save:
        save_results(args, info)

    print("END")
    exit()


if __name__ == '__main__':
    main()
