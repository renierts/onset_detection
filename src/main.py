"""
Main Code to reproduce the results in the paper "Feature Engineering and
Stacked Echo State Networks for Musical Onset Detection".
"""

# Authors: Peter Steiner <peter.steiner@tu-dresden.de>,
# License: BSD 3-Clause

import logging

import matplotlib.pyplot as plt
import seaborn as sns
import madmom

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import PredefinedSplit
from sklearn.metrics import make_scorer
from scipy.stats import uniform
from sklearn.base import clone
from sklearn.utils.fixes import loguniform
from dataset import OnsetDataset
from input_to_node import ClusterInputToNode
from metrics import cosine_distance
from signal_processing import OnsetPreProcessor
from model_selection import PredefinedTrainValidationTestSplit
import numpy as np
from joblib import dump, load
from itertools import product

from pyrcn.echo_state_network import ESNRegressor
from pyrcn.model_selection import SequentialSearchCV


def evaluate_onsets(predictions, annotations):
    evals = []
    for ann, det in zip(annotations, predictions):
        e = madmom.evaluation.onsets.OnsetEvaluation(
            det, ann, combine=0.03, window=0.025)
        evals.append(e)
    se = madmom.evaluation.onsets.OnsetSumEvaluation(evals)
    me = madmom.evaluation.onsets.OnsetMeanEvaluation(evals)
    return se, me


LOGGER = logging.getLogger(__name__)


def main(plot=False, frame_sizes=(1024, 2048, 4096), num_bands=(3, 6, 12)):
    """
    This is the main function to reproduce all visualizations and models for
    the paper "Feature Engineering and Stacked Echo State Networks for Musical
    Onset Detection".

    It is controlled via command line arguments:

    Params
    ------
    plot : bool, default=False
        Create all plots in the publication.
    frame_sizes: default=(1024, 2048, 4096)
        The window sizes to be considered for the feature extraction.
    num_bands:
        The number of filters per octave to be considered for each window size.

    Returns
    -------
    results : dict
        Results that are stored in data/results.dat
    """
    decoded_frame_sizes = "_".join(map(str, frame_sizes))
    LOGGER.info("Loading the dataset...")
    pre_processor = OnsetPreProcessor(frame_sizes=frame_sizes,
                                      num_bands=num_bands)
    dataset = OnsetDataset(
        path=r"/scratch/ws/1/s2575425-onset-detection/onset_detection/data",
        audio_suffix=".flac")
    X, y = dataset.return_X_y(pre_processor=pre_processor)
    test_fold = np.zeros(shape=X.shape)
    start_idx = 0
    for k, fold in enumerate(dataset.folds):
        test_fold[start_idx:start_idx + len(fold)] = k
        start_idx += len(fold)
    cv_vali = PredefinedTrainValidationTestSplit(test_fold=test_fold)
    cv_test = PredefinedTrainValidationTestSplit(test_fold=test_fold,
                                                 validation=False)
    LOGGER.info("... done!")

    if plot:
        fig, axs = plt.subplots()
        sns.heatmap(data=X[0].T, ax=axs, square=False)
        axs.invert_yaxis()
        [axs.axvline(x=ann*100, color='w', linestyle=':', linewidth=1)
         for ann in dataset.annotations[0]]

    LOGGER.info(f"Creating ESN pipeline...")
    initial_esn_params = {
        'hidden_layer_size': 50, 'k_in': 10, 'input_scaling': 0.4,
        'input_activation': 'identity', 'bias_scaling': 0.0,
        'spectral_radius': 0.0, 'leakage': 1.0, 'k_rec': 10,
        'reservoir_activation': 'tanh', 'bidirectional': False,
        'alpha': 1e-5, 'random_state': 42}

    base_esn = ESNRegressor(**initial_esn_params)
    LOGGER.info("... done!")
    # Run model selection
    LOGGER.info(f"Performing the optimization...")
    step1_params = {'input_scaling': uniform(loc=1e-2, scale=1),
                    'spectral_radius': uniform(loc=0, scale=2)}
    step2_params = {'leakage': uniform(1e-1, 1e0)}
    step3_params = {'bias_scaling': uniform(loc=0, scale=2)}

    kwargs_step1 = {
        'n_iter': 200, 'random_state': 42, 'verbose': 1, 'n_jobs': -1,
        'scoring': make_scorer(cosine_distance, greater_is_better=False),
        "cv": cv_vali}
    kwargs_step2 = {
        'n_iter': 50, 'random_state': 42, 'verbose': 1, 'n_jobs': -1,
        'scoring': make_scorer(cosine_distance, greater_is_better=False),
        "cv": cv_vali}
    kwargs_step3 = {
        'n_iter': 50, 'random_state': 42, 'verbose': 1, 'n_jobs': -1,
        'scoring': make_scorer(cosine_distance, greater_is_better=False),
        "cv": cv_vali}

    searches = [
        ('step1', RandomizedSearchCV, step1_params, kwargs_step1),
        ('step2', RandomizedSearchCV, step2_params, kwargs_step2),
        ('step3', RandomizedSearchCV, step3_params, kwargs_step3)]

    try:
        search = load(f'./results/sequential_search_basic_esn_'
                      f'{decoded_frame_sizes}.joblib')
    except FileNotFoundError:
        search = SequentialSearchCV(base_esn, searches=searches).fit(X, y)
        dump(search, f'./results/sequential_search_basic_esn_'
                     f'{decoded_frame_sizes}.joblib')
    LOGGER.info("... done!")

    base_esn = ESNRegressor(input_to_node=ClusterInputToNode(),
                            **initial_esn_params)

    try:
        search = load(f'./results/sequential_search_kmeans_esn_'
                      f'{decoded_frame_sizes}.joblib')
    except FileNotFoundError:
        search = SequentialSearchCV(base_esn, searches=searches).fit(X, y)
        dump(search, f'./results/sequential_search_kmeans_esn_'
                     f'{decoded_frame_sizes}.joblib')

    kwargs_final = {
        'n_iter': 50, 'random_state': 42, 'verbose': 1, 'n_jobs': -1,
        'scoring': make_scorer(cosine_distance, greater_is_better=False)}
    param_distributions_final = {'alpha': loguniform(1e-5, 1e1)}
    hidden_layer_sizes = (
        50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600)
    bi_directional = (False, True)

    for hidden_layer_size, bidirectional in product(
            hidden_layer_sizes, bi_directional):
        params = {"hidden_layer_size": hidden_layer_size,
                  "bidirectional": bidirectional}
        for k, (train_index, vali_index) in enumerate(cv_vali.split()):
            test_fold = np.zeros(
                shape=(len(train_index) + len(vali_index), ), dtype=int)
            test_fold[:len(train_index)] = -1
            ps = PredefinedSplit(test_fold=test_fold)
            try:
                esn = load(f"./results/km_esn_{decoded_frame_sizes}_"
                           f"{hidden_layer_size}_{bidirectional}_{k}.joblib")
                print(esn.best_estimator_.regressor.alpha)
            except FileNotFoundError:
                esn = RandomizedSearchCV(
                    estimator=clone(search.best_estimator_).set_params(
                        **params), cv=ps,
                    param_distributions=param_distributions_final,
                    **kwargs_final).fit(
                    X[np.hstack((train_index, vali_index))],
                    y[np.hstack((train_index, vali_index))])
                dump(esn, f"./results/km_esn_{decoded_frame_sizes}_"
                          f"{hidden_layer_size}_{bidirectional}_{k}.joblib")

    if plot:
        plt.show()
    return search.all_cv_results_


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--frame_sizes", type=int, nargs="+",
                        default=(1024, 2048, 4096))
    parser.add_argument("--num_bands", type=int, nargs="+", default=(3, 6, 12))
    args = vars(parser.parse_args())
    logging.basicConfig(format="%(asctime)s - [%(levelname)8s]: %(message)s",
                        handlers=[
                            logging.FileHandler("main.log", encoding="utf-8"),
                            logging.StreamHandler()
                        ], level=logging.DEBUG)
    LOGGER.setLevel(logging.DEBUG)
    main(**args)
    exit(0)
