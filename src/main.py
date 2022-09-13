"""
Main Code to reproduce the results in the paper
'Template Repository for Research Papers with Python Code'.
"""

# Authors: Peter Steiner <peter.steiner@tu-dresden.de>,
# License: BSD 3-Clause

import logging

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer
from sklearn.utils.fixes import loguniform
from scipy.stats import uniform
from dataset import OnsetDataset
from metrics import cosine_distance
from signal_processing import OnsetPreProcessor
from model_selection import PredefinedTrainValidationTestSplit
import numpy as np
from joblib import dump, load

from pyrcn.echo_state_network import ESNRegressor
from pyrcn.model_selection import SequentialSearchCV


LOGGER = logging.getLogger(__name__)


def main(plot=False, export=False):
    """
    This is the main function to reproduce all visualizations and models for
    the paper "Template Repository for Research Papers with Python Code".

    It is controlled via command line arguments:

    Params
    ------
    plot : bool, default=False
        Create all plots in the publication.
    export: default=False
        Store the results in ``data/results.dat``
    serialize:
        Store the fitted model in ``data/model.joblib``

    Returns
    -------
    results : dict
        Results that are stored in data/results.dat
    """

    LOGGER.info("Loading the dataset...")
    pre_processor = OnsetPreProcessor()
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
        fig, axs = plt.subplots(2, 1)
        sns.heatmap(data=X[0].T, ax=axs[0], square=False, )
        axs[0].invert_yaxis()
        plt.tight_layout()
        sns.lineplot(x=list(range(len(y[0]))), y=y[0], ax=axs[1])

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
    step1_params = {
        'esn__regressor__input_scaling': uniform(loc=1e-2, scale=1),
        'esn__regressor__spectral_radius': uniform(loc=0, scale=2)}
    step2_params = {'esn__regressor__leakage': uniform(1e-1, 1e0)}
    step3_params = {
        'esn__regressor__bias_scaling': uniform(loc=0, scale=2)}

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
        search = load(f'./results/sequential_search_basic_esn.joblib')
    except FileNotFoundError:
        search = SequentialSearchCV(base_esn, searches=searches).fit(X, y)
        dump(search, f'./results/sequential_search_basic_esn.joblib')
    LOGGER.info("... done!")

    if plot:
        y_pred = search.predict(X)
        sns.lineplot(
            x=list(range(len(y_pred[0]))), y=y_pred[0].flatten(), ax=axs[1])

    if plot:
        plt.show()
    return search.all_cv_results_


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # TODO: Specify command line arguments to add runtime options for the code.
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--export", action="store_true")
    args = vars(parser.parse_args())
    logging.basicConfig(format="%(asctime)s - [%(levelname)8s]: %(message)s",
                        handlers=[
                            logging.FileHandler("main.log", encoding="utf-8"),
                            logging.StreamHandler()
                        ], level=logging.DEBUG)
    LOGGER.setLevel(logging.DEBUG)
    main(**args)
    exit(0)
