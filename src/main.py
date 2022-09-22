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
from sklearn.utils.fixes import loguniform
from sklearn.base import clone
from scipy.stats import uniform
from dataset import OnsetDataset
from metrics import cosine_distance
from signal_processing import OnsetPreProcessor
from model_selection import PredefinedTrainValidationTestSplit
import numpy as np
from joblib import dump, load
import pandas as pd
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
        path=r"/scratch/ws/s2575425-onset-detection/onset_detection/data",
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

    if plot:
        df = pd.DataFrame(search.all_cv_results_["step1"])
        fig, axs = plt.subplots()
        sns.scatterplot(
            data=df, x="param_spectral_radius", y="param_input_scaling",
            hue="mean_test_score", ax=axs, palette="RdBu")

        norm = plt.Normalize(
            df["mean_test_score"].min(), df["mean_test_score"].max())
        sm = plt.cm.ScalarMappable(cmap="RdBu", norm=norm)
        sm.set_array([])
        plt.xlim((0, 2.05))
        plt.ylim((0, 1.05))
        # Remove the legend and add a colorbar
        axs.get_legend().remove()
        axs.figure.colorbar(sm)

        df = pd.DataFrame(search.all_cv_results_["step2"])
        fig, axs = plt.subplots()
        sns.lineplot(data=df, x="param_leakage", y="mean_test_score", ax=axs)
        axs.set_xlim((0.0, 1.0))

        df = pd.DataFrame(search.all_cv_results_["step3"])
        fig, axs = plt.subplots()
        sns.lineplot(data=df, x="param_bias_scaling", y="mean_test_score",
                     ax=axs)
        axs.set_xlim((0.0, 1.0))

    kwargs_final = {
        'n_iter': 50, 'random_state': 42, 'verbose': 1, 'n_jobs': -1,
        'scoring': make_scorer(cosine_distance, greater_is_better=False)}
    param_distributions_final = {'alpha': loguniform(1e-5, 1e1)}
    hidden_layer_sizes = (
        50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600)
    bi_directional = (False, True)
    thresholds = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

    for hidden_layer_size, bidirectional in product(
            hidden_layer_sizes, bi_directional):
        params = {"hidden_layer_size": hidden_layer_size,
                  "bidirectional": bidirectional}
        LOGGER.info(hidden_layer_size, bidirectional)
        for k, (train_index, vali_index) in enumerate(cv_vali.split()):
            test_fold = np.zeros(
                shape=(len(train_index) + len(vali_index), ), dtype=int)
            test_fold[:len(train_index)] = -1
            ps = PredefinedSplit(test_fold=test_fold)
            try:
                esn = load(f"./results/esn_{decoded_frame_sizes}_"
                           f"{hidden_layer_size}_{bidirectional}_{k}.joblib")
            except FileNotFoundError:
                esn = RandomizedSearchCV(
                    estimator=clone(search.best_estimator_).set_params(
                        **params), cv=ps,
                    param_distributions=param_distributions_final,
                    **kwargs_final).fit(
                    X[np.hstack((train_index, vali_index))],
                    y[np.hstack((train_index, vali_index))])
                dump(esn, f"./results/esn_{decoded_frame_sizes}_"
                          f"{hidden_layer_size}_{bidirectional}_{k}.joblib")

    columns = [
        "param_hidden_layer_size", "param_bidirectional", "param_threshold",
        "params",
        "split0_avg_test_score", "split1_avg_test_score",
        "split2_avg_test_score", "split3_avg_test_score",
        "split4_avg_test_score", "split5_avg_test_score",
        "split6_avg_test_score", "split7_avg_test_score",
        "split0_sum_test_score", "split1_sum_test_score",
        "split2_sum_test_score", "split3_sum_test_score",
        "split4_sum_test_score", "split5_sum_test_score",
        "split6_sum_test_score", "split7_sum_test_score",
        "split0_avg_train_score", "split1_avg_train_score",
        "split2_avg_train_score", "split3_avg_train_score",
        "split4_avg_train_score", "split5_avg_train_score",
        "split6_avg_train_score", "split7_avg_train_score",
        "split0_sum_train_score", "split1_sum_train_score",
        "split2_sum_train_score", "split3_sum_train_score",
        "split4_sum_train_score", "split5_sum_train_score",
        "split6_sum_train_score", "split7_sum_train_score",
        "mean_avg_test_score", "mean_sum_test_score",
        "mean_avg_train_score", "mean_sum_train_score",
        "std_avg_test_score", "std_sum_test_score",
        "std_avg_train_score", "std_sum_train_score"]
    df_results = pd.DataFrame(columns=columns)
    df_results[
        ["param_hidden_layer_size", "param_bidirectional", "param_threshold",
         "params"]] = [[hidden_layer_size, bidirectional, threshold,
                        {"hidden_layer_size": hidden_layer_size,
                         "bidirectional": bidirectional,
                         "threshold": threshold}]
                       for hidden_layer_size, bidirectional, threshold in
                       product(hidden_layer_sizes, bi_directional, thresholds)]

    for hidden_layer_size, bidirectional in product(
            hidden_layer_sizes, bi_directional):
        params = {"hidden_layer_size": hidden_layer_size,
                  "bidirectional": bidirectional}
        print(hidden_layer_size, bidirectional)
        for k, (train_index, test_index) in enumerate(cv_test.split()):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            esn = load(f"../results/esn_{decoded_frame_sizes}_"
                       f"{hidden_layer_size}_{bidirectional}_{k}.joblib")
            y_train_pred = esn.predict(X_train)
            y_test_pred = esn.predict(X_test)
            for thr in thresholds:
                rnn_peak_picking = \
                    madmom.features.onsets.OnsetPeakPickingProcessor(
                        threshold=thr, pre_max=0.01, post_max=0.01,
                        smooth=0.07, combine=0.03)
                detections = [rnn_peak_picking(act) for act in y_train_pred]
                annotations = [
                    ann for ann in np.asarray(
                        dataset.annotations, dtype=object)[train_index]]
                se, me = evaluate_onsets(detections, annotations)
                df_results.loc[
                    (df_results['param_hidden_layer_size'] == hidden_layer_size) &
                    (df_results['param_bidirectional'] == bidirectional) &
                    (df_results['param_threshold'] == thr), f"split{k}_avg_train_score"] = me.metrics
                df_results.loc[
                    (df_results['param_hidden_layer_size'] == hidden_layer_size) &
                    (df_results['param_bidirectional'] == bidirectional) &
                    (df_results['param_threshold'] == thr), f"split{k}_sum_train_score"] = se.metrics
                detections = [rnn_peak_picking(act) for act in y_test_pred]
                annotations = [
                    ann for ann in np.asarray(
                        dataset.annotations, dtype=object)[test_index]]
                se, me = evaluate_onsets(detections, annotations)
                df_results.loc[
                    (df_results['param_hidden_layer_size'] == hidden_layer_size) &
                    (df_results['param_bidirectional'] == bidirectional) &
                    (df_results['param_threshold'] == thr), f"split{k}_avg_test_score"] = me.metrics
                df_results.loc[
                    (df_results['param_hidden_layer_size'] == hidden_layer_size) &
                    (df_results['param_bidirectional'] == bidirectional) &
                    (df_results['param_threshold'] == thr), f"split{k}_sum_test_score"] = se.metrics

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
