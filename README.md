# Supplemental Code Repository for the research paper "Feature Engineering and Stacked Echo State Networks for Musical Onset Detection"
## Metadata
- Authors: [Peter Steiner](mailto:peter.steiner@tu-dresden.de), Azarakhsh Jalalvand, 
  Simon Stone, Peter Birkholz
- Conference: 2020 25th International Conference on Pattern Recognition (ICPR), Milan,
  2021
- Weblink: [10.1109/ICPR48806.2021.9413205](http://dx.doi.org/10.1109/ICPR48806.2021.9413205)

## Summary and Contents
In music analysis, one of the most fundamental tasks is note onset detection - detecting
the beginning of new note events. As the target function of onset detection is related to
other tasks, such as beat tracking or tempo estimation, onset detection is the basis for
such related tasks. Furthermore, it can help to improve Automatic Music Transcription
(AMT). Typically, different approaches for onset detection follow a similar outline: An
audio signal is transformed into an Onset Detection Function (ODF), which should have
rather low values (i.e. close to zero) for most of the time but with pronounced peaks at
onset times, which can then be extracted by applying peak picking algorithms on the ODF.
In the recent years, several kinds of neural networks were used successfully to compute
the ODF from feature vectors. Currently, Convolutional Neural Networks (CNNs) define the
state of the art. In this paper, we build up on an alternative approach to obtain a ODF
by Echo State Networks (ESNs), which have achieved comparable results to CNNs in several
tasks, such as speech and image recognition. In contrast to the typical iterative
training procedures of deep learning architectures, such as CNNs or networks consisting
of Long-Short-Term Memory Cells (LSTMs), in ESNs only a very small part of the weights is
easily trained in one shot using linear regression.

## File list
- The following scripts are provided in this repository
    - `scripts/run.sh`: UNIX Bash script to reproduce the results in the paper.
    - `scripts/run_jupyter-lab.sh`: UNIX Bash script to start the Jupyter Notebook for 
   the paper.
    - `scripts/run.bat`: Windows batch script to reproduce the results in the paper.
    - `scripts/run_jupyter-lab.bat`: Windows batch script to start the Jupyter Notebook 
  for the paper.
- The following python code and modules are provided in `src`
    - `src/dataset`: Utility functions for storing and loading data and models.
    - `src/model_selection`: Wrapper class for `sklearn.model_selection.PredefinedSplit`
  to support splitting a dataset in training/validation/test.
    - `src/signal_processing`: Utility functions to do the feature 
      extraction using madmom and librosa.
    - `src/main.py`: The main script to reproduce all results.
- `requirements.txt`: Text file containing all required Python modules to be installed. 
- `README.md`: The README displayed here.
- `LICENSE`: Textfile containing the license for this source code.
- `data/`: The empty directory, in which the dataset is getting downloaded.
- `results/`:
    - (Pre)-trained models
    - ...

## Usage
The easiest way to reproduce the results is to use a service like 
[Binder](https://mybinder.org/) and run the Jupyter Notebook (if available).

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/renierts/onset_detection/HEAD)

To run the scripts or to start the Jupyter Notebook locally, at first, please ensure 
that you have a valid Python distribution installed on your system. Here, at least Python
3.8 is required.

You can then call `run_jupyter-lab.ps1` or `run_jupyter-lab.sh`. This will install a new 
[Python venv](https://docs.python.org/3/library/venv.html), which is our recommended way 
of getting started.

To manually reproduce the results, you should create a new Python venv as well.
Therefore, you can run the script `create_venv.sh` on a UNIX bash or `create_venv.ps1`
that will automatically install all packages from PyPI. Afterwards, just type 
`source .virtualenv/bin/activate` in a UNIX bash or `.virtualenv/Scripts/activate.ps1`
in a PowerShell.

At first, we import required Python modules. Then, we start loading the data. 
The dataset can be either downloaded from here or manually downloaded from 
[here](https://drive.google.com/file/d/1ICEfaZ2r_cnqd3FLNC5F_UOEUalgV7cv/view?usp=sharing).

```python
import matplotlib.pyplot as plt
import seaborn as sns

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
```

After downloading the dataset, please extract it to the ``data`` directory, which should
in the end contain three subdirectories ``annotations``, ``audio``, ``splits``,
respectively.

In any case, the ``OnsetDataset`` object is responsible to providing the dataset. It is
initialized with a path to the dataset and optional arguments, such as custom file 
endings for the different files to be searched for. Importantly, we deal with ``.flac``
files.

From the dataset class, we load the spectrograms and the target labels in ``(X, y)``,
where each element is a spectrogram and the corresponding target sequence.

The dataset has a predefined split in training, validation and test folds. To utilize the
split, we prepare the ``test_fold``, which assigns each input and target sequence to the
correct fold.

```python
frame_sizes=(1024, 2048, 4096)
num_bands=(3, 6, 12)

dataset = OnsetDataset(
  path="/scratch/ws/1/s2575425-onset-detection/onset_detection/data",
  audio_suffix=".flac")
X, y = dataset.return_X_y(pre_processor=OnsetPreProcessor(frame_sizes=frame_sizes, 
                                                          num_bands=num_bands))
test_fold = np.zeros(shape=X.shape)
start_idx = 0
for k, fold in enumerate(dataset.folds):
  test_fold[start_idx:start_idx + len(fold)] = k
  start_idx += len(fold)
cv_vali = PredefinedTrainValidationTestSplit(test_fold=test_fold)
cv_test = PredefinedTrainValidationTestSplit(test_fold=test_fold,
                                             validation=False)
```

We optimize a model using a sequence of random searches. The target for the optimization
is to maximize the cross correlation between the computed output and the ground truth
output. This randomized approach is slightly different from the grid search described in
the paper. Consequently, the resulting hyper-parameters are slightly better, and the 
results will also be slightly different. However, the main outline is still the same.

```python
decoded_frame_sizes = "_".join(map(str, frame_sizes))

initial_esn_params = {
  'hidden_layer_size': 50, 'k_in': 10, 'input_scaling': 0.4,
  'input_activation': 'identity', 'bias_scaling': 0.0,
  'spectral_radius': 0.0, 'leakage': 1.0, 'k_rec': 10,
  'reservoir_activation': 'tanh', 'bidirectional': False,
  'alpha': 1e-5, 'random_state': 42}

base_esn = ESNRegressor(**initial_esn_params)
# Run model selection
step1_params = {'input_scaling': uniform(loc=1e-2, scale=1),
                'spectral_radius': uniform(loc=0, scale=2)}
step2_params = {'leakage': uniform(loc=1e-2, scale=0.99)}
step3_params = {'bias_scaling': uniform(loc=0, scale=2)}

kwargs_step1 = {
  'n_iter': 200, 'random_state': 42, 'verbose': 10, 'n_jobs': -1,
  'scoring': make_scorer(cosine_distance, greater_is_better=False),
  "cv": cv_vali}
kwargs_step2 = {
  'n_iter': 50, 'random_state': 42, 'verbose': 10, 'n_jobs': -1,
  'scoring': make_scorer(cosine_distance, greater_is_better=False),
  "cv": cv_vali}
kwargs_step3 = {
  'n_iter': 50, 'random_state': 42, 'verbose': 10, 'n_jobs': -1,
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
```

Next, we fit models with increased reservoir sizes and an optional bidirectional mode.
For each configuration, we optimize the regularization parameter.

One model for each fold is fitted to stay in line with the reference publications.

```python
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
```

Finally, we predict the test data.

```python
y_pred = esn.predict(X_test)
```

After you finished your experiments, please do not forget to deactivate the venv by 
typing `deactivate` in your command prompt.

The aforementioned steps are summarized in the script `main.py`. The easiest way to
reproduce the results is to either download and extract this Github repository in the
desired directory, open a Linux Shell and call `run.sh` or open a Windows PowerShell and
call `run.ps1`. 

In that way, again, a [Python venv](https://docs.python.org/3/library/venv.html) is 
created, where all required packages (specified by `requirements.txt`) are installed.
Afterwards, the script `main.py` is excecuted with all default arguments activated in
order to reproduce all results in the paper.

If you want to suppress any options, simply remove the particular option.

## Acknowledgements
```
The parameter optimizations were performed on a Bull Cluster at the Center for 
Information Services and High Performance Computing (ZIH) at TU Dresden.

This research was financed by Europäischer Sozialfonds (ESF) and the Free State of 
Saxony (Application number: 100327771) and Ghent University.

```


## License and Referencing
This program is licensed under the BSD 3-Clause License. If you in any way use this
code for research that results in publications, please cite our original
article listed above.

You can use the following BibTeX entry
```
@INPROCEEDINGS{9413205,
  author={Steiner, Peter and Jalalvand, Azarakhsh and Stone, Simon and Birkholz, Peter},
  booktitle={2020 25th International Conference on Pattern Recognition (ICPR)},
  title={Feature Engineering and Stacked Echo State Networks for Musical Onset Detection},
  year={2021},
  volume={},
  number={},
  pages={9537--9544},
  keywords={},
  doi={10.1109/ICPR48806.2021.9413205},
  ISSN={1051-4651},
  month={Jan},
}
```
## Appendix
For any questions, do not hesitate to open an issue or to drop a line to [Peter Steiner](mailto:peter.steiner@tu-dresden.de)
