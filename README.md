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
The dataset can be either downloaded from here or 

Since the data is stored as a Pandas dataframe, we can theoretically multiple features.
Here, we restrict the data features to the living area. With the function
`select_features`, we obtain numpy arrays and the feature transformer that can also be
used for transforming the test data later. Next, we normalize them to zero mean and
unitary variance.

```python
from dataset import OnsetDataset
from signal_processing import OnsetPreProcessor
from model_selection import PredefinedTrainValidationTestSplit
import numpy as np


dataset = OnsetDataset(
  path="/scratch/ws/1/s2575425-onset-detection/onset_detection/data",
  audio_suffix=".flac")
X, y = dataset.return_X_y(pre_processor=OnsetPreProcessor())
test_fold = np.zeros(shape=X.shape)
start_idx = 0
for k, fold in enumerate(dataset.folds):
  test_fold[start_idx:start_idx + len(fold)] = k
  start_idx += len(fold)
cv_vali = PredefinedTrainValidationTestSplit(test_fold=test_fold)
cv_test = PredefinedTrainValidationTestSplit(test_fold=test_fold,
                                             validation=False)
```



```python
from sklearn.preprocessing import StandardScaler
from preprocessing import select_features


X, y, feature_trf = select_features(
    df=training_data, input_features=["GrLivArea"], target="SalePrice")
scaler = StandardScaler().fit(X)
X_train = scaler.transform(X)
y_train = y
```

We optimize a model using a random search.

```python
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils.fixes import loguniform
from scipy.stats import uniform

from pyrcn.extreme_learning_machine import ELMRegressor


model = RandomizedSearchCV(
    estimator=ELMRegressor(input_activation="relu", random_state=42,
                           hidden_layer_size=50),
    param_distributions={"input_scaling": uniform(loc=0, scale=2),
                         "bias_scaling": uniform(loc=0, scale=2),
                         "alpha": loguniform(1e-5, 1e1)},
    random_state=42, n_iter=200, refit=True).fit(X, y)
```

We load and transform test data.

```python
from file_handling import load_data


test_data = load_data("../data/test.csv")
X = feature_trf.transform(test_data)
X_test = scaler.transform(X)
```

Finally, we predict the test data.

```python
y_pred = model.predict(X_test)
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
This research was supported by
```
Nobody
```


## License and Referencing
This program is licensed under the BSD 3-Clause License. If you in any way use this
code for research that results in publications, please cite our original
article listed above.

More information about licensing can be found [here](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository)
and [here](https://en.wikipedia.org/wiki/License).

You can use the following BibTeX entry
```
@inproceedings{src:Steiner-22,
  author    = "Peter Steiner",
  title     = "Template Repository for Research Papers",
  booktitle = "Proceedings of the Research Seminar",
  year      = 2022,
  pages     = "1--6",
}
```
## Appendix
For any questions, do not hesitate to open an issue or to drop a line to [Peter Steiner](mailto:peter.steiner@tu-dresden.de)
