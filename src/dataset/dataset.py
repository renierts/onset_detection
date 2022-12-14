"""Class to hold the onset dataset."""
import gdown
import zipfile
import os
import warnings
from pathlib import Path
import numpy as np
import madmom


class OnsetDataset(object):
    """
    The class OnsetDataset is adapted from the ISMIR 2018 tutorial "Deep
    Learning for Music Information Retrieval" (
    https://github.com/slychief/ismir2018_tutorial), particularly on the Onset
    part.

    It tries to find all files of the onset detection dataset in a given
    directory. If this fails, the files are downloaded from Google Drive and
    then stored.

    Parameters
    ----------
    path : str
        /path/to/onset/data
        The path to the onset dataset, which contains the subdirectories
        "annotations", "audio" and "splits"
    audio_suffix : str, default = ".wav".
        The suffix of the audio files.
    annotation_suffix : str, default = ".onsets".
        The suffix of the annotation files.
    """
    def __init__(self, path, audio_suffix='.wav', annotation_suffix='.onsets'):
        """Construct the OnsetDataset."""
        if len(os.listdir(path)) < 2:
            gdown.download(id="1ICEfaZ2r_cnqd3FLNC5F_UOEUalgV7cv",
                           output=f"{path}/tmp.zip")
            with zipfile.ZipFile(f"{path}/tmp.zip", 'r') as zip_ref:
                zip_ref.extractall(path)
            os.remove(f"{path}/tmp.zip")
            for file_name in os.listdir(
                    os.path.join(path, "onsets_ISMIR_2012")):
                os.rename(os.path.join(path, "onsets_ISMIR_2012", file_name),
                          os.path.join(path, file_name))
            os.rmdir(os.path.join(path, "onsets_ISMIR_2012"))
        self.path = path
        # populate lists containing audio and annotation files
        audio_files = sorted(
            Path(os.path.join(self.path, "audio")).rglob(f"*{audio_suffix}"))
        annotation_files = sorted(
            Path(os.path.join(self.path, "annotations")).rglob(
                f"*{annotation_suffix}"))
        # match annotation to audio files
        unlabeled_audio_files = sorted(
            Path(os.path.join(self.path, "unlabeled_audio")).rglob(
                f"*{audio_suffix}"))
        self.files = []
        self.audio_files = []
        self.unlabeled_audio_files = []
        self.annotation_files = []
        for audio_file, annotation_file in zip(audio_files, annotation_files):
            base_audio_file = os.path.splitext(os.path.basename(audio_file))
            base_annotation_file = os.path.splitext(
                os.path.basename(annotation_file))
            # search matching audio file
            if base_audio_file[0] == base_annotation_file[0]:
                self.audio_files.append(audio_file)
                self.annotation_files.append(annotation_file)
                # save the base name
                self.files.append(os.path.basename(base_annotation_file[0]))
            else:
                warnings.warn(
                    f"skipping {annotation_file}, no audio file found")
        for audio_file in unlabeled_audio_files:
            base_audio_file = os.path.splitext(os.path.basename(audio_file))
            self.unlabeled_audio_files.append(audio_file)
        self._load_splits()
        self._load_annotations()

    def _load_splits(self, fold_suffix='.fold'):
        """
        Load the onset splits.

        Parameters
        ----------
        fold_suffix : str, default = '.fold'.
            The suffix of the text files that contain the cross validation
            splits.
        """
        path = os.path.join(self.path, "splits")
        self.split_files = sorted(Path(path).rglob(f"*{fold_suffix}"))
        # populate folds
        self.folds = []
        for i, split_file in enumerate(self.split_files):
            fold_idx = []
            with open(split_file) as f:
                for file in f:
                    file = file.strip()
                    # get matching file idx
                    try:
                        idx = self.files.index(file)
                        fold_idx.append(idx)
                    except ValueError:
                        warnings.warn(
                            f'no matching audio/annotation files: {file}')
                        continue
            # set indices for fold
            self.folds.append(np.array(fold_idx))

    def return_X_y(self, pre_processor, kernel=(0.5, 1.0, 0.5)):
        """
        Return the dataset in a PyRCN-conform way.

        Parameters
        ----------
        pre_processor : object
            The object to preprocess each audio file.
        kernel : Tuple, default = (0.5, 1.0, 0.5)
            Kernel to convolve the target with. If the target should not be
            widened, pass (0.0, 1.0, 0.0) as the kernel.

        Returns
        -------
        X : np.ndarray(shape=(n_sequences, ), dtype=object)
            The extracted sequences. Each element of X is a numpy array of
            shape (n_samples, n_features), where n_samples is the sequence
            length.
        y : np.ndarray(shape=(n_sequences, ), dtype=object)
            The pre-processed targets. Each element of y is a numpy array of
            shape (n_samples, 1), where n_samples is the sequence  length.
        """
        X = np.empty(shape=(len(self.audio_files), ), dtype=object)
        y = np.empty(shape=(len(self.audio_files), ), dtype=object)
        for k, (audio_file, annotation) in enumerate(
                zip(self.audio_files, self.annotations)):
            X[k], y[k] = self._pre_process(
                audio_file, annotation, pre_processor, kernel)
        return X, y

    def return_X(self, pre_processor):
        """
        Return the dataset in a PyRCN-conform way.

        Parameters
        ----------
        pre_processor : object
            The object to preprocess each audio file.

        Returns
        -------
        X : np.ndarray(shape=(n_sequences, ), dtype=object)
            The extracted sequences. Each element of X is a numpy array of
            shape (n_samples, n_features), where n_samples is the sequence
            length.
        """
        X = np.empty(shape=(len(self.unlabeled_audio_files), ), dtype=object)
        for k, audio_file in enumerate(self.unlabeled_audio_files):
            X[k] = self._pre_process_unlabeled(audio_file, pre_processor)
        return X

    @staticmethod
    def _pre_process(audio_file, annotation, pre_processor,
                     kernel=(0.5, 1.0, 0.5)):
        """
        Pre-process the dataset.

        Parameters
        ----------
        audio_file : Union[Path, str]
            Full path to the audio file to be pre-processed.
        annotation : np.ndarray
            The onset events, i.e., the time stamps at which onsets occur.
        pre_processor : object
            The object to preprocess each audio file.
        kernel : Tuple, default = (0.5, 1.0, 0.5)
            Kernel to convolve the target with. If the target should not be
            widened, pass (0.0, 1.0, 0.0) as the kernel.

        Returns
        -------
        X : np.ndarray, shape = (n_samples, n_features)
            The features extracted from the audio file.
        y : np.ndarray, shape = (n_samples, 1)
            The quantized and widened targets that correspond to the audio
            file.
        """
        X = pre_processor(str(audio_file))
        y = madmom.utils.quantize_events(
            annotation, fps=100, length=X.shape[0])
        y = madmom.audio.signal.smooth(y, np.array(kernel))
        return X, y

    @staticmethod
    def _pre_process_unlabeled(audio_file, pre_processor):
        """
        Pre-process the dataset.

        Parameters
        ----------
        audio_file : Union[Path, str]
            Full path to the audio file to be pre-processed.
        pre_processor : object
            The object to preprocess each audio file.

        Returns
        -------
        X : np.ndarray, shape = (n_samples, n_features)
            The features extracted from the audio file.
        """
        X = pre_processor(str(audio_file))
        return X

    def _load_annotations(self):
        """Load the onset annotations from an annotation file."""
        self.annotations = [
            np.loadtxt(file, ndmin=1) for file in self.annotation_files]
