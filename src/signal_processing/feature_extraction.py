"""Feature Extraction utilities."""
import librosa
from sklearn.base import BaseEstimator, TransformerMixin
from madmom.processors import SequentialProcessor, ParallelProcessor
from madmom.audio import SignalProcessor, FramedSignalProcessor
from madmom.audio.stft import ShortTimeFourierTransformProcessor
from madmom.audio.spectrogram import (
    FilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor,
    SpectrogramDifferenceProcessor)
import numpy as np


# define pre-processor
class OnsetPreProcessor(SequentialProcessor):

    def __init__(self, sample_rate=44100., frame_rate=100, fmin=30, fmax=17000,
                 norm_filters=True, unique_filters=True,
                 frame_sizes=(1024, 2048, 4096), num_bands=(3, 6, 12)):
        """
        Resample to a fixed sample rate in order to get always the same number
        of filter bins.
        """
        self.sample_rate = sample_rate
        sig = SignalProcessor(num_channels=1, sample_rate=self.sample_rate)
        # process multi-resolution spec & diff in parallel
        multi = ParallelProcessor([])
        for frame_size, num_bands in zip(frame_sizes, num_bands):
            # split audio signal in overlapping frames
            frames = FramedSignalProcessor(
                frame_size=frame_size, fps=frame_rate)
            # compute STFT
            stft = ShortTimeFourierTransformProcessor()
            # filter the magnitudes
            filt = FilteredSpectrogramProcessor(
                num_bands=num_bands, fmin=fmin, fmax=fmax,
                norm_filters=norm_filters, unique_filters=unique_filters)
            # scale them logarithmically
            spec = LogarithmicSpectrogramProcessor(log=np.log10, mul=1, add=1)
            # stack positive differences
            diff = SpectrogramDifferenceProcessor(
                diff_ratio=.25, positive_diffs=True, stack_diffs=np.hstack)
            # process each frame size with spec and diff sequentially
            multi.append(SequentialProcessor((frames, stft, filt, spec, diff)))
        # instantiate a SequentialProcessor
        super(OnsetPreProcessor, self).__init__((sig, multi, np.hstack))

    def process(self, data, **kwargs):
        """
        Process the data sequentially with the defined processing chain.

        Parameters
        ----------
        data : depends on the first processor of the processing chain
            Data to be processed.
        kwargs : dict, optional
            Keyword arguments for processing.

        Returns
        -------
        depends on the last processor of the processing chain
            Processed data.

        """
        # sequentially process the data
        y, sr = librosa.load(data, sr=self.sample_rate, mono=True)
        y = librosa.util.normalize(y, norm=np.inf)
        return super().process(y)
