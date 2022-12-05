"""Feature Extraction utilities."""
import librosa
from madmom.processors import SequentialProcessor, ParallelProcessor
from madmom.audio import SignalProcessor, FramedSignalProcessor
from madmom.audio.stft import ShortTimeFourierTransformProcessor
from madmom.audio.spectrogram import (
    FilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor,
    SpectrogramDifferenceProcessor)
import numpy as np


# define pre-processor
class OnsetPreProcessor(SequentialProcessor):
    """
    Processor class for the feature extraction.

    Parameters
    ----------
    sample_rate : float, default = 44100
        The sampling frequency in Hertt to be utilized. If the original
        sampling frequency is different from the parameter, the audio file is
        resampled prior to the consecutive steps.
    frame_rate : float, default = 100
        The frame rate in Hz, i.e., 1 / window_shift.
    fmin : float, default = 30
        The minimum frequency to be considered in the filterbank.
    fmax : float, default = 17000
        The maximum frequency to be considered in the analysis.
    norm_filters : bool, default = True
        Whether to normalize the filter to an area of 1.
    unique_filters : bool, default = True
        Whether to remove duplicate filters from the filterbank.
    frame_sizes : tuple, default = (1024, 2048, 4096)
        The frame sizes in samples to be used for the multi-resolution feature
        extraction.
    num_bands : tuple, default = (3, 6, 12)
        The number of bands to be used for the different window sizes.
    """

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
