import numpy
import scipy.fft
from typing import Tuple


class FourierAnalysis():

    def __init__(self, sampling_frequency: int):
        """
        :param sampling_frequency: sampling_frequency [Hz]
        """
        self.sampling_frequency = sampling_frequency
    
    def run_fast_fourier_transform(self, signal: numpy.array) -> Tuple[numpy.array, numpy.array]:
        """
        Return the complex values of the Fast Fourier Transform.
        :param signal: matrix with the eeg recording [channels x samples]
        """
        spectral_components = scipy.fft.rfft(signal)
        frequency_range = scipy.fft.rfftfreq(signal.shape[1], 1 / self.sampling_frequency)
        return frequency_range, spectral_components
