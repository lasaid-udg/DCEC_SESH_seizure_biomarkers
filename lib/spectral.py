import numpy
import scipy.fft


class FourierAnalysis():

    def __init__(self, sampling_frequency: int):
        self.sampling_frequency = sampling_frequency
    
    def run_fast_fourier_transform(self, signal: numpy.array) -> tuple:
        """
        Return the absolute values of the fourier spectrum.
        :param signal: matrix with the set of signals
        """
        spectral_components = numpy.abs(scipy.fft.rfft(signal))
        frequency_range = scipy.fft.rfftfreq(signal.shape[1], 1 / self.sampling_frequency)
        return frequency_range, spectral_components
