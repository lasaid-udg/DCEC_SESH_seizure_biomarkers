import numpy
import networkx
import scipy.stats
import scipy.signal
import mne_features.univariate
from typing import Iterable


class FeatureGateway():

    def __init__(self):
        pass

    def __call__(self, feature: str, dataset: numpy.array, sampling_frequency: int = None) -> float:
        """
        :param feature: name of the feature
        :param dataset: single eeg channel [1 x samples]
        :param sampling_frequency: sampling_frequency [Hz]
        """
        if feature not in ["power_spectral_density", "coherence"]:
            value = getattr(self, feature)(dataset)
        else:
            value = getattr(self, feature)(dataset, sampling_frequency)
        return value

    def kurtosis(self, data: numpy.array) -> float:
        return numpy.round(scipy.stats.kurtosis(data), decimals=5)

    def skew(self, data: numpy.array) -> float:
        return numpy.round(scipy.stats.skew(data), decimals=5)

    def hjorth_mobility(self, data: numpy.array) -> float:
        return numpy.round(mne_features.univariate.compute_hjorth_mobility(data), decimals=5)

    def hjorth_complexity(self, data: numpy.array) -> float:
        return numpy.round(mne_features.univariate.compute_hjorth_complexity(data), decimals=5)

    def katz_fractal_dimension(self, data: numpy.array) -> float:
        data = numpy.expand_dims(data, axis=0)
        return numpy.round(mne_features.univariate.compute_katz_fd(data)[0], decimals=5)

    def approximate_entropy(self, data: numpy.array) -> float:
        data = numpy.expand_dims(data, axis=0)
        return numpy.round(mne_features.univariate.compute_app_entropy(data)[0], decimals=5)

    def power_spectral_density(self, data: numpy.array, sampling_frequency: int) -> numpy.array:
        data = numpy.expand_dims(data, axis=0)
        densities = mne_features.univariate.compute_pow_freq_bands(sampling_frequency,
                                                                   data,
                                                                   numpy.array([0.5, 4., 8., 13., 30.]),
                                                                   normalize=True,
                                                                   psd_method="welch")
        densities = numpy.append(densities, [0])
        return numpy.round(densities, decimals=5)

    def phase_lock_value(self, data: numpy.array) -> float:
        return numpy.round(mne_features.bivariate.compute_phase_lock_val(data, include_diag=False),
                           decimals=5)

    def coherence(self, data: numpy.array, sampling_frequency: int) -> numpy.array:
        number_channels, _ = data.shape
        number_coefficients = number_channels * (number_channels - 1) // 2
        mean_coherences = [numpy.empty((number_coefficients,)),
                           numpy.empty((number_coefficients,)),
                           numpy.empty((number_coefficients,)),
                           numpy.empty((number_coefficients,)),
                           numpy.empty((number_coefficients,))]

        for s, i, j in self.get_upper_triangle_indices(number_channels):
            frequencies, coherence = scipy.signal.coherence(data[i, :], data[j, :],
                                                            fs=sampling_frequency, nperseg=128)
            band_counter = 0
            previous_idx = 0
            bands = [4, 8, 12, 30]

            for next_idx, frequency in enumerate(frequencies):
                if frequency > bands[band_counter]:
                    mean_coherences[band_counter][s] = numpy.mean(coherence[previous_idx: next_idx])
                    previous_idx = next_idx
                    band_counter += 1
                if band_counter == len(bands):
                    break
            mean_coherences[-1][s] = numpy.mean(coherence)

        return (numpy.round(coherence, decimals=5) for coherence in mean_coherences)

    def global_efficiency(self, graph: list) -> float:
        graph = networkx.Graph(graph)
        return numpy.round(networkx.global_efficiency(graph), 4)

    def local_efficiency(self, graph: list) -> float:
        graph = networkx.Graph(graph)
        return numpy.round(networkx.local_efficiency(graph), 4)

    def get_upper_triangle_indices(self, number_channels: int) -> Iterable[tuple]:
        """
        Enumeration of the upper-triangular part of a squre matrix.
        """
        position = -1
        for idx1 in range(number_channels):
            for idx2 in range(idx1, number_channels):
                if idx1 == idx2:
                    continue
                else:
                    position += 1
                    yield position, idx1, idx2
