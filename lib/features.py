import numpy
import pandas
import scipy.stats
import mne_features.univariate


class UnivariateFeatureGateway():

    def __init__(self):
        pass

    def __call__(self, feature: str, dataset: numpy.array, sampling_frequency: int=None) -> float:
        """
        :param feature: name of the feature to compute
        :param dataset: single eeg channel [1 x samples]
        :param sampling_frequency: sampling_frequency [Hz]
        """
        if feature != "power_spectral_density":
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

    def power_spectral_density(self, data: numpy.array, sampling_frequency: int) -> float:
        data = numpy.expand_dims(data, axis=0)
        densities = mne_features.univariate.compute_pow_freq_bands(sampling_frequency,
                                                                   data,
                                                                   numpy.array([0.5, 4., 8., 13., 30.]),
                                                                   normalize=True,
                                                                   psd_method="welch")
        densities = numpy.append(densities, [0])
        return numpy.round(densities, decimals=5)
