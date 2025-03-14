import numpy
import pandas
import scipy.stats
import mne_features.univariate 


class UnivariateFeatureGateway():

    def __init__(self):
        pass

    def __call__(self, feature: str, dataset: numpy.array) -> float:
        """
        :param feature: name of the feature to compute
        :param dataset: single eeg channel [1 x samples]
        """
        value = getattr(self, feature)(dataset)
        return value

    def kurtosis(self, data: numpy.array) -> float:
        return numpy.round(scipy.stats.kurtosis(data), decimals=5)

    def hjorth_mobility(self, data: numpy.array) -> float:
        return numpy.round(mne_features.univariate.compute_hjorth_mobility(data), decimals=5)

    def hjorth_complexity(self, data: numpy.array) -> float:
        return numpy.round(mne_features.univariate.compute_hjorth_complexity(data), decimals=5)

    def katz_fractal_dimension(self, data: numpy.array) -> float:
        data = numpy.expand_dims(data, axis=0)
        return numpy.round(mne_features.univariate.compute_katz_fd(data)[0], decimals=5)