import os
import numpy
import pandas
import scipy.stats 


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