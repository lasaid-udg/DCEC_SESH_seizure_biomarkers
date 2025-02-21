import numpy
import logging
from statsmodels.tsa import stattools


class StatisticalTests():

    def __init__(self):
        self.significance_level = 0.05

    def run_kpss_test(self, time_serie: numpy.array):
        """
        Computes the Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test
        for the null hypothesis that data serie is level or trend stationary
        :param time_serie: the data serie to test
        """
        kpss_stat, p_value, _, _ = stattools.kpss(time_serie, regression="c")
        details = f"p_value = {p_value}, significance level = {self.significance_level}"

        if p_value > 0.05:
            logging.info(f"KPSS test, null hyphotesis was not rejected, {details}")
            is_stationary = True
        else:
            logging.info(f"KPSS test, null hyphotesis was rejected, {details}")
            is_stationary = False
        
        return kpss_stat, p_value, is_stationary

    def run_adf_test(self, time_serie: numpy.array):
        """
        Computes the Augmented Dickey-Fuller test for the null hypothesis 
        that there is a unit root
        :param time_serie: the data serie to test
        """
        adf_stat, p_value, _, _, _, _ = stattools.adfuller(time_serie, regression="c")
        details = f"p_value = {p_value}, significance level = {self.significance_level}"

        if p_value > 0.05:
            logging.info(f"ADF test, null hyphotesis was not rejected, {details}")
            is_stationary = False
        else:
            logging.info(f"ADF test, null hyphotesis was rejected, {details}")
            is_stationary = True
        
        return adf_stat, p_value, is_stationary

    def check_stationarity(self, time_serie: numpy.array):
        """
        Run KPSS and ADF test to check for trend and difference stationarity
        :param time_serie: the data serie to test
        """ 
        _, _, is_stationary_kpss = self.run_kpss_test(time_serie)
        _, _, is_stationary_adf = self.run_adf_test(time_serie)

        if is_stationary_adf and is_stationary_kpss:
            logging.info("The time serie is stationary")
        elif not is_stationary_adf and not is_stationary_kpss:
            logging.info("The time serie is stationary")
        elif not is_stationary_adf and is_stationary_kpss:
            logging.info("The series is trend stationary  (not strict stationarity).")
        elif is_stationary_adf and not is_stationary_kpss:
            logging.info("The series is difference stationary (not strict stationarity)")
        