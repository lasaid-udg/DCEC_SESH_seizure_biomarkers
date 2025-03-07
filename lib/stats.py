import os
import pandas
import numpy
import logging
from pandasql import sqldf
from statsmodels.tsa import stattools
from statsmodels.stats import diagnostic
from .visuals import plot_stationarity_bar_chart


class StatisticalTests():

    def __init__(self):
        self.significance_level = 0.05

    def run_lilliefors_test(self, samples: numpy.array):
        """
        Computes the Lilliefors test for the null hypothesis 
        that sample comes from a normal distribution
        :param values: distribution samples
        """
        _, p_value = diagnostic.lilliefors(samples)
        details = f"p_value = {p_value}, significance level = {self.significance_level}"

        if p_value > 0.05:
            logging.info(f"Lilliefors test, null hyphotesis was not rejected, {details}")
            return p_value, "Data comes from normal distribution"
        else:
            logging.info(f"Lilliefors test, null hyphotesis was rejected, {details}")
            return p_value, "Data doesn't come from normal distribution"
        

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
        that there is a unit root (non-stationarity)
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
            return "Stationary"
        elif not is_stationary_adf and not is_stationary_kpss:
            logging.info("The time serie is not stationary")
            return "Not stationary"
        elif not is_stationary_adf and is_stationary_kpss:
            logging.info("The series is trend stationary  (not strict stationarity)")
            return "Trend stationary"
        elif is_stationary_adf and not is_stationary_kpss:
            logging.info("The series is difference stationary (not strict stationarity)")
            return "Difference stationary"


class StationarityFile():

    def __init__(self, database: str, windows_lengths: list):
        """
        :param database: any of [siena, tusz, chb]
        :param windows_lengths: list of lengths (in seconds) to be compared
        """
        self.database = database
        self.windows_lenghts = windows_lengths
        self.stationarity_results = os.path.join(os.getenv("BIOMARKERS_PROJECT_HOME"), "reports")
    
    @property
    def stationarity_results(self) -> pandas.DataFrame:
        return self._stationarity_results

    @stationarity_results.setter
    def stationarity_results(self, directory: str) -> None:
        """
        :param directory: full path to the stationarity results directory
        """
        self._stationarity_results = []
        for length in self.windows_lenghts:
            filename = f"stationarity_{self.database}_{length}s.csv"
            full_path = os.path.join(directory, filename)
            self._stationarity_results.append(pandas.read_csv(full_path))
    
    def bar_chart(self):
        stationarity_results = []
        for result in self._stationarity_results:
            grouped_result = sqldf("SELECT result, SUM(count) AS count FROM result GROUP BY result")
            stationarity_results.append(grouped_result)
        plot_stationarity_bar_chart(stationarity_results, self.windows_lenghts)