import os
import pandas
import numpy
import logging
import scipy.stats
import scikit_posthocs
from pandasql import sqldf
import statsmodels.api
from statsmodels.tsa import stattools
from statsmodels.stats import diagnostic
from . import settings
from .visuals import plot_stationarity_bar_chart


class StatisticalTests():

    def __init__(self):
        self.significance_level = 0.05

    def run_posthoc_nemenyi(self, samples: list):
        """
        Calculate pairwise comparisons using Nemenyi post hoc test.
        The null hypothesis is that mean value for each of the populations is equal.
        :param samples: ranked samples
        """
        sequential_p_values = []
        data = numpy.array(samples)
        bonferroni_factor = (len(samples) * len(samples) - 1) / 2
        p_values = scikit_posthocs.posthoc_nemenyi_friedman(data.T)
        p_values = bonferroni_factor * p_values

        for idx1, category1 in zip(range(len(samples)), settings["intra_windows_categories"]):
            for idx2, category2 in zip(range(idx1, len(samples)),
                                       settings["intra_windows_categories"][idx1:]):
                if category1[3] == category2[3]:
                    continue
                sequential_p_values.append([f"{category1[3]}-{category2[3]}", p_values.iloc[idx1][idx2]])

        return sequential_p_values

    def run_friedman_test(self, samples: list):
        """
        Computes the Friedman test for the null hypothesis that repeated samples of
        the same individuals have the same distribution.
        :param samples: ranked samples
        """
        _, p_value = scipy.stats.friedmanchisquare(*samples)
        details = f"p_value = {p_value}, significance level = {self.significance_level}"

        if p_value > 0.05:
            logging.info(f"Friedman test, null hyphotesis was not rejected, {details}")
            return numpy.round(p_value, 4), "Samples come from same distribution"
        else:
            logging.info(f"Friedman test, null hyphotesis was rejected, {details}")
            return numpy.round(p_value, 4), "Samples doesn't come from same distribution"

    def run_posthoc_dunns(self, samples: list):
        """
        Calculate pairwise comparisons using Dunn's post hoc test.
        The null hypothesis is that the probability of observing a randomly selected value
        from the first group that is larger than a randomly selected value from the second group is 0.5
        (therefore both distributions have the same median)
        :param samples: samples
        """
        sequential_p_values = []
        p_values = scikit_posthocs.posthoc_dunn(samples, p_adjust="bonferroni")
        for idx1, category1 in zip(range(len(samples)), settings["inter_windows_categories"]):
            for idx2, category2 in zip(range(idx1, len(samples)),
                                       settings["inter_windows_categories"][idx1:]):
                if category1[0] == category2[0]:
                    continue
                sequential_p_values.append([f"{category1[0]}-{category2[0]}", p_values.iloc[idx1][idx2 + 1]])
        return sequential_p_values

    def run_kruskal_wallis(self, samples: list):
        """
        Computes the Kruskal Wallis H-test for the null hypothesis that
        the population median of all of the groups are equal.
        :param samples: samples
        """
        _, p_value = scipy.stats.kruskal(*samples)
        details = f"p_value = {p_value}, significance level = {self.significance_level}"

        if p_value > 0.05:
            logging.info(f"Kruskal test, null hyphotesis was not rejected, {details}")
            return numpy.round(p_value, 4), "Population median are equal"
        else:
            logging.info(f"Kruskal test, null hyphotesis was rejected, {details}")
            return numpy.round(p_value, 4), "Population median are not equal"

    def run_lilliefors_test(self, samples: numpy.array):
        """
        Computes the Lilliefors test for the null hypothesis
        that sample comes from a normal distribution
        :param samples: distribution samples
        """
        _, p_value = diagnostic.lilliefors(samples)
        details = f"p_value = {p_value}, significance level = {self.significance_level}"

        if p_value > 0.05:
            logging.info(f"Lilliefors test, null hyphotesis was not rejected, {details}")
            return p_value, "Data comes from normal distribution"
        else:
            logging.info(f"Lilliefors test, null hyphotesis was rejected, {details}")
            return p_value, "Data doesn't come from normal distribution"

    def run_white_test_for_heteroscedasticity(self, time_serie: numpy.array):
        """
        Computes the White’s Lagrange Multiplier Test. The null hyphotesis
        is that time series is homoscedastic (i.e. it has a time-independent variance),
        Another interpretation is that variance of the errors in a regression model is constant.
        """
        time_range = numpy.linspace(0, len(time_serie) / 256, len(time_serie))
        time_range = statsmodels.api.add_constant(time_range)
        model = statsmodels.api.OLS(time_serie, time_range).fit()
        _, _, fvalue, f_p_value = diagnostic.het_white(model.resid,  model.model.exog)
        details = f"p_value = {f_p_value}, significance level = {self.significance_level}"

        if f_p_value > 0.05:
            logging.info(f"White test, null hyphotesis was not rejected, {details}")
            is_variance_stationary = True
        else:
            logging.info(f"White test, null hyphotesis was rejected, {details}")
            is_variance_stationary = False
        return f_p_value, is_variance_stationary

    def run_kpss_test(self, time_serie: numpy.array):
        """
        Computes the Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test
        for the null hypothesis that data serie is level or trend stationary
        :param time_serie: the data serie to test
        """
        _, p_value, _, _ = stattools.kpss(time_serie, regression="c")
        details = f"p_value = {p_value}, significance level = {self.significance_level}"

        if p_value > 0.05:
            logging.info(f"KPSS test, null hyphotesis was not rejected, {details}")
            is_trend_stationary = True
        else:
            logging.info(f"KPSS test, null hyphotesis was rejected, {details}")
            is_trend_stationary = False

        return p_value, is_trend_stationary

    def run_adf_test(self, time_serie: numpy.array):
        """
        Computes the Augmented Dickey-Fuller test for the null hypothesis
        that there is a unit root (non-stationarity)
        :param time_serie: the data serie to test
        """
        _, p_value, _, _, _, _ = stattools.adfuller(time_serie, regression="c")
        details = f"p_value = {p_value}, significance level = {self.significance_level}"

        if p_value > 0.05:
            logging.info(f"ADF test, null hyphotesis was not rejected, {details}")
            is_stationary = False
        else:
            logging.info(f"ADF test, null hyphotesis was rejected, {details}")
            is_stationary = True

        return p_value, is_stationary

    def check_stationarity(self, time_serie: numpy.array):
        """
        Run KPSS and White test to check for trend and difference stationarity
        :param time_serie: the data serie to test
        """
        _, is_trend_stationary = self.run_kpss_test(time_serie)
        _, is_variance_stationary = self.run_white_test_for_heteroscedasticity(time_serie)

        if is_variance_stationary and is_trend_stationary:
            logging.info("The time serie is stationary")
            return "Stationary"
        elif not is_variance_stationary and not is_trend_stationary:
            logging.info("The time serie is not stationary")
            return "Not stationary"
        elif not is_variance_stationary and is_trend_stationary:
            logging.info("The series is trend stationary only (not strict stationarity)")
            return "Trend stationary"
        elif is_variance_stationary and not is_trend_stationary:
            logging.info("The series is variance stationary only (not strict stationarity)")
            return "Variance stationary"


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
