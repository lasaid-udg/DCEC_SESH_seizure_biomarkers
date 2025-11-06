#!/var/tmp/venv-project-1/bin/python
"""
Usage:
    analyze_charts_univariate_features.py (--feature=<feature>)
"""
import os
import sys
import time
import logging
import warnings
from docopt import docopt
warnings.filterwarnings("ignore")
sys.path.append("../")
from lib import settings
from lib.analyzers import IntraUnivariateChbAnalyzer, IntraUnivariateSienaAnalyzer, \
                          IntraUnivariateTuszAnalyzer, InterUnivariateSienaAnalyzer, \
                          InterUnivariateTuszAnalyzer, InterUnivariateChbAnalyzer


FEATURE = docopt(__doc__)["--feature"]
OUTPUT_DIRECTORY = os.getenv("BIOMARKERS_PROJECT_HOME")


def main():

    logging.info("Processing CHB-MIT vs TUEP")

    logging.info(f"Processing feature = {FEATURE}")
    analyzer = InterUnivariateChbAnalyzer(FEATURE)
    analyzer.univariate_zone_bar_chart()
    analyzer.univariate_zone_violin_chart()
    time.sleep(3)

    logging.info("Processing Siena vs TUEP")

    logging.info(f"Processing feature = {FEATURE}")
    analyzer = InterUnivariateSienaAnalyzer(FEATURE)
    analyzer.univariate_zone_bar_chart()
    analyzer.univariate_zone_violin_chart()
    time.sleep(3)

    logging.info("Processing Tusz vs TUEP")

    logging.info(f"Processing feature = {FEATURE}")
    analyzer = InterUnivariateTuszAnalyzer(FEATURE)
    analyzer.univariate_zone_bar_chart()
    analyzer.univariate_zone_violin_chart()
    time.sleep(3)

    logging.info("Processing database CHB-MIT")

    logging.info(f"Processing feature = {FEATURE}")
    analyzer = IntraUnivariateChbAnalyzer(FEATURE)
    analyzer.univariate_zone_bar_chart()
    analyzer.univariate_zone_dist_chart()
    for band in ["delta", "theta", "alpha", "beta", "gamma", "all"]:
        analyzer.univariate_topo_plot_average(band)
        time.sleep(3)

    logging.info("Processing database Siena")

    logging.info(f"Processing feature = {FEATURE}")
    analyzer = IntraUnivariateSienaAnalyzer(FEATURE)
    analyzer.univariate_zone_bar_chart()
    analyzer.univariate_zone_dist_chart()
    time.sleep(3)
    for band in ["delta", "theta", "alpha", "beta", "gamma", "all"]:
        analyzer.univariate_topo_plot_average(band)
        time.sleep(3)

    logging.info("Processing database TUSZ")

    logging.info(f"Processing feature = {FEATURE}")
    analyzer = IntraUnivariateTuszAnalyzer(FEATURE)
    analyzer.univariate_zone_bar_chart()
    analyzer.univariate_zone_dist_chart()
    for band in ["delta", "theta", "alpha", "beta", "gamma", "all"]:
        analyzer.univariate_topo_plot_average(band)
        time.sleep(3)
    time.sleep(3)


if __name__ == "__main__":
    main()
