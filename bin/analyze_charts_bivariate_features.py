#!/var/tmp/venv-project-1/bin/python
"""
Usage:
    analyze_charts_bivariate_features.py (--feature=<feature>)
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
from lib.analyzers import IntraBivariateChbAnalyzer, IntraBivariateSienaAnalyzer, \
                          IntraBivariateTuszAnalyzer, InterBivariateSienaAnalyzer, \
                          InterBivariateTuszAnalyzer


FEATURE = docopt(__doc__)["--feature"]
OUTPUT_DIRECTORY = os.getenv("BIOMARKERS_PROJECT_HOME")


def main():

    logging.info("Processing Siena vs TUEP")

    logging.info(f"Processing feature = {FEATURE}")
    analyzer = InterBivariateSienaAnalyzer(FEATURE)
    analyzer.bivariate_zone_bar_chart("frontal")
    analyzer.bivariate_zone_violin_chart("frontal")
    analyzer.bivariate_zone_bar_chart("temporal")
    analyzer.bivariate_zone_violin_chart("temporal")
    time.sleep(3)
    analyzer.bivariate_zone_bar_chart("parietal")
    analyzer.bivariate_zone_violin_chart("parietal")
    time.sleep(3)
    analyzer.bivariate_zone_bar_chart("occipital")
    analyzer.bivariate_zone_violin_chart("occipital")
    time.sleep(3)

    logging.info("Processing Tusz vs TUEP")

    logging.info(f"Processing feature = {FEATURE}")
    analyzer = InterBivariateTuszAnalyzer(FEATURE)
    analyzer.bivariate_zone_bar_chart("frontal")
    analyzer.bivariate_zone_violin_chart("frontal")
    time.sleep(3)
    analyzer.bivariate_zone_bar_chart("temporal")
    analyzer.bivariate_zone_violin_chart("temporal")
    time.sleep(3)
    analyzer.bivariate_zone_bar_chart("parietal")
    analyzer.bivariate_zone_violin_chart("parietal")
    time.sleep(3)
    analyzer.bivariate_zone_bar_chart("occipital")
    analyzer.bivariate_zone_violin_chart("occipital")
    time.sleep(3)

    logging.info("Processing database CHB-MIT")

    logging.info(f"Processing feature = {FEATURE}")
    analyzer = IntraBivariateChbAnalyzer(FEATURE)

    analyzer.bivariate_zone_bar_chart("left", "unknown")
    analyzer.bivariate_zone_dist_chart("left", "unknown")
    time.sleep(3)
    analyzer.bivariate_zone_bar_chart("right", "unknown")
    analyzer.bivariate_zone_dist_chart("right", "unknown")
    time.sleep(3)
    for band in ["delta", "theta", "alpha", "beta", "all"]:
        analyzer.bivariate_network_plot_average("unknown", band)
        time.sleep(3)
    time.sleep(3)

    logging.info("Processing database Siena")

    logging.info(f"Processing feature = {FEATURE}")
    analyzer = IntraBivariateSienaAnalyzer(FEATURE)
    for seizure_type in settings["siena"]["valid_seizure_types"]:
        analyzer.bivariate_zone_bar_chart("frontal", seizure_type)
        analyzer.bivariate_zone_dist_chart("frontal", seizure_type)
        time.sleep(3)
        analyzer.bivariate_zone_bar_chart("temporal", seizure_type)
        analyzer.bivariate_zone_dist_chart("temporal", seizure_type)
        time.sleep(3)
        analyzer.bivariate_zone_bar_chart("parietal", seizure_type)
        analyzer.bivariate_zone_dist_chart("parietal", seizure_type)
        time.sleep(3)
        analyzer.bivariate_zone_bar_chart("occipital", seizure_type)
        analyzer.bivariate_zone_dist_chart("occipital", seizure_type)
        time.sleep(3)
        for band in ["delta", "theta", "alpha", "beta", "all"]:
            analyzer.bivariate_network_plot_average(seizure_type, band)
            time.sleep(3)
        time.sleep(3)

    logging.info("Processing database TUSZ")

    logging.info(f"Processing feature = {FEATURE}")
    analyzer = IntraBivariateTuszAnalyzer(FEATURE)
    for seizure_type in settings["tusz"]["valid_seizure_types"]:
        analyzer.bivariate_zone_bar_chart("frontal", seizure_type)
        analyzer.bivariate_zone_dist_chart("frontal", seizure_type)
        time.sleep(3)
        analyzer.bivariate_zone_bar_chart("temporal", seizure_type)
        analyzer.bivariate_zone_dist_chart("temporal", seizure_type)
        time.sleep(3)
        analyzer.bivariate_zone_bar_chart("parietal", seizure_type)
        analyzer.bivariate_zone_dist_chart("parietal", seizure_type)
        time.sleep(3)
        analyzer.bivariate_zone_bar_chart("occipital", seizure_type)
        analyzer.bivariate_zone_dist_chart("occipital", seizure_type)
        time.sleep(3)
        for band in ["delta", "theta", "alpha", "beta", "all"]:
            analyzer.bivariate_network_plot_average(seizure_type, band)
            time.sleep(3)
        time.sleep(3)


if __name__ == "__main__":
    main()
