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
from lib.analyzers import IntraUnivariateChbAnalyzer, IntraUnivariateSienaAnalyzer, IntraUnivariateTuszAnalyzer, \
                          InterUnivariateSienaAnalyzer, InterUnivariateTuszAnalyzer, MlUnivariateFeatureAnalyzer


FEATURE = docopt(__doc__)["--feature"]
OUTPUT_DIRECTORY = os.getenv("BIOMARKERS_PROJECT_HOME")


def main():

    logging.info("Processing Siena vs TUEP")

    logging.info(f"Processing feature = {FEATURE}")
    analyzer = InterUnivariateSienaAnalyzer(FEATURE)
    analyzer.univariate_zone_bar_chart("frontal")
    analyzer.univariate_zone_violin_chart("frontal")
    analyzer.univariate_zone_bar_chart("temporal")
    analyzer.univariate_zone_violin_chart("temporal")
    time.sleep(3)
    analyzer.univariate_zone_bar_chart("parietal")
    analyzer.univariate_zone_violin_chart("parietal")
    time.sleep(3)
    analyzer.univariate_zone_bar_chart("occipital")
    analyzer.univariate_zone_violin_chart("occipital")
    time.sleep(3)

    logging.info("Processing Tusz vs TUEP")

    logging.info(f"Processing feature = {FEATURE}")
    analyzer = InterUnivariateTuszAnalyzer(FEATURE)
    analyzer.univariate_zone_bar_chart("frontal")
    analyzer.univariate_zone_violin_chart("frontal")
    time.sleep(3)
    analyzer.univariate_zone_bar_chart("temporal")
    analyzer.univariate_zone_violin_chart("temporal")
    time.sleep(3)
    analyzer.univariate_zone_bar_chart("parietal")
    analyzer.univariate_zone_violin_chart("parietal")
    time.sleep(3)
    analyzer.univariate_zone_bar_chart("occipital")
    analyzer.univariate_zone_violin_chart("occipital")
    time.sleep(3)


    logging.info("Processing database CHB-MIT")

    logging.info(f"Processing feature = {FEATURE}")
    analyzer = IntraUnivariateChbAnalyzer(FEATURE)

    analyzer.hemisfere_bar_chart("left", "unknown")
    analyzer.hemisfere_dist_chart("left", "unknown")
    time.sleep(3)
    analyzer.hemisfere_bar_chart("right", "unknown")
    analyzer.hemisfere_dist_chart("right", "unknown")
    time.sleep(3)
    for band in ["delta", "theta", "alpha", "beta", "all"]:
        analyzer.univariate_topo_plot_average("unknown", band)
        time.sleep(3)
    ml_analyzer = MlUnivariateFeatureAnalyzer("chb", FEATURE)
    ml_analyzer.ml_bar_chart("unknown")
    time.sleep(3)


    logging.info("Processing database Siena")

    logging.info(f"Processing feature = {FEATURE}")
    analyzer = IntraUnivariateSienaAnalyzer(FEATURE)
    for seizure_type in settings["siena"]["valid_seizure_types"]:
        analyzer.region_bar_chart("frontal", seizure_type)
        analyzer.region_dist_chart("frontal", seizure_type)
        time.sleep(3)
        analyzer.region_bar_chart("temporal", seizure_type)
        analyzer.region_dist_chart("temporal", seizure_type)
        time.sleep(3)
        analyzer.region_bar_chart("parietal", seizure_type)
        analyzer.region_dist_chart("parietal", seizure_type)
        time.sleep(3)
        analyzer.region_bar_chart("occipital", seizure_type)
        analyzer.region_dist_chart("occipital", seizure_type)
        time.sleep(3)
        for band in ["delta", "theta", "alpha", "beta", "all"]:
            analyzer.univariate_topo_plot_average(seizure_type, band)
            time.sleep(3)
        ml_analyzer = MlUnivariateFeatureAnalyzer("siena", FEATURE)
        ml_analyzer.ml_bar_chart(seizure_type)
        time.sleep(3)

    logging.info("Processing database TUSZ")

    logging.info(f"Processing feature = {FEATURE}")
    analyzer = IntraUnivariateTuszAnalyzer(FEATURE)
    for seizure_type in settings["tusz"]["valid_seizure_types"]:
        analyzer.region_bar_chart("frontal", seizure_type)
        analyzer.region_dist_chart("frontal", seizure_type)
        time.sleep(3)
        analyzer.region_bar_chart("temporal", seizure_type)
        analyzer.region_dist_chart("temporal", seizure_type)
        time.sleep(3)
        analyzer.region_bar_chart("parietal", seizure_type)
        analyzer.region_dist_chart("parietal", seizure_type)
        time.sleep(3)
        analyzer.region_bar_chart("occipital", seizure_type)
        analyzer.region_dist_chart("occipital", seizure_type)
        time.sleep(3)
        for band in ["delta", "theta", "alpha", "beta", "all"]:
            analyzer.univariate_topo_plot_average(seizure_type, band)
            time.sleep(3)
        ml_analyzer = MlUnivariateFeatureAnalyzer("tusz", FEATURE)
        ml_analyzer.ml_bar_chart(seizure_type)
        time.sleep(3)


if __name__ == "__main__":
    main()