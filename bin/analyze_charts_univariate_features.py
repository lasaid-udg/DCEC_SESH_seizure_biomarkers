#!/var/tmp/venv-project-1/bin/python
import os
import sys
import time
import logging
import warnings
warnings.filterwarnings("ignore")
sys.path.append("../")
from lib import settings
from lib.analyzers import IntraUnivariateChbAnalyzer, IntraUnivariateSienaAnalyzer, IntraUnivariateTuszAnalyzer


OUTPUT_DIRECTORY = os.getenv("BIOMARKERS_PROJECT_HOME")


def main():

    logging.info("Processing database CHB-MIT")

    for feature in settings["univariate_features"]:
        logging.info(f"Processing feature = {feature}")
        analyzer = IntraUnivariateChbAnalyzer(feature)
        analyzer.hemisfere_bar_chart("left", "unknown")
        analyzer.hemisfere_dist_chart("left", "unknown")
        time.sleep(3)
        analyzer.hemisfere_bar_chart("right", "unknown")
        analyzer.hemisfere_dist_chart("right", "unknown")
        time.sleep(3)

    logging.info("Processing database Siena")

    for feature in settings["univariate_features"]:
        logging.info(f"Processing feature = {feature}")
        analyzer = IntraUnivariateSienaAnalyzer(feature)
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

    logging.info("Processing database TUSZ")

    for feature in settings["univariate_features"]:
        logging.info(f"Processing feature = {feature}")
        analyzer = IntraUnivariateTuszAnalyzer(feature)
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


if __name__ == "__main__":
    main()