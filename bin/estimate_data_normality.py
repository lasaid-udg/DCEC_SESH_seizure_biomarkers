#!/var/tmp/venv-project-1/bin/python
"""
Usage:
    estimate_data_normality.py (--feature=<feature>)
"""
import os
import sys
import pandas
import logging
import warnings
from docopt import docopt
warnings.filterwarnings("ignore")
sys.path.append("../")
from lib import settings
from lib.stats import StatisticalTests
from lib.analyzers import IntraUnivariateChbAnalyzer, IntraUnivariateSienaAnalyzer, \
                          IntraUnivariateTuszAnalyzer, IntraUnivariateTuepAnalyzer


FEATURE = docopt(__doc__)["--feature"]
OUTPUT_DIRECTORY = os.getenv("BIOMARKERS_PROJECT_HOME")


def main():

    stats_gateway = StatisticalTests()

    logging.info("Processing database TUEP")

    logging.info(f"Processing feature = {FEATURE}")
    analyzer = IntraUnivariateTuepAnalyzer(FEATURE)
    test_results = []

    for region in ["frontal", "temporal", "parietal", "occipital"]:
        groups = analyzer.processed_data_for_lilliefors(region)
        for group in groups:
            p_value, interpretation = stats_gateway.run_lilliefors_test(group[1])
            test_results.append({"feature": FEATURE,
                                 "region": region,
                                 "band": group[0],
                                 "group_size": group[1].shape[0],
                                 "seizure_type": "NA",
                                 "time_point": "NA",
                                 "p_value": p_value,
                                 "interpretation": interpretation})
    output_file = f"normality_TUEP_{FEATURE}.csv"
    output_file = os.path.join(OUTPUT_DIRECTORY, "reports", output_file)

    normalized_stationarity_evaluation = pandas.DataFrame(test_results)
    normalized_stationarity_evaluation.to_csv(output_file, index=False)

    logging.info("Processing database CHB-MIT")

    logging.info(f"Processing feature = {FEATURE}")
    analyzer = IntraUnivariateChbAnalyzer(FEATURE)
    test_results = []

    for region in ["left", "right"]:
        groups = analyzer.processed_data_for_lilliefors(region, "unknown")
        for group in groups:
            p_value, interpretation = stats_gateway.run_lilliefors_test(group[2])
            test_results.append({"feature": FEATURE,
                                 "region": region,
                                 "band": group[0],
                                 "group_size": group[2].shape[0],
                                 "time_point": group[1],
                                 "p_value": p_value,
                                 "interpretation": interpretation})

    output_file = f"normality_CHB_{FEATURE}.csv"
    output_file = os.path.join(OUTPUT_DIRECTORY, "reports", output_file)

    normalized_stationarity_evaluation = pandas.DataFrame(test_results)
    normalized_stationarity_evaluation.to_csv(output_file, index=False)

    logging.info("Processing database Siena")

    logging.info(f"Processing feature = {FEATURE}")
    analyzer = IntraUnivariateSienaAnalyzer(FEATURE)
    test_results = []

    for region in ["frontal", "temporal", "parietal", "occipital"]:
        for seizure_type in settings["siena"]["valid_seizure_types"]:
            groups = analyzer.processed_data_for_lilliefors(region, seizure_type)
            for group in groups:
                p_value, interpretation = stats_gateway.run_lilliefors_test(group[2])
                test_results.append({"feature": FEATURE,
                                     "region": region,
                                     "band": group[0],
                                     "group_size": group[2].shape[0],
                                     "seizure_type": seizure_type,
                                     "time_point": group[1],
                                     "p_value": p_value,
                                     "interpretation": interpretation})

    output_file = f"normality_Siena_{FEATURE}.csv"
    output_file = os.path.join(OUTPUT_DIRECTORY, "reports", output_file)

    normalized_stationarity_evaluation = pandas.DataFrame(test_results)
    normalized_stationarity_evaluation.to_csv(output_file, index=False)

    logging.info("Processing database TUSZ")

    logging.info(f"Processing feature = {FEATURE}")
    analyzer = IntraUnivariateTuszAnalyzer(FEATURE)
    test_results = []

    for region in ["frontal", "temporal", "parietal", "occipital"]:
        for seizure_type in settings["tusz"]["valid_seizure_types"]:
            groups = analyzer.processed_data_for_lilliefors(region, "unknown")
            for group in groups:
                p_value, interpretation = stats_gateway.run_lilliefors_test(group[2])
                test_results.append({"feature": FEATURE,
                                     "region": region,
                                     "band": group[0],
                                     "group_size": group[2].shape[0],
                                     "seizure_type": seizure_type,
                                     "time_point": group[1],
                                     "p_value": p_value,
                                     "interpretation": interpretation})

    output_file = f"normality_TUSZ_{FEATURE}.csv"
    output_file = os.path.join(OUTPUT_DIRECTORY, "reports", output_file)

    normalized_stationarity_evaluation = pandas.DataFrame(test_results)
    normalized_stationarity_evaluation.to_csv(output_file, index=False)


if __name__ == "__main__":
    main()
