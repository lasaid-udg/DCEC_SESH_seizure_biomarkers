#!/var/tmp/venv-project-1/bin/python
"""
Usage:
    analyze_ml_univariate_features.py (--feature=<feature>)
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
from lib.learners import MlTools
from lib.analyzers import IntraUnivariateChbAnalyzer, IntraUnivariateSienaAnalyzer, \
                          IntraUnivariateTuszAnalyzer


FEATURE = docopt(__doc__)["--feature"]
OUTPUT_DIRECTORY = os.getenv("BIOMARKERS_PROJECT_HOME")


def main():

    ########################################################################
    logging.info("Processing database CHB-MIT")
    test_results = []

    logging.info(f"Processing feature = {FEATURE}")
    analyzer = IntraUnivariateChbAnalyzer(FEATURE)
    groups = analyzer.processed_data_for_naive_bayes("unknown")

    for band in groups:
        logging.info(f"Processing band = {band[0]}")
        for idx, group_1 in enumerate(band[1][:5]):
            for group_2 in band[1][idx + 1: 5]:
                logging.info(f"Running experiment = {group_1[0]} vs {group_2[0]}")
                x_train, x_test, y_train, y_test = MlTools.get_train_test_data(group_1[1], group_2[1])
                accuracy = MlTools.train_and_score_naive_bayes(x_train, x_test, y_train, y_test)
                logging.info(f"Accuracy = {accuracy}")
                test_result = {"feature": FEATURE,
                               "seizure_type": "unknown",
                               "band": band[0],
                               "instances": group_1[1].shape[1] + group_1[1].shape[1],
                               "experiment": f"{group_1[0]}, {group_2[0]}",
                               "accuracy": accuracy}
                test_results.append(test_result)

    output_file = f"naive_chb_intra_{FEATURE}.csv"
    output_file = os.path.join(OUTPUT_DIRECTORY, "reports", output_file)

    test_results = pandas.DataFrame(test_results)
    test_results.to_csv(output_file, index=False)

    ########################################################################
    logging.info("Processing database Siena")
    test_results = []

    logging.info(f"Processing feature = {FEATURE}")
    analyzer = IntraUnivariateSienaAnalyzer(FEATURE)

    for seizure_type in settings["siena"]["valid_seizure_types"]:
        groups = analyzer.processed_data_for_naive_bayes(seizure_type)

        for band in groups:
            logging.info(f"Processing band = {band[0]}")
            for idx, group_1 in enumerate(band[1][:5]):
                for group_2 in band[1][idx + 1: 5]:
                    logging.info(f"Running experiment = {group_1[0]} vs {group_2[0]}")
                    x_train, x_test, y_train, y_test = MlTools.get_train_test_data(group_1[1], group_2[1])
                    accuracy = MlTools.train_and_score_naive_bayes(x_train, x_test, y_train, y_test)
                    logging.info(f"Accuracy = {accuracy}")
                    test_result = {"feature": FEATURE,
                                   "seizure_type": seizure_type,
                                   "band": band[0],
                                   "instances": group_1[1].shape[1] + group_1[1].shape[1],
                                   "experiment": f"{group_1[0]}, {group_2[0]}",
                                   "accuracy": accuracy}
                    test_results.append(test_result)

    output_file = f"naive_siena_intra_{FEATURE}.csv"
    output_file = os.path.join(OUTPUT_DIRECTORY, "reports", output_file)

    test_results = pandas.DataFrame(test_results)
    test_results.to_csv(output_file, index=False)

    ########################################################################
    logging.info("Processing database Tusz")
    test_results = []

    logging.info(f"Processing feature = {FEATURE}")
    analyzer = IntraUnivariateTuszAnalyzer(FEATURE)

    for seizure_type in settings["tusz"]["valid_seizure_types"]:
        groups = analyzer.processed_data_for_naive_bayes(seizure_type)

        for band in groups:
            logging.info(f"Processing band = {band[0]}")
            for idx, group_1 in enumerate(band[1][:5]):
                for group_2 in band[1][idx + 1: 5]:
                    logging.info(f"Running experiment = {group_1[0]} vs {group_2[0]}")
                    x_train, x_test, y_train, y_test = MlTools.get_train_test_data(group_1[1], group_2[1])
                    accuracy = MlTools.train_and_score_naive_bayes(x_train, x_test, y_train, y_test)
                    logging.info(f"Accuracy = {accuracy}")
                    test_result = {"feature": FEATURE,
                                   "seizure_type": seizure_type,
                                   "band": band[0],
                                   "instances": group_1[1].shape[1] + group_1[1].shape[1],
                                   "experiment": f"{group_1[0]}, {group_2[0]}",
                                   "accuracy": accuracy}
                    test_results.append(test_result)

    output_file = f"naive_tusz_intra_{FEATURE}.csv"
    output_file = os.path.join(OUTPUT_DIRECTORY, "reports", output_file)

    test_results = pandas.DataFrame(test_results)
    test_results.to_csv(output_file, index=False)


if __name__ == "__main__":
    main()
