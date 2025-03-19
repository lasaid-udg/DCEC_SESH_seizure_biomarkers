#!/var/tmp/venv-project-1/bin/python
"""
Usage:
    analyze_stats_univariate_features.py (--feature=<feature>)
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
from lib.analyzers import IntraUnivariateChbAnalyzer, IntraUnivariateSienaAnalyzer, IntraUnivariateTuszAnalyzer, \
                          InterUnivariateSienaAnalyzer, InterUnivariateTuszAnalyzer


FEATURE = docopt(__doc__)["--feature"]
OUTPUT_DIRECTORY = os.getenv("BIOMARKERS_PROJECT_HOME")


def main():

    stats_gateway = StatisticalTests()

    logging.info("Processing database Siena vs TUEP")
    test_results = []

    logging.info(f"Processing feature = {FEATURE}")
    analyzer = InterUnivariateSienaAnalyzer(FEATURE)

    for region in ["frontal", "temporal", "parietal", "occipital"]:
        groups = analyzer.processed_data_for_kruskal_wallis(region)

        for group in groups:
            try:
                kruskal_p_value, _ = stats_gateway.run_kruskal_wallis(group[1])
            except ValueError as exc:
                logging.error(f"Error in Kruskal-Wallis test = {exc}")
                continue
            dunn_p_values = stats_gateway.run_posthoc_dunns(group[1])
            test_result = {"feature": FEATURE,
                           "region": region,
                           "band": group[0],
                           "kruskal_p_value": kruskal_p_value}
            test_result.update({key: value for (key, value) in dunn_p_values})
            test_results.append(test_result)

    output_file = f"kruskal_siena_inter_{FEATURE}.csv"
    output_file = os.path.join(OUTPUT_DIRECTORY, "reports", output_file)

    normalized_stationarity_evaluation = pandas.DataFrame(test_results)
    normalized_stationarity_evaluation
    normalized_stationarity_evaluation.to_csv(output_file, index=False)

    ########################################################################
    logging.info("Processing database Tusz vs TUEP")
    test_results = []

    logging.info(f"Processing feature = {FEATURE}")
    analyzer = InterUnivariateTuszAnalyzer(FEATURE)

    for region in ["frontal", "temporal", "parietal", "occipital"]:
        groups = analyzer.processed_data_for_kruskal_wallis(region)

        for group in groups:
            try:
                kruskal_p_value, _ = stats_gateway.run_kruskal_wallis(group[1])
            except ValueError as exc:
                logging.error(f"Error in Kruskal-Wallis test = {exc}")
                continue
            dunn_p_values = stats_gateway.run_posthoc_dunns(group[1])
            test_result = {"feature": FEATURE,
                           "region": region,
                           "band": group[0],
                           "kruskal_p_value": kruskal_p_value}
            test_result.update({key: value for (key, value) in dunn_p_values})
            test_results.append(test_result)

    output_file = f"kruskal_tusz_inter_{FEATURE}.csv"
    output_file = os.path.join(OUTPUT_DIRECTORY, "reports", output_file)

    normalized_stationarity_evaluation = pandas.DataFrame(test_results)
    normalized_stationarity_evaluation.to_csv(output_file, index=False)

    ########################################################################
    logging.info("Processing database CHB-MIT")
    test_results = []

    logging.info(f"Processing feature = {FEATURE}")
    analyzer = IntraUnivariateChbAnalyzer(FEATURE)

    for region in ["left", "right"]:
        groups = analyzer.processed_data_for_friedman(region, "unknown")

        for group in groups:
            try:
                friedman_p_value, _ = stats_gateway.run_friedman_test(group[1])
            except ValueError as exc:
                logging.error(f"Error in Friedmam test = {exc}")
                continue
            nemenyi_p_values = stats_gateway.run_posthoc_nemenyi(group[1])
            test_result = {"feature": FEATURE,
                            "seizure_type": "unknown",
                            "group_size": len(group[1][0]),
                            "region": region,
                            "band": group[0],
                            "friedman_p_value": friedman_p_value}
            test_result.update({key: value for (key, value) in nemenyi_p_values})
            test_results.append(test_result)

    output_file = f"friedman_chb_intra_{FEATURE}.csv"
    output_file = os.path.join(OUTPUT_DIRECTORY, "reports", output_file)

    normalized_stationarity_evaluation = pandas.DataFrame(test_results)
    normalized_stationarity_evaluation.to_csv(output_file, index=False)

    ########################################################################
    logging.info("Processing database Siena")
    test_results = []

    logging.info(f"Processing feature = {FEATURE}")
    analyzer = IntraUnivariateSienaAnalyzer(FEATURE)

    for seizure_type in settings["siena"]["valid_seizure_types"]:
        for region in ["frontal", "temporal", "parietal", "occipital"]:
            groups = analyzer.processed_data_for_friedman(region, seizure_type)

            for group in groups:
                try:
                    friedman_p_value, _ = stats_gateway.run_friedman_test(group[1])
                except ValueError as exc:
                    logging.error(f"Error in Friedmam test = {exc}")
                    continue
                nemenyi_p_values = stats_gateway.run_posthoc_nemenyi(group[1])
                test_result = {"feature": FEATURE,
                               "seizure_type": seizure_type,
                               "group_size": len(group[1][0]),
                               "region": region,
                               "band": group[0],
                               "friedman_p_value": friedman_p_value}
                test_result.update({key: value for (key, value) in nemenyi_p_values})
                test_results.append(test_result)

    output_file = f"friedman_siena_intra_{FEATURE}.csv"
    output_file = os.path.join(OUTPUT_DIRECTORY, "reports", output_file)

    normalized_stationarity_evaluation = pandas.DataFrame(test_results)
    normalized_stationarity_evaluation.to_csv(output_file, index=False)

    ########################################################################
    logging.info("Processing database Tusz")
    test_results = []

    logging.info(f"Processing feature = {FEATURE}")
    analyzer = IntraUnivariateTuszAnalyzer(FEATURE)

    for seizure_type in settings["tusz"]["valid_seizure_types"]:
        for region in ["frontal", "temporal", "parietal", "occipital"]:
            groups = analyzer.processed_data_for_friedman(region, seizure_type)

            for group in groups:
                try:
                    friedman_p_value, _ = stats_gateway.run_friedman_test(group[1])
                except ValueError as exc:
                    logging.error(f"Error in Friedmam test = {exc}")
                    continue
                nemenyi_p_values = stats_gateway.run_posthoc_nemenyi(group[1])
                test_result = {"feature": FEATURE,
                               "seizure_type": seizure_type,
                               "group_size": len(group[1][0]),
                               "region": region,
                               "band": group[0],
                               "friedman_p_value": friedman_p_value}
                test_result.update({key: value for (key, value) in nemenyi_p_values})
                test_results.append(test_result)

    output_file = f"friedman_tusz_intra_{FEATURE}.csv"
    output_file = os.path.join(OUTPUT_DIRECTORY, "reports", output_file)

    normalized_stationarity_evaluation = pandas.DataFrame(test_results)
    normalized_stationarity_evaluation.to_csv(output_file, index=False)


if __name__ == "__main__":
    main()