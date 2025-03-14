#!/var/tmp/venv-project-1/bin/python
import os
import sys
import pandas
import logging
import warnings
warnings.filterwarnings("ignore")
sys.path.append("../")
from lib import settings
from lib.stats import StatisticalTests
from lib.analyzers import IntraUnivariateChbAnalyzer, IntraUnivariateSienaAnalyzer, \
                          IntraUnivariateTuszAnalyzer, IntraUnivariateTuepAnalyzer


OUTPUT_DIRECTORY = os.getenv("BIOMARKERS_PROJECT_HOME")


def main():

    stats_gateway = StatisticalTests()

    logging.info("Processing database TUEP")
    test_results = []

    for feature in settings["univariate_features"]:
        logging.info(f"Processing feature = {feature}")
        analyzer = IntraUnivariateTuepAnalyzer(feature)
        for region in ["frontal", "temporal", "parietal", "occipital"]:
            groups = analyzer.processed_data_for_lilliefors(region)
            for group in groups:
                p_value, interpretation = stats_gateway.run_lilliefors_test(group[1])
                test_results.append({"feature": feature,
                                     "region": region,
                                     "band": group[0],
                                     "seizure_type": "NA",
                                     "time_point": "NA",
                                     "p_value": p_value, 
                                     "interpretation": interpretation})
    output_file = f"normality_TUEP_{feature}.csv"
    output_file = os.path.join(OUTPUT_DIRECTORY, "reports", output_file)

    normalized_stationarity_evaluation = pandas.DataFrame(test_results)
    normalized_stationarity_evaluation.to_csv(output_file, index=False)



    logging.info("Processing database CHB-MIT")
    test_results = []

    for feature in settings["univariate_features"]:
        logging.info(f"Processing feature = {feature}")
        analyzer = IntraUnivariateChbAnalyzer(feature)
        for region in ["left", "right"]:
            groups = analyzer.processed_data_for_lilliefors(region, "unknown")
            for group in groups:
                p_value, interpretation = stats_gateway.run_lilliefors_test(group[2])
                test_results.append({"feature": feature,
                                     "region": region,
                                     "band": group[0],
                                     "time_point": group[1],
                                     "p_value": p_value, 
                                     "interpretation": interpretation})
    
    output_file = f"normality_CHB_{feature}.csv"
    output_file = os.path.join(OUTPUT_DIRECTORY, "reports", output_file)

    normalized_stationarity_evaluation = pandas.DataFrame(test_results)
    normalized_stationarity_evaluation.to_csv(output_file, index=False)


    logging.info("Processing database Siena")
    test_results = []

    for feature in settings["univariate_features"]:
        logging.info(f"Processing feature = {feature}")
        analyzer = IntraUnivariateSienaAnalyzer(feature)
        for region in ["frontal", "temporal", "parietal", "occipital"]:
            for seizure_type in settings["siena"]["valid_seizure_types"]:
                groups = analyzer.processed_data_for_lilliefors(region, seizure_type)
                for group in groups:
                    p_value, interpretation = stats_gateway.run_lilliefors_test(group[2])
                    test_results.append({"feature": feature,
                                         "region": region,
                                         "band": group[0],
                                         "seizure_type": seizure_type,
                                         "time_point": group[1],
                                         "p_value": p_value, 
                                         "interpretation": interpretation})
    
    output_file = f"normality_Siena_{feature}.csv"
    output_file = os.path.join(OUTPUT_DIRECTORY, "reports", output_file)

    normalized_stationarity_evaluation = pandas.DataFrame(test_results)
    normalized_stationarity_evaluation.to_csv(output_file, index=False)


    logging.info("Processing database TUSZ")
    test_results = []

    for feature in settings["univariate_features"]:
        logging.info(f"Processing feature = {feature}")
        analyzer = IntraUnivariateTuszAnalyzer(feature)
        for region in ["frontal", "temporal", "parietal", "occipital"]:
            for seizure_type in settings["tusz"]["valid_seizure_types"]:
                groups = analyzer.processed_data_for_lilliefors(region, "unknown")
                for group in groups:
                    p_value, interpretation = stats_gateway.run_lilliefors_test(group[2])
                    test_results.append({"feature": feature,
                                         "region": region,
                                         "band": group[0],
                                         "seizure_type": seizure_type,
                                         "time_point": group[1],
                                         "p_value": p_value, 
                                         "interpretation": interpretation})
    
    output_file = f"normality_TUSZ_{feature}.csv"
    output_file = os.path.join(OUTPUT_DIRECTORY, "reports", output_file)

    normalized_stationarity_evaluation = pandas.DataFrame(test_results)
    normalized_stationarity_evaluation.to_csv(output_file, index=False)

if __name__ == "__main__":
    main()