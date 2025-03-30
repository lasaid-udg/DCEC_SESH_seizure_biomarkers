#!/var/tmp/venv-project-1/bin/python
"""
Usage:
    compute_graph_features.py (--feature=<feature>) (--threshold=<threshold>)
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
from lib.features import FeatureGateway
from lib.analyzers import IntraBivariateChbAnalyzer, IntraBivariateSienaAnalyzer, IntraBivariateTuszAnalyzer


FEATURE = docopt(__doc__)["--feature"]
THRESHOLD = docopt(__doc__)["--threshold"]
OUTPUT_DIRECTORY = os.getenv("BIOMARKERS_PROJECT_HOME")


def main():
    feature_estimator = FeatureGateway()

    logging.info(f"Processing feature = {FEATURE}, threshold = {THRESHOLD}")
    feature_list = []
    analyzer = IntraBivariateChbAnalyzer(FEATURE)

    logging.info("Processing seizure_type = unknown")
    for (seizure_number, band_name, stage, window, network) in analyzer.processed_data_for_efficiency("unknown", threshold=float(THRESHOLD)):
        value = feature_estimator.global_efficiency(network)
        feature_list.append({"band": band_name,
                             "seizure_type": "unknown",
                             "seizure_stage": stage,
                             "time_point": window,
                             "feature": "global_efficiency",
                             "seizure_number": seizure_number,
                             "value": value})

    feature_df = pandas.DataFrame(feature_list)
    _threshold = THRESHOLD.replace(".", "_")
    output_file_eeg = os.path.join(OUTPUT_DIRECTORY, "features", "chb-mit",
                                   f"global_efficiency?{FEATURE}_{_threshold}.csv")
    feature_df.to_csv(output_file_eeg, index=False)

    logging.info(f"Processing feature = {FEATURE}, threshold = {THRESHOLD}")
    feature_list = []
    analyzer = IntraBivariateSienaAnalyzer(FEATURE)

    for seizure_type in settings["siena"]["valid_seizure_types"]:
        logging.info(f"Processing seizure_type = {seizure_type}")
        for (seizure_number, band_name, stage, window, network) in analyzer.processed_data_for_efficiency(seizure_type, threshold=float(THRESHOLD)):
            value = feature_estimator.global_efficiency(network)
            feature_list.append({"band": band_name,
                                 "seizure_type": seizure_type,
                                 "seizure_stage": stage,
                                 "time_point": window,
                                 "feature": "global_efficiency",
                                 "seizure_number": seizure_number,
                                 "value": value})

    feature_df = pandas.DataFrame(feature_list)
    _threshold = THRESHOLD.replace(".", "_")
    output_file_eeg = os.path.join(OUTPUT_DIRECTORY, "features", "siena",
                                   f"global_efficiency_{FEATURE}_{_threshold}.csv")
    feature_df.to_csv(output_file_eeg, index=False)

    logging.info(f"Processing feature = {FEATURE}, threshold = {THRESHOLD}")
    feature_list = []
    analyzer = IntraBivariateTuszAnalyzer(FEATURE)

    for seizure_type in settings["tusz"]["valid_seizure_types"]:
        logging.info(f"Processing seizure_type = {seizure_type}")
        for (seizure_number, band_name, stage, window, network) in analyzer.processed_data_for_efficiency(seizure_type, threshold=0.5):
            value = feature_estimator.global_efficiency(network)
            feature_list.append({"band": band_name,
                                 "seizure_type": seizure_type,
                                 "seizure_stage": stage,
                                 "time_point": window,
                                 "feature": "global_efficiency",
                                 "seizure_number": seizure_number,
                                 "value": value})

    feature_df = pandas.DataFrame(feature_list)
    _threshold = THRESHOLD.replace(".", "_")
    output_file_eeg = os.path.join(OUTPUT_DIRECTORY, "features", "tusz",
                                   f"global_efficiency_{FEATURE}_{_threshold}.csv")
    feature_df.to_csv(output_file_eeg, index=False)


if __name__ == "__main__":
    main()
