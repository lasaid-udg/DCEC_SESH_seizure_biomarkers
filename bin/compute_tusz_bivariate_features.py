#!/var/tmp/venv-project-1/bin/python
"""
Usage:
    compute_tusz_bivariate_features.py (--feature=<feature>)
"""
import os
import sys
import pandas
import logging
import warnings
from docopt import docopt
warnings.filterwarnings("ignore")
sys.path.append("../")
from lib.slices import EegWindowsTusz
from lib.filters import BandEstimator
from lib.features import FeatureGateway


FEATURE = docopt(__doc__)["--feature"]
OUTPUT_DIRECTORY = os.getenv("BIOMARKERS_PROJECT_HOME")


def main():
    windows_tusz = EegWindowsTusz()
    feature_estimator = FeatureGateway()
    feature_list = []
    counter = 0

    for metadata, windows in iter(windows_tusz):
        logging.info(f"Processing patient = {metadata['patient']}")
        counter += 1
        for window_number, seizure_stage in metadata["windows"].items():
            eeg_array = windows[int(window_number), :, :]

            if FEATURE == "coherence":
                logging.info(f"Processing instance")
                delta, theta, alpha, beta, all = feature_estimator(FEATURE, eeg_array, metadata["sampling_frequency"])
                
                for band_name, band in zip(["delta", "theta", "alpha", "beta", "all"], [delta, theta, alpha, beta, all]):
                    logging.info(f"Processing band = {band_name}")
                    feature_counter = 0
                    for i in range(len(metadata["channels"])):
                        for j in range(i, len(metadata["channels"])):
                            if i == j:
                                continue
                            feature_list.append({"patient": metadata["patient"],
                                                    "band": band_name,
                                                    "seizure_type": metadata["seizure_type"],
                                                    "channels": f"{metadata['channels'][i]}_{metadata['channels'][j]}",
                                                    "index_channels": f"{i}_{j}",
                                                    "seizure_stage": seizure_stage[0],
                                                    "time_point": seizure_stage[1],
                                                    "feature": FEATURE,
                                                    "seizure_number": counter,
                                                    "value": band[feature_counter]})
                            feature_counter += 1
            else:
                delta, theta, alpha, beta = BandEstimator.get_eeg_bands(eeg_array,
                                                                        metadata["sampling_frequency"])
                
                for band_name, band in zip(["delta", "theta", "alpha", "beta", "all"], [delta, theta, alpha, beta, eeg_array]):
                    logging.info(f"Processing band = {band_name}")

                    feature_value = feature_estimator(FEATURE, band)

                    feature_counter = 0
                    for i in range(len(metadata["channels"])):
                        for j in range(i, len(metadata["channels"])):
                            if i == j:
                                continue
                            feature_list.append({"patient": metadata["patient"],
                                                "band": band_name,
                                                "seizure_type": metadata["seizure_type"],
                                                "channels": f"{metadata['channels'][i]}_{metadata['channels'][j]}",
                                                "index_channels": f"{i}_{j}",
                                                "seizure_stage": seizure_stage[0],
                                                "time_point": seizure_stage[1],
                                                "feature": FEATURE,
                                                "seizure_number": counter,
                                                "value": feature_value[feature_counter]})
                            feature_counter += 1

    feature_df = pandas.DataFrame(feature_list)
    output_file_eeg = os.path.join(OUTPUT_DIRECTORY, "features", "tusz", f"{FEATURE}.csv")
    feature_df.to_csv(output_file_eeg, index=False)


if __name__ == "__main__":
    main()