#!/var/tmp/venv-project-1/bin/python
"""
Usage:
    compute_chb_univariate_features.py (--feature=<feature>)
"""
import os
import sys
import pandas
import logging
import warnings
from docopt import docopt
warnings.filterwarnings("ignore")
sys.path.append("../")
from lib.slices import EegWindowsChb
from lib.filters import BandEstimator
from lib.features import FeatureGateway


FEATURE = docopt(__doc__)["--feature"]
OUTPUT_DIRECTORY = os.getenv("BIOMARKERS_PROJECT_HOME")


def main():
    windows_chb = EegWindowsChb()
    feature_estimator = FeatureGateway()
    feature_list = []
    counter = 0

    for metadata, windows in iter(windows_chb):
        logging.info(f"Processing patient = {metadata['patient']}")
        counter += 1
        for window_number, seizure_stage in metadata["windows"].items():
            eeg_array = windows[int(window_number), :, :]

            if FEATURE != "power_spectral_density":
                delta, theta, alpha, beta = BandEstimator.get_eeg_bands(eeg_array,
                                                                        metadata["sampling_frequency"])

                for band_name, band in zip(["delta", "theta", "alpha", "beta", "all"],
                                           [delta, theta, alpha, beta, eeg_array]):
                    logging.info(f"Processing band = {band_name}")

                    for channel_number, channel_name in enumerate(metadata["channels"]):
                        feature_value = feature_estimator(FEATURE, band[channel_number, :])
                        feature_list.append({"patient": metadata["patient"],
                                             "band": band_name,
                                             "seizure_type": metadata["seizure_type"],
                                             "channel": channel_name,
                                             "seizure_stage": seizure_stage[0],
                                             "time_point": seizure_stage[1],
                                             "feature": FEATURE,
                                             "seizure_number": counter,
                                             "value": feature_value})
            else:
                for channel_number, channel_name in enumerate(metadata["channels"]):
                    logging.info("Processing instance")
                    densities = feature_estimator(FEATURE, eeg_array[channel_number, :],
                                                  metadata["sampling_frequency"])
                    for band_name, density in zip(["delta", "theta", "alpha", "beta", "all"], densities):
                        feature_list.append({"patient": metadata["patient"],
                                             "band": band_name,
                                             "seizure_type": metadata["seizure_type"],
                                             "channel": channel_name,
                                             "seizure_stage": seizure_stage[0],
                                             "time_point": seizure_stage[1],
                                             "feature": FEATURE,
                                             "seizure_number": counter,
                                             "value": density})

    feature_df = pandas.DataFrame(feature_list)
    output_file_eeg = os.path.join(OUTPUT_DIRECTORY, "features", "chb-mit", f"{FEATURE}.csv")
    feature_df.to_csv(output_file_eeg, index=False)


if __name__ == "__main__":
    main()
