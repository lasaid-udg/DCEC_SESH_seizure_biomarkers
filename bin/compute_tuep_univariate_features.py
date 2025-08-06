#!/var/tmp/venv-project-1/bin/python
"""
Usage:
    compute_tusz_univariate_features.py (--feature=<feature>)
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
from lib.slices import EegWindowsTuep
from lib.filters import BandEstimator
from lib.features import FeatureGateway


FEATURE = docopt(__doc__)["--feature"]
OUTPUT_DIRECTORY = os.getenv("BIOMARKERS_PROJECT_HOME")


def main():
    windows_tuep = EegWindowsTuep()
    feature_estimator = FeatureGateway()
    feature_list = []
    counter = 0

    channels_lookup = {}
    for hemisphere, channels in settings["siena"]["univariate_channels_groups"].items():
        for channel in channels:
            channels_lookup[channel] = hemisphere

    for metadata, window in iter(windows_tuep):
        logging.info(f"Processing patient = {metadata['patient']}")
        counter += 1
        eeg_array = window[:, :]

        if FEATURE != "power_spectral_density":
            delta, theta, alpha, beta, gamma = BandEstimator.get_eeg_bands(eeg_array,
                                                                    metadata["sampling_frequency"])

            for band_name, band in zip(["delta", "theta", "alpha", "beta", "gamma", "all"],
                                       [delta, theta, alpha, beta, gamma, eeg_array]):
                logging.info(f"Processing band = {band_name}")

                for channel_number, channel_name in enumerate(metadata["channels"]):
                    hemisphere = channels_lookup[channel_name]
                    feature_value = feature_estimator(FEATURE, band[channel_number, :])
                    feature_list.append({"patient": metadata["patient"],
                                         "band": band_name,
                                         "hemisphere": hemisphere,
                                         "channel": channel_name,
                                         "feature": FEATURE,
                                         "value": feature_value,
                                         "id": metadata["file_number"]})

        else:
            for channel_number, channel_name in enumerate(metadata["channels"]):
                logging.info("Processing instance")
                densities = feature_estimator(FEATURE, eeg_array[channel_number, :],
                                              metadata["sampling_frequency"])
                for band_name, density in zip(["delta", "theta", "alpha", "beta", "gamma", "all"], densities):
                    feature_list.append({"patient": metadata["patient"],
                                         "band": band_name,
                                         "hemisphere": hemisphere,
                                         "channel": channel_name,
                                         "feature": FEATURE,
                                         "value": density,
                                         "id": metadata["file_number"]})

    feature_df = pandas.DataFrame(feature_list)
    output_file_eeg = os.path.join(OUTPUT_DIRECTORY, "features", "tuep", f"{FEATURE}.csv")
    feature_df.to_csv(output_file_eeg, index=False)


if __name__ == "__main__":
    main()
