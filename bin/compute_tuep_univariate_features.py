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
from lib.slices import EegWindowsTuep
from lib.filters import BandEstimator
from lib.features import UnivariateFeatureGateway


FEATURE = docopt(__doc__)["--feature"]
OUTPUT_DIRECTORY = os.getenv("BIOMARKERS_PROJECT_HOME")


def main():
    windows_tusz = EegWindowsTuep()
    feature_estimator = UnivariateFeatureGateway()
    feature_list = []
    counter = 0

    for metadata, window in iter(windows_tusz):
        logging.info(f"Processing patient = {metadata['patient']}")
        counter += 1
        eeg_array = window[:, :]
        delta, theta, alpha, beta = BandEstimator.get_eeg_bands(eeg_array,
                                                                metadata["sampling_frequency"])

        for band_name, band in zip(["delta", "theta", "alpha", "beta", "all"], [delta, theta, alpha, beta, eeg_array]):
            logging.info(f"Processing band = {band_name}")

            for channel_number, channel_name in enumerate(metadata["channels"]): 
                feature_value = feature_estimator(FEATURE, band[channel_number, :])
                feature_list.append({"patient": metadata["patient"],
                                     "band": band_name,
                                     "channel": channel_name,
                                     "feature": FEATURE,
                                     "value": feature_value})

    feature_df = pandas.DataFrame(feature_list)
    output_file_eeg = os.path.join(OUTPUT_DIRECTORY, "features", "tuep", f"{FEATURE}.csv")
    feature_df.to_csv(output_file_eeg, index=False)


if __name__ == "__main__":
    main()