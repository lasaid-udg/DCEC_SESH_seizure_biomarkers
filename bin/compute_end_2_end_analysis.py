#!/var/tmp/venv-project-1/bin/python
"""
Usage:
    compute_end_2_end_analysis.py
"""
import os
import sys
import json
import uuid
import time
import numpy
import logging
import warnings
warnings.filterwarnings("ignore")
sys.path.append("../")
from lib.slices import EegWindowsTusz
from lib.filters import FilterBank
from lib.signals import EegProcessorTusz
from lib.features import FeatureGateway
from lib.metadata import ChannelBasedMetadataTusz
from lib.visuals import plot_topoplot_features_frame


OUTPUT_DIRECTORY = os.getenv("BIOMARKERS_PROJECT_HOME")


def main():
    windows_tusz = EegWindowsTusz()

    for metadata, _ in iter(windows_tusz):
        logging.info(f"Processing patient = {metadata['patient']}")
        channel_metadata = ChannelBasedMetadataTusz(metadata["source_file"])

        ##########################################################
        try:
            processor = EegProcessorTusz(metadata["source_file"])
        except FileNotFoundError as exc:
            logging.error(f"Error found {exc}")
            continue
        ##########################################################
        processor.scale()
        ##########################################################
        try:
            processor.select_channels()
        except KeyError as exc:
            logging.error(f"Not valid channel set = {exc}")
            continue
        ##########################################################
        try:
            processor.resample()
        except AssertionError as exc:
            logging.error(f"Not valid sampling frequency = {exc}")
            continue
        ##########################################################
        processor.filter_bank = FilterBank(0)
        processor.remove_drift()
        processor.remove_hfo()
        processor.remove_power_noise()

        ###########################################################
        eeg_slice = processor._data
        eeg_slice = EegProcessorTusz.rereference_to_average(numpy.abs(eeg_slice))
        eeg_slice = EegProcessorTusz.standardize(eeg_slice)

        ###########################################################
        recording_lenght = int(eeg_slice.shape[1] / metadata["sampling_frequency"])
        event_start = recording_lenght

        for second, channels in channel_metadata.seizure_ranges.items():
            if len(channels):
                event_start = second - 60
                break

        if (event_start + 120) < recording_lenght:
            event_end = event_start + 120
        else:
            event_end = recording_lenght

        ###########################################################
        feature_estimator = FeatureGateway()
        entropy_event_features = []
        theta_psd_event_features = []
        alpha_psd_event_features = []
        event_impacted_channels = []
        entropy_min_value = 1000000
        theta_psd_min_value = 1000000
        alpha_psd_min_value = 1000000
        entropy_max_value = 0
        theta_psd_max_value = 0
        alpha_psd_max_value = 0

        for second in range(event_start, event_end):
            index_start = second * metadata["sampling_frequency"]
            index_end = (second + 1) * metadata["sampling_frequency"]
            window = eeg_slice[:, index_start: index_end]

            entropy_features = []
            theta_psd_features = []
            alpha_psd_features = []
            event_impacted_channels.append(channel_metadata.seizure_ranges.get(second, []))

            for channel_number, channel_name in enumerate(metadata["channels"]):
                entropy = feature_estimator("approximate_entropy", window[channel_number, :])
                feature_value = feature_estimator("power_spectral_density", window[channel_number, :],
                                                  metadata["sampling_frequency"])
                theta_psd = feature_value[1]
                alpha_psd = feature_value[2]

                entropy_features.append((channel_name, entropy))
                theta_psd_features.append((channel_name, theta_psd))
                alpha_psd_features.append((channel_name, alpha_psd))

                if entropy < entropy_min_value:
                    entropy_min_value = entropy
                if entropy > entropy_max_value:
                    entropy_max_value = entropy

                if theta_psd < theta_psd_min_value:
                    theta_psd_min_value = theta_psd
                if theta_psd > theta_psd_max_value:
                    theta_psd_max_value = theta_psd

                if alpha_psd < alpha_psd_min_value:
                    alpha_psd_min_value = alpha_psd
                if alpha_psd > alpha_psd_max_value:
                    alpha_psd_max_value = alpha_psd

            entropy_event_features.append(entropy_features)
            theta_psd_event_features.append(theta_psd_features)
            alpha_psd_event_features.append(alpha_psd_features)

        counter = 0
        unique_id = str(uuid.uuid4().int)[-8:]
        base_path = os.getenv("BIOMARKERS_PROJECT_HOME")
        base_path = os.path.join(base_path, "videos", unique_id)
        os.makedirs(base_path, exist_ok=True)

        for single_channels, entropy, theta, alpha in zip(event_impacted_channels,
                                                          entropy_event_features,
                                                          theta_psd_event_features,
                                                          alpha_psd_event_features):
            logging.info(f"Processing frame = {counter}")
            output_file = os.path.join(base_path, f"image_{counter}.png")
            plot_topoplot_features_frame(single_channels, entropy, theta, alpha,
                                         entropy_min_value, entropy_max_value,
                                         theta_psd_min_value, theta_psd_max_value,
                                         alpha_psd_min_value, alpha_psd_max_value,
                                         output_file)
            counter += 1
            time.sleep(1)

        output_file = os.path.join(base_path, "metadata.json")
        with open(output_file, "w") as fp:
            json.dump(metadata, fp, indent=4)


if __name__ == "__main__":
    main()
