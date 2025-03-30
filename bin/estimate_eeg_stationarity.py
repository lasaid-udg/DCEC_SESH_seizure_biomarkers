#!/var/tmp/venv-project-1/bin/python
import os
import sys
import pandas
import logging
import warnings
warnings.filterwarnings("ignore")
sys.path.append("../")
from lib.slices import WindowSelector, EegSlicesChb, EegSlicesSiena, EegSlicesTusz
from lib.signals import EegProcessorBaseClass
from lib.stats import StatisticalTests


OUTPUT_DIRECTORY = os.getenv("BIOMARKERS_PROJECT_HOME")
WINDOW_PROPOSALS = [0.5, 1, 3, 5]


def main():

    stats_gateway = StatisticalTests()
    database_slicers = [EegSlicesChb, EegSlicesSiena, EegSlicesTusz]
    database_names = ["chb", "siena", "tusz"]

    for database_slicer, database_name in zip(database_slicers, database_names):
        logging.info(f"Processing database = {database_name}")

        for window_length in WINDOW_PROPOSALS:
            stationarity_evaluation = dict()
            slicer = database_slicer()
            global_metadata = slicer.metadata

            for patient, patient_metadata in global_metadata.items():
                logging.info(f"Processing patient = {patient}")
                for seizure_number in range(len(patient_metadata)):
                    logging.info("Processing eeg slice")
                    seizure_metadata, eeg_slice = slicer.get(patient, seizure_number)
                    eeg_slice = EegProcessorBaseClass.standardize(eeg_slice)

                    selector = WindowSelector(seizure_metadata["sampling_frequency"])
                    selector.window_lenght = window_length * seizure_metadata["sampling_frequency"]
                    try:
                        windows_metadata, eeg_windows = selector.get_windows(seizure_metadata, eeg_slice)
                    except ValueError as exc:
                        logging.error(f"Window computation failed, error = {exc}")
                        continue

                    for window in range(eeg_windows.shape[0]):
                        for idx, channel in enumerate(windows_metadata["channels"]):
                            stationarity_result = stats_gateway.check_stationarity(eeg_windows[window, idx, :])
                            if channel not in stationarity_evaluation:
                                stationarity_evaluation[channel] = dict()
                            if stationarity_result not in stationarity_evaluation[channel]:
                                stationarity_evaluation[channel][stationarity_result] = 0
                            stationarity_evaluation[channel][stationarity_result] += 1

            normalized_stationarity_evaluation = list()
            for channel, item in stationarity_evaluation.items():
                for result, counter in item.items():
                    normalized_stationarity_evaluation.append({"channel": channel,
                                                               "result": result,
                                                               "count": counter})

            output_file = f"stationarity_{database_name}_{window_length}s.csv"
            output_file = os.path.join(OUTPUT_DIRECTORY, "reports", output_file)

            normalized_stationarity_evaluation = pandas.DataFrame(normalized_stationarity_evaluation)
            normalized_stationarity_evaluation.to_csv(output_file, index=False)


if __name__ == "__main__":
    main()
