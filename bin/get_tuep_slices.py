#!/var/tmp/venv-project-1/bin/python
import os
import sys
import json
import numpy
import logging
import warnings
warnings.filterwarnings("ignore")
sys.path.append("../")
from lib.metadata import MetadataListTuep
from lib.signals import EegProcessorTuep, EegSlicer
from lib.filters import FilterBank


OUTPUT_DIRECTORY = os.getenv("BIOMARKERS_PROJECT_HOME")


def main():
    metadata = MetadataListTuep()

    for patient, patient_metadata in metadata.patient_metadata.items():
        logging.info(f"Analizyng patient = {patient}")
        counter = 0

        for filename, recordings in patient_metadata.recordings.items():
            logging.info((f"Analyzing file = {filename}"))

            try:
                processor = EegProcessorTuep(recordings["full_file"])
            except FileNotFoundError as exc:
                logging.error(f"Error found {exc}")
                continue

            processor.scale()

            try:
                processor.select_channels()
            except KeyError as exc:
                logging.error(f"Not valid channel set = {exc}")
                continue

            try:
                processor.resample()
            except AssertionError as exc:
                logging.error(f"Not valid sampling frequency = {exc}")
                continue

            processor.filter_bank = FilterBank(0)
            processor.remove_drift()
            processor.remove_hfo()
            processor.remove_power_noise()

            slicer = EegSlicer(processor.sampling_frequency)
            for slice_eeg in slicer.compute_random_slices(4, processor._data):
                output_file_eeg = f"{patient}_{counter}.npy"
                output_file_eeg = os.path.join(OUTPUT_DIRECTORY, "slices", "tuep", output_file_eeg)
                numpy.save(output_file_eeg, slice_eeg)

                slice_metadata = {"patient": patient,
                                  "sampling_frequency": processor.sampling_frequency,
                                  "source_file": recordings["full_file"],
                                  "slice_file": output_file_eeg}

                output_file_eeg = output_file_eeg.replace(".npy", ".json")
                with open(output_file_eeg, "w") as fp:
                    json.dump(slice_metadata, fp, indent=4)

                counter += 1


if __name__ == "__main__":
    main()