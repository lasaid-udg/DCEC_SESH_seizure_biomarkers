#!/var/tmp/venv-project-1/bin/python
import os
import sys
import json
import numpy
import logging
import warnings
warnings.filterwarnings("ignore")
sys.path.append("../")
from lib.metadata import MetadataListTusz
from lib.signals import EegProcessorTusz, EegSlicer
from lib.filters import FilterBank
from lib.bss import EogDenoiser, EmgDenoiser


OUTPUT_DIRECTORY = os.getenv("BIOMARKERS_PROJECT_HOME")


def main():
    metadata = MetadataListTusz()

    for patient, patient_metadata in metadata.patient_metadata.items():
        logging.info(f"Analizyng patient = {patient}")
        seizure_counter = 0

        for filename, seizures in patient_metadata.seizure_ranges.items():
            logging.info((f"Analyzing file = {filename}"))
            logging.info((f"Number of seizures = {len(seizures['seizures']) - 1}"))

            if len(seizures["seizures"]) < 2:
                logging.info((f"Not enough seizures, skipping patient"))
                continue

            try:
                processor = EegProcessorTusz(seizures["full_file"])
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

            end_of_file = int(processor._data.shape[1] / processor.sampling_frequency)
            seizures["seizures"] = list(seizures["seizures"])
            seizures["seizures"].append((end_of_file, end_of_file, "NULL"))

            slicer = EegSlicer(processor.sampling_frequency)
            for slice_metadata, slice_eeg in slicer.compute_slices(seizures["seizures"], processor._data):

                output_file_eeg = f"{patient}_{seizure_counter}.npy"
                output_file_eeg = os.path.join(OUTPUT_DIRECTORY, "slices", "tusz", output_file_eeg)
                numpy.save(output_file_eeg, slice_eeg)

                slice_metadata["patient"] = patient
                slice_metadata["sampling_frequency"] = processor.sampling_frequency
                slice_metadata["source_file"] = seizures["full_file"]
                slice_metadata["slice_file"] = output_file_eeg

                output_file_eeg = output_file_eeg.replace(".npy", ".json")

                with open(output_file_eeg, "w") as fp:
                    json.dump(slice_metadata, fp, indent=4)

                seizure_counter += 1


if __name__ == "__main__":
    main()