#!/var/tmp/venv-project-1/bin/python
import os
import sys
import json
import numpy
import logging
import warnings
warnings.filterwarnings("ignore")
sys.path.append("../")
from lib.slices import WindowSelector, EegSlicesTuep
from lib.signals import EegProcessorBaseClass


OUTPUT_DIRECTORY = os.getenv("BIOMARKERS_PROJECT_HOME")


def main():
    slices_tuep = EegSlicesTuep()
    global_metadata = slices_tuep.metadata
    
    for patient, patient_metadata in global_metadata.items():
        logging.info(f"Processing patient = {patient}")
        for slice_number in range(len(patient_metadata)):
            logging.info(f"Processing eeg slice")
            slice_metadata, eeg_slice = slices_tuep.get(patient, slice_number)
            eeg_slice = EegProcessorBaseClass.standardize(eeg_slice)

            selector = WindowSelector(slice_metadata["sampling_frequency"])

            try:
                slice_metadata, eeg_window = selector.get_random_window(slice_metadata, eeg_slice)
            except ValueError as exc:
                logging.error(f"Window computation failed, error = {exc}")
                continue

            output_file_eeg = f"{patient}_{slice_number}.npy"
            output_file_eeg = os.path.join(OUTPUT_DIRECTORY, "windows", "tuep", output_file_eeg)
            numpy.save(output_file_eeg, eeg_window)

            output_file_eeg = output_file_eeg.replace(".npy", ".json")

            with open(output_file_eeg, "w") as fp:
                json.dump(slice_metadata, fp, indent=4)


if __name__ == "__main__":
    main()