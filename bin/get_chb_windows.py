#!/var/tmp/venv-project-1/bin/python
import os
import sys
import json
import numpy
import logging
import warnings
warnings.filterwarnings("ignore")
sys.path.append("../")
from lib.slices import WindowSelector, EegSlicesChb
from lib.signals import EegProcessorBaseClass


OUTPUT_DIRECTORY = os.getenv("BIOMARKERS_PROJECT_HOME")


def main():
    slices_chb = EegSlicesChb()
    global_metadata = slices_chb.metadata
    
    for patient, patient_metadata in global_metadata.items():
        logging.info(f"Processing patient = {patient}")
        for seizure_number in range(len(patient_metadata)):
            logging.info(f"Processing eeg slice")
            seizure_metadata, eeg_slice = slices_chb.get(patient, seizure_number)
            eeg_slice = EegProcessorBaseClass.standardize(eeg_slice)

            selector = WindowSelector(seizure_metadata["sampling_frequency"])
            seizure_metadata, eeg_windows = selector.get_windows(seizure_metadata, eeg_slice)

            output_file_eeg = f"{patient}_{seizure_number}.npy"
            output_file_eeg = os.path.join(OUTPUT_DIRECTORY, "windows", "chb-mit", output_file_eeg)
            numpy.save(output_file_eeg, eeg_windows)

            output_file_eeg = output_file_eeg.replace(".npy", ".json")

            with open(output_file_eeg, "w") as fp:
                json.dump(seizure_metadata, fp, indent=4)


if __name__ == "__main__":
    main()