#!/var/tmp/venv-project-1/bin/python
import os
import sys
import json
import numpy
import logging
import warnings
warnings.filterwarnings("ignore")
sys.path.append("../")
from lib.slices import WindowSelector, EegSlicesSiena
from lib.signals import EegProcessorBaseClass
from lib.bss import EogDenoiser, EmgDenoiser


OUTPUT_DIRECTORY = os.getenv("BIOMARKERS_PROJECT_HOME")


def main():
    slices_siena = EegSlicesSiena()
    global_metadata = slices_siena.metadata

    for patient, patient_metadata in global_metadata.items():
        logging.info(f"Processing patient = {patient}")
        for seizure_number in range(len(patient_metadata)):
            logging.info("Processing eeg slice")
            seizure_metadata, eeg_slice = slices_siena.get(patient, seizure_number)

            ###########################################################
            eog_denoiser = EogDenoiser(seizure_metadata["sampling_frequency"])
            _, eeg_slice = eog_denoiser.apply_by_segments(eeg_slice)

            emg_denoiser = EmgDenoiser(seizure_metadata["sampling_frequency"])
            _, eeg_slice = emg_denoiser.apply_by_segments(eeg_slice)

            ###########################################################
            eeg_slice = EegProcessorBaseClass.rereference_to_average(numpy.abs(eeg_slice))

            ###########################################################
            eeg_slice = EegProcessorBaseClass.standardize(eeg_slice)

            selector = WindowSelector(seizure_metadata["sampling_frequency"])
            try:
                seizure_metadata, eeg_windows = selector.get_windows(seizure_metadata, eeg_slice)
            except ValueError as exc:
                logging.error(f"Window computation failed, error = {exc}")
                continue

            output_file_eeg = f"{patient}_{seizure_number}.npy"
            output_file_eeg = os.path.join(OUTPUT_DIRECTORY, "windows", "siena", output_file_eeg)
            numpy.save(output_file_eeg, eeg_windows)

            output_file_eeg = output_file_eeg.replace(".npy", ".json")

            with open(output_file_eeg, "w") as fp:
                json.dump(seizure_metadata, fp, indent=4)


if __name__ == "__main__":
    main()
