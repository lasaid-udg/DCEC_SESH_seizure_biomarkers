import os
import glob
import json
import numpy
import random
from typing import Tuple
from . import settings


class EegSlices():

    def __init__(self):
        base_path = os.getenv("BIOMARKERS_PROJECT_HOME")
        base_path = os.path.join(base_path, "slices", self.DATASET)
        self.channels = self.clean_bipolar_channels(settings[self.DATASET]["univariate_channels_groups"])
        self._metadata = {}
        self.base_path = base_path
        self.metadata = base_path

    @property
    def metadata(self) -> dict:
        return self._metadata

    @metadata.setter
    def metadata(self, directory: str) -> None:
        """
        Load the metadata files from slices directory
        :param directory: folder where slices are stored into
        """
        for x in glob.glob(os.path.join(directory, "*.json")):
            with open(x, "r") as fp:
                single_metadata = json.load(fp)
            if single_metadata["patient"] not in self._metadata:
                self._metadata[single_metadata["patient"]] = []
            self._metadata[single_metadata["patient"]].append(single_metadata)

    def clean_bipolar_channels(self, bipolar_channels: dict):
        bipolar_channels = sum(list(bipolar_channels.values()), [])
        return bipolar_channels

    def get(self, patient: str, seizure_number: int) -> Tuple[dict, numpy.array]:
        """
        Return metadata and eeg slice for a single patient
        :param patient: patient id
        :param seizure_number: seizure id
        """
        metadata = self._metadata[patient][seizure_number]
        metadata["channels"] = self.channels
        if "Initiative1_epic1" in metadata["slice_file"]:
            metadata["slice_file"] = metadata["slice_file"].replace("Initiative1_epic1", "Initiative1_univariate")
        eeg_slice = numpy.load(metadata["slice_file"])
        return metadata, eeg_slice

    def summarize(self) -> None:
        """
        Print the count of events per seizure type
        """
        summary = {}
        patients = {}
        for patient_metadata in self.metadata.values():
            for metadata in patient_metadata:
                if metadata["seizure_type"] not in summary:
                    summary[metadata["seizure_type"]] = 0
                    patients[metadata["seizure_type"]] = set()
                summary[metadata["seizure_type"]] += 1
                patients[metadata["seizure_type"]].add(metadata["patient"])
        print(json.dumps(summary, indent=4))

        for seizure_type, quantity in patients.items():
            print(f"Seizure type = {seizure_type}, quantity = {len(quantity)}")


class EegSlicesChb(EegSlices):
    DATASET = "chb-mit"


class EegSlicesSiena(EegSlices):
    DATASET = "siena"


class EegSlicesTusz(EegSlices):
    DATASET = "tusz"


class EegSlicesTuep(EegSlices):
    DATASET = "tuep"


class WindowSelector():

    def __init__(self, sampling_frequency: str):
        """
        :param sampling_frequency: sampling_frequency [Hz]
        """
        self.sampling_frequency = sampling_frequency
        self.window_lenght = settings["window_length"] * sampling_frequency
        self.tolerance_lenght = settings["tolerance_length"] * sampling_frequency
        self.preictal_windows_start = settings["preictal_windows_start"]
        self.postictal_windows_start = settings["postictal_windows_start"]

    def get_preictal_windows(self, metadata: dict, eeg_array: numpy.array) -> list:
        """
        :param metadata: seizure and eeg recording details
        :param eeg_array: eeg recording [channels x samples]
        """
        seizure_start = metadata["seizure_start"] * self.sampling_frequency
        windows = list()

        counter = len(metadata["windows"])
        for start in self.preictal_windows_start:
            metadata["windows"][counter] = ("preictal", start)
            window_end = (seizure_start - self.tolerance_lenght) - start * self.sampling_frequency
            window_start = window_end - self.window_lenght
            windows.append(eeg_array[:, int(window_start): int(window_end)])
            counter += 1

        return windows

    def get_ictal_windows(self, metadata: dict, eeg_array: numpy.array) -> list:
        """
        :param metadata: seizure and eeg recording details
        :param eeg_array: eeg recording [channels x samples]
        """
        counter = len(metadata["windows"])

        seizure_start = metadata["seizure_start"] * self.sampling_frequency
        window_start = (seizure_start + self.tolerance_lenght)
        window_end = window_start + self.window_lenght
        window_1 = eeg_array[:, int(window_start): int(window_end)]
        metadata["windows"][counter] = ("ictal", 1)

        middle_point = (metadata["seizure_end"] - metadata["seizure_start"]) / 2

        window_start = ((metadata["seizure_start"] + middle_point) * self.sampling_frequency)
        window_end = window_start + self.window_lenght
        window_2 = eeg_array[:, int(window_start): int(window_end)]
        metadata["windows"][counter + 1] = ("ictal", 2)

        seizure_end = metadata["seizure_end"] * self.sampling_frequency
        window_end = (seizure_end - self.tolerance_lenght)
        window_start = window_end - self.window_lenght
        window_3 = eeg_array[:, int(window_start): int(window_end)]
        metadata["windows"][counter + 2] = ("ictal", -1)

        return [window_1, window_2, window_3]

    def get_postictal_windows(self, metadata: dict, eeg_array: numpy.array) -> list:
        """
        :param metadata: seizure and eeg recording details
        :param eeg_array: eeg recording [channels x samples]
        """
        seizure_end = metadata["seizure_end"] * self.sampling_frequency
        windows = list()

        counter = len(metadata["windows"])
        for start in self.postictal_windows_start:
            metadata["windows"][counter] = ("postictal", start)
            window_start = (seizure_end + self.tolerance_lenght) + start * self.sampling_frequency
            window_end = window_start + self.window_lenght
            windows.append(eeg_array[:, int(window_start): int(window_end)])
            counter += 1

        return windows

    def get_windows(self, metadata: dict, eeg_array: numpy.array, ) -> Tuple[dict, numpy.array]:
        """
        Compute the eeg windows for pre-, post- and ictal stages
        :param metadata: seizure and eeg recording details
        :param eeg_array: eeg recording [channels x samples]
        """
        metadata["windows"] = dict()
        preictal_windows = self.get_preictal_windows(metadata, eeg_array)
        ictal_windows = self.get_ictal_windows(metadata, eeg_array)
        postictal_windows = self.get_postictal_windows(metadata, eeg_array)

        return metadata, numpy.stack(preictal_windows + ictal_windows + postictal_windows)

    def get_random_window(self, metadata: dict, eeg_array: numpy.array) -> Tuple[dict, numpy.array]:
        """
        Crop a random window from eeg_array
        :param metadata: seizure and eeg recording details
        :param eeg_array: eeg recording [channels x samples]
        """
        window_start = random.randint(0, eeg_array.shape[1] - self.window_lenght)
        window_end = window_start + self.window_lenght
        eeg_window = eeg_array[:, int(window_start): int(window_end)]
        return metadata, eeg_window


class EegWindows():

    def __init__(self):
        base_path = os.getenv("BIOMARKERS_PROJECT_HOME")
        base_path = os.path.join(base_path, "windows", self.DATASET)
        self.file_list = base_path

    def __iter__(self):
        self._counter = 0
        return self

    def __next__(self) -> Tuple[str, str]:
        if self._counter >= len(self._file_list):
            raise StopIteration
        metadata, windows = self._file_list[self._counter]

        windows = numpy.load(windows)
        with open(metadata) as fp:
            metadata = json.load(fp)

        self._counter += 1

        return metadata, windows

    @property
    def file_list(self) -> list:
        return self._file_list

    @file_list.setter
    def file_list(self, directory: str) -> None:
        """
        Read list of files for metadata and eeg windows
        :param directory: windows directory
        """
        metadata_files = sorted(glob.glob(os.path.join(directory, "*.json")))
        windows_files = sorted(glob.glob(os.path.join(directory, "*.npy")))
        self._file_list = [(metadata, window) for metadata, window in zip(metadata_files, windows_files)]


class EegWindowsChb(EegWindows):
    DATASET = "chb-mit"


class EegWindowsSiena(EegWindows):
    DATASET = "siena"


class EegWindowsTusz(EegWindows):
    DATASET = "tusz"


class EegWindowsTuep(EegWindows):
    DATASET = "tuep"
