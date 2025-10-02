import re
import mne
import random
import logging
import numpy as numpy
import scipy.signal
from typing import Iterable
from . import settings


class EegProcessorBaseClass():

    EXPECTED_SAMPLING_FREQUENCIES = [250, 256]

    def __init__(self, filename: str):
        """
        :param filename: full path of the edf file
        """
        self.sampling_frequency = settings[self.DATASET]["f_samp"]
        self.gain = settings[self.DATASET]["gain"]
        self.units = settings[self.DATASET]["units"]
        self.ekg_units = settings[self.DATASET]["ekg_units"]
        self.power_noise_frequency = settings[self.DATASET]["power_noise_frequency"]
        self.selected_channels = settings[self.DATASET]["channels"]
        self.bipolar_channels = self.clean_bipolar_channels(settings[self.DATASET]["univariate_channels_groups"])
        self.drift_frequency = settings["drift_frequency"]
        self.hfo_frequency = settings["hfo_frequency"]
        self.channels = None
        self.data = filename
        self.filter_bank = None
    
    def clean_bipolar_channels(self, bipolar_channels: dict):
        bipolar_channels = sum(list(bipolar_channels.values()), [])
        bipolar_channels = [tuple([y.lower() for y in x.split("-")]) for x in bipolar_channels]
        return bipolar_channels

    def resample(self) -> None:
        """
        Downsample the signal to match any of EXPECTED_SAMPLING_FREQUENCIES.
        Raise error if current sampling frequency is lower than or
        not a multiple of any of EXPECTED_SAMPLING_FREQUENCIES
        """
        for expected_frequency in self.EXPECTED_SAMPLING_FREQUENCIES:
            if (self.sampling_frequency >= expected_frequency and
                    self.sampling_frequency % expected_frequency == 0):
                selected_frequency = expected_frequency
                break
        else:
            assert False, "Not a valid sampling frequency"

        logging.info(f"Selected frequency is = {selected_frequency}")
        downsampling_factor = int(self.sampling_frequency / selected_frequency)
        logging.info(f"Dowsampling factor is = {downsampling_factor}")
        self._data = scipy.signal.decimate(self._data, downsampling_factor)
        self.sampling_frequency = selected_frequency

    def scale(self, ekg_reference: bool = False) -> None:
        """
        Convert data from unitless to milivolts
        """
        if not ekg_reference:
            self._data = self._data * self.gain / self.units
        else:
            self._data[:-1, :] = self._data[:-1, :] * self.gain / self.units
            self._data[-1, :] = (self._data[-1, :] - numpy.mean(self._data[-1, :])) / self.ekg_units

    def remove_drift(self) -> None:
        """
        Remove the signal drift, caused by subject movement and electrode-skin contact
        """
        self.filter_bank.sampling_frequency = self.sampling_frequency
        self._data = self.filter_bank.apply_filter(self._data, "highpass_butter",
                                                   cut_frequency=self.drift_frequency)

    def remove_hfo(self) -> None:
        """
        Remove the high frequency oscillation, frequencies above gamma band
        """
        self.filter_bank.sampling_frequency = self.sampling_frequency
        self._data = self.filter_bank.apply_filter(self._data, "lowpass_butter",
                                                   cut_frequency=self.hfo_frequency)

    def remove_power_noise(self) -> None:
        """
        Remove the noise caused by the machine power line
        """
        self.filter_bank.sampling_frequency = self.sampling_frequency
        self._data = self.filter_bank.apply_filter(self._data, "notch",
                                                   power_frequency=self.power_noise_frequency)

    def convert_to_bipolar(self, eeg_array: numpy.array) -> None:
        """
        Convert the recording channels from monopolar to bipolar longitudinal
        :param eeg_array: matrix with the eeg recording [channels x samples]
        """
        bipolar_array = numpy.zeros((len(self.bipolar_channels), eeg_array.shape[1]))
        channel_to_idx = {channel.lower(): idx for idx, channel in enumerate(self.selected_channels)}
        for idx, (ch1, ch2) in enumerate(self.bipolar_channels):
            bipolar_array[idx, :] = eeg_array[channel_to_idx[ch1], :] - eeg_array[channel_to_idx[ch2], :]
        return bipolar_array

    @classmethod
    def rereference_to_average(cls, eeg_array: numpy.array) -> numpy.array:
        """
        Set the reference to average reference
        :param eeg_array: matrix with the eeg recording [channels x samples]
        """
        reference_data = eeg_array.mean(0, keepdims=True)
        eeg_array -= reference_data
        return eeg_array

    @classmethod
    def standardize(cls, eeg_array: numpy.array) -> numpy.array:
        """
        Remove the mean and divide by the standard deviation
        :param eeg_array: matrix with the eeg recording [channels x samples]
        """
        signal_mean = numpy.mean(eeg_array, axis=1)
        signal_std = numpy.std(eeg_array, axis=1)

        for idx in range(len(signal_std.shape)):
            eeg_array[idx, :] = (eeg_array[idx, :] - signal_mean[idx]) / signal_std[idx]
        return eeg_array


class EegProcessorChb(EegProcessorBaseClass):

    DATASET = "chb-mit"

    def __init__(self, filename: str):
        """
        :param filename: full path of the edf file
        """
        super().__init__(filename)

    @property
    def data(self) -> numpy.array:
        return self._data

    @data.setter
    def data(self, filename: str) -> None:
        """
        Read eeg recording and channels from edf file
        :param filename: full path of the edf file
        """
        data = mne.io.read_raw_edf(filename)
        self._data = data.get_data()
        self.channels = data.ch_names
        logging.info(f"Recording contains channels = {self.channels}")

    def select_channels(self) -> None:
        """
        Select a subset of the available channels, then the channels are sorted.
        The list of expected channels is detailed in settings
        """
        regex = "-[0-9]$"
        channels_to_idx = {re.sub(regex, "", x).lower(): y for x, y in zip(self.channels,
                                                                   range(len(self.channels)))}
        temp_eeg = numpy.zeros([len(self.selected_channels),
                               self._data.shape[1]])
        for idx, channel in zip(range(len(self.selected_channels)),
                                self.selected_channels):
            new_channel = self._data[channels_to_idx[channel.lower()], :]
            temp_eeg[idx, :] = new_channel
        self._data = temp_eeg


class EegProcessorSiena(EegProcessorBaseClass):

    DATASET = "siena"

    def __init__(self, filename: str):
        """
        :param filename: full path of the edf file
        """
        super().__init__(filename)

    @property
    def data(self) -> numpy.array:
        return self._data

    @data.setter
    def data(self, filename: str) -> None:
        """
        Read eeg recording and channels from edf file
        :param filename: full path of the edf file
        """
        data = mne.io.read_raw_edf(filename)
        self._data = data.get_data()
        self.channels = data.ch_names
        logging.info(f"Recording contains channels = {self.channels}")

    def select_channels(self) -> None:
        """
        Select a subset of the available channels, then the channels are sorted.
        The list of expected channels is detailed in settings
        """
        regex_eeg = "^EEG "
        regex_ekg = "^ekg "
        channels_to_idx = {re.sub(regex_eeg, "", x).lower(): y
                           for x, y in zip(self.channels,
                                           range(len(self.channels)))}
        channels_to_idx = {re.sub(regex_ekg, "", channel).lower(): idx
                           for channel, idx in channels_to_idx.items()}
        temp_eeg = numpy.zeros([len(self.selected_channels),
                               self._data.shape[1]])
        for idx, channel in zip(range(len(self.selected_channels)),
                                self.selected_channels):
            new_channel = (self._data[channels_to_idx[channel.lower()], :])
            temp_eeg[idx, :] = new_channel
        self._data = temp_eeg


class EegProcessorTusz(EegProcessorBaseClass):

    DATASET = "tusz"
    female = 0
    male = 0
    unknown = 0
    sampling_frequencies = set()

    def __init__(self, filename: str):
        """
        :param filename: full path of the edf file
        """
        super().__init__(filename)

    @property
    def data(self) -> numpy.array:
        return self._data

    @data.setter
    def data(self, filename: str) -> None:
        """
        Read eeg recording and channels from edf file
        :param filename: full path of the edf file
        Note: subject sex 0=unknown, 1=male, 2=female
        """
        data = mne.io.read_raw_edf(filename)
        if data.info["subject_info"]["sex"] == 1:
            EegProcessorTusz.male += 1
        elif data.info["subject_info"]["sex"] == 2:
            EegProcessorTusz.female += 1
        else:
            EegProcessorTusz.unknown += 1

        self._data = data.get_data()
        self.sampling_frequency = data.info["sfreq"]
        EegProcessorTusz.sampling_frequencies.add(self.sampling_frequency)
        self.channels = data.ch_names
        logging.info(f"Recording contains channels = {self.channels}")

    def select_channels(self) -> None:
        """
        Select a subset of the available channels, then the channels are sorted.
        The list of expected channels is detailed in settings
        """
        channels_to_idx = {}
        for x, y in zip(self.channels, range(len(self.channels))):
            key = re.sub("^EEG ", "", x).split("-")[0].lower()
            channels_to_idx[key] = y

        temp_eeg = numpy.zeros([len(self.selected_channels),
                               self._data.shape[1]])
        for idx, channel in zip(range(len(self.selected_channels)),
                                self.selected_channels):
            new_channel = (self._data[channels_to_idx[channel.lower()], :])
            temp_eeg[idx, :] = new_channel
        self._data = temp_eeg


class EegProcessorTuep(EegProcessorBaseClass):

    DATASET = "tuep"

    def __init__(self, filename: str):
        """
        :param filename: full path of the edf file
        """
        super().__init__(filename)

    @property
    def data(self) -> numpy.array:
        return self._data

    @data.setter
    def data(self, filename: str) -> None:
        """
        Read eeg recording and channels from edf file
        :param filename: full path of the edf file
        """
        data = mne.io.read_raw_edf(filename)
        self._data = data.get_data()
        self.sampling_frequency = data.info["sfreq"]
        self.channels = data.ch_names
        logging.info(f"Recording contains channels = {self.channels}")

    def select_channels(self) -> None:
        """
        Select a subset of the available channels, then the channels are sorted.
        The list of expected channels is detailed in settings
        """
        channels_to_idx = {}
        for x, y in zip(self.channels, range(len(self.channels))):
            key = re.sub("^EEG ", "", x).split("-")[0].lower()
            channels_to_idx[key] = y

        temp_eeg = numpy.zeros([len(self.selected_channels),
                               self._data.shape[1]])
        for idx, channel in zip(range(len(self.selected_channels)),
                                self.selected_channels):
            new_channel = (self._data[channels_to_idx[channel.lower()], :])
            temp_eeg[idx, :] = new_channel
        self._data = temp_eeg


class EegSlicer():

    def __init__(self, sampling_frequency: int):
        """
        :param sampling_frequency: sampling_frequency [Hz]
        """
        self.preictal_min_length = settings["preictal_min_length"]
        self.ictal_min_length = settings["ictal_min_lenght"]
        self.postictal_min_lenght = settings["postictal_min_lenght"]
        self.sampling_frequency = sampling_frequency

    def compute_slices(self, seizure_ranges: list, eeg_array: numpy.array) -> Iterable[tuple]:
        """
        Split a full seizure cycle: preictal stage - ictal stage - postictal state.
        Eeg array and metadata are stored.
        :param seizure_ranges: list of seizure ocurrences [start time, end time, seizure type]
        :param eeg_array: full eeg recording
        """
        for idx in range(1, len(seizure_ranges[:-1])):
            if (seizure_ranges[idx][0] - seizure_ranges[idx - 1][1]) < self.preictal_min_length:
                logging.info(f"Preictal period is less than = {self.preictal_min_length}, skipping seizure")
                continue
            if (seizure_ranges[idx][1] - seizure_ranges[idx][0]) < self.ictal_min_length:
                logging.info(f"Ictal period is less than = {self.ictal_min_length}, skipping seizure")
                continue
            if (seizure_ranges[idx + 1][0] - seizure_ranges[idx][1]) < self.postictal_min_lenght:
                logging.info(f"Postictal period is less than = {self.postictal_min_lenght}, skipping seizure")
                continue

            logging.info("Seizure meets the pre-, post- and ictal tolerance")

            slice_start = (seizure_ranges[idx][0] - self.preictal_min_length) * self.sampling_frequency
            slice_end = (seizure_ranges[idx][1] + self.postictal_min_lenght) * self.sampling_frequency

            seizure_duration = seizure_ranges[idx][1] - seizure_ranges[idx][0]
            new_seizure_start = self.preictal_min_length
            new_seizure_end = int(self.preictal_min_length + seizure_duration)
            eeg_slice = eeg_array[:, int(slice_start): int(slice_end)]
            metadata = {"seizure_type": seizure_ranges[idx][2],
                        "seizure_start": new_seizure_start,
                        "seizure_end": new_seizure_end}

            logging.info(f"Valid slice was found, seizure_duration = {new_seizure_end - new_seizure_start}")

            yield metadata, eeg_slice

    def compute_random_slices(self, number_slices: int, eeg_array: numpy.array) -> Iterable[numpy.array]:
        """
        Select a set of random slices from an eeg recording.
        :param number_slices: number of slices
        :param eeg_array: full eeg recording [channels x samples]
        """
        for _ in range(number_slices):
            middle_point = random.randint(0, eeg_array.shape[1])
            slice_start = middle_point - (self.preictal_min_length * self.sampling_frequency)
            slice_end = middle_point + (self.postictal_min_lenght * self.sampling_frequency)

            slice_start = slice_start if slice_start > 0 else 0
            slice_end = slice_end if slice_end < eeg_array.shape[1] else eeg_array.shape[1]
            eeg_slice = eeg_array[:, slice_start: slice_end]

            logging.info("Valid slice was found")
            yield eeg_slice
