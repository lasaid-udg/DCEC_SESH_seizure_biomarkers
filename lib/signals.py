import re
import mne
import logging
import numpy as numpy
import scipy.signal
from . import settings


class EegProcessorBaseClass():

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
        self.drift_frequency = settings["drift_frequency"]
        self.hfo_frequency = settings["hfo_frequency"]
        self.channels = None
        self.data = filename
        self.filter_bank = None

    def resample(self) -> None:
        """
        Downsample the signal to match the EXPECTED_SAMPLING_FREQUENCY.
        Raise error if current sampling frequency is lower than or
        not a multiple of EXPECTED_SAMPLING_FREQUENCY
        """
        assert self.sampling_frequency >= self.EXPECTED_SAMPLING_FREQUENCY
        assert self.sampling_frequency % self.EXPECTED_SAMPLING_FREQUENCY == 0

        downsampling_factor = int(self.sampling_frequency / self.EXPECTED_SAMPLING_FREQUENCY)
        logging.info(f"Dowsampling factor is = {downsampling_factor}")
        self._data = scipy.signal.decimate(self._data, downsampling_factor)
        self.sampling_frequency = self.EXPECTED_SAMPLING_FREQUENCY

    def scale(self, ekg_reference: bool=False) -> None:
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

    def rereference_to_average(self) -> None:
        """
        Set the reference to average reference
        """
        reference_data = self._data.mean(0, keepdims=True)
        self._data -= reference_data


class EegProcessorChb(EegProcessorBaseClass):

    DATASET = "chb-mit"
    EXPECTED_SAMPLING_FREQUENCY = 256

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
        channels_to_idx = {re.sub(regex, "", x): y for x, y in zip(self.channels,
                                                                   range(len(self.channels)))}
        temp_eeg = numpy.zeros([len(self.selected_channels),
                               self._data.shape[1]])
        for idx, channel in zip(range(len(self.selected_channels)),
                                self.selected_channels):
            new_channel = self._data[channels_to_idx[channel], :]
            temp_eeg[idx, :] = new_channel
        self._data = temp_eeg


class EegProcessorSiena(EegProcessorBaseClass):

    DATASET = "siena"
    EXPECTED_SAMPLING_FREQUENCY = 256

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
    EXPECTED_SAMPLING_FREQUENCY = 256

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
        self.f_samp = data.info["sfreq"]
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
