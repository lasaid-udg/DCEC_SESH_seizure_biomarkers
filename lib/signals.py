import re
import mne
import numpy as np
from . import settings


class EegProcessorChb():

    DATASET = "chb-mit"

    def __init__(self, filename: str):
        """
        :param filename: full path of the edf file
        """
        self.sampling_frequency = settings[EegProcessorChb.DATASET]["f_samp"]
        self.gain = settings[EegProcessorChb.DATASET]["gain"]
        self.units = settings[EegProcessorChb.DATASET]["units"]
        self.selected_channels = settings[EegProcessorChb.DATASET]["channels"]
        self.channels = None
        self.data = filename

    @property
    def data(self) -> np.array:
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

    def scale(self) -> None:
        """
        Convert data from unitless to milivolts
        """
        self._data = self._data * self.gain / self.units

    def select_channels(self) -> None:
        """
        Select a subset of the available channels, then the channels are sorted.
        The list of expected channels is detailed in settings
        """
        regex = "-[0-9]$"
        channels_to_idx = {re.sub(regex, "", x): y for x, y in zip(self.channels,
                                                                   range(len(self.channels)))}
        temp_eeg = np.zeros([len(self.selected_channels),
                            self._data.shape[1]])
        for idx, channel in zip(range(len(self.selected_channels)),
                                self.selected_channels):
            new_channel = self._data[channels_to_idx[channel], :]
            temp_eeg[idx, :] = new_channel
        self._data = temp_eeg


class EegProcessorSiena():

    DATASET = "siena"

    def __init__(self, filename: str):
        """
        :param filename: full path of the edf file
        """
        self.sampling_frequency = settings[EegProcessorSiena.DATASET]["f_samp"]
        self.gain = settings[EegProcessorSiena.DATASET]["gain"]
        self.units = settings[EegProcessorSiena.DATASET]["units"]
        self.selected_channels = settings[EegProcessorSiena.DATASET]["channels"]
        self.channels = None
        self.data = filename

    @property
    def data(self) -> np.array:
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

    def scale(self) -> None:
        """
        Convert data from unitless to milivolts
        """
        self._data = self._data * self.gain / self.units

    def select_channels(self) -> None:
        """
        Select a subset of the available channels, then the channels are sorted.
        The list of expected channels is detailed in settings
        """
        regex = "^EEG "
        channels_to_idx = {re.sub(regex, "", x).lower(): y
                           for x, y in zip(self.channels,
                                           range(len(self.channels)))}
        temp_eeg = np.zeros([len(self.selected_channels),
                            self._data.shape[1]])
        for idx, channel in zip(range(len(self.selected_channels)),
                                self.selected_channels):
            new_channel = (self._data[channels_to_idx[channel.lower()], :])
            temp_eeg[idx, :] = new_channel
        self._data = temp_eeg


class EegProcessorTusz():

    DATASET = "tusz"

    def __init__(self, filename: str):
        """
        :param filename: full path of the edf file
        """
        self.sampling_frequency = settings[EegProcessorTusz.DATASET]["f_samp"]
        self.gain = settings[EegProcessorTusz.DATASET]["gain"]
        self.units = settings[EegProcessorTusz.DATASET]["units"]
        self.selected_channels = settings[EegProcessorTusz.DATASET]["channels"]
        self.channels = None
        self.data = filename

    @property
    def data(self) -> np.array:
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

    def scale(self) -> None:
        """
        Convert data from unitless to milivolts
        """
        self._data = self._data * self.gain / self.units

    def select_channels(self) -> None:
        """
        Select a subset of the available channels, then the channels are sorted.
        The list of expected channels is detailed in settings
        """
        channels_to_idx = {}
        for x, y in zip(self.channels, range(len(self.channels))):
            key = re.sub("^EEG ", "", x).split("-")[0].lower()
            channels_to_idx[key] = y

        temp_eeg = np.zeros([len(self.selected_channels),
                            self._data.shape[1]])
        for idx, channel in zip(range(len(self.selected_channels)),
                                self.selected_channels):
            new_channel = (self._data[channels_to_idx[channel.lower()], :])
            temp_eeg[idx, :] = new_channel
        self._data = temp_eeg
