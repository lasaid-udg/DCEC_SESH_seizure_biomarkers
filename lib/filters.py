import numpy
import scipy.signal
from . import settings
from typing import Tuple


class FilterBank():

    def __init__(self, sampling_frequency: int):
        """
        :param sampling_frequency: sampling_frequency [Hz]
        """
        self.sampling_frequency = sampling_frequency
        self.notch_qa_factor = settings["notch_qa_factor"]
        self.highpass_butter_order = settings["drift_butter_order"]
        self.lowpass_butter_order = settings["hfo_butter_order"]


    def notch(self, power_frequency: int=50, get_freqz: bool=False) -> Tuple[numpy.array, numpy.array]:
        """
        Return numerator (b) and denominator (a) polynomials of the notch filter.
        :param power_frequency: power line frequency [Hz]
        :param get_freqz: if true it will return the filter's frequency response
        """
        b, a = scipy.signal.iirnotch(power_frequency, self.notch_qa_factor,
                                     self.sampling_frequency)
    
        if get_freqz:
            frequency_range, h = scipy.signal.freqz(b, a, fs=self.sampling_frequency)
            return frequency_range, h
        return b, a

    def highpass_butter(self, cut_frequency: int=0.5, get_freqz: bool=False) -> Tuple[numpy.array, numpy.array]:
        """
        Return numerator (b) and denominator (a) polynomials of the butterworth filter.
        :param cut_frequency: critical frequency [Hz]
        :param get_freqz: if true it will return the filter's frequency response
        """
        b, a = scipy.signal.butter(self.highpass_butter_order, cut_frequency,
                                   "high", fs=self.sampling_frequency)
    
        if get_freqz:
            frequency_range, h = scipy.signal.freqz(b, a, fs=self.sampling_frequency)
            return frequency_range, h
        return b, a

    def lowpass_butter(self, cut_frequency: int=80, get_freqz: bool=False) -> Tuple[numpy.array, numpy.array]:
        """
        Return numerator (b) and denominator (a) polynomials of the butterworth filter.
        :param cut_frequency: critical frequency [Hz]
        :param get_freqz: if true it will return the filter's frequency response
        """
        b, a = scipy.signal.butter(self.lowpass_butter_order, cut_frequency,
                                   "low", fs=self.sampling_frequency)
    
        if get_freqz:
            frequency_range, h = scipy.signal.freqz(b, a, fs=self.sampling_frequency)
            return frequency_range, h
        return b, a

    def bandpass_butter(self, cut_frequency: tuple=(0, 0), order: int=2) -> Tuple[numpy.array, numpy.array]:
        """
        Return numerator (b) and denominator (a) polynomials of the butterworth filter.
        :param cut_frequency: critical frequency [Low Hz, High Hz]
        """
        b, a = scipy.signal.butter(order, cut_frequency,
                                   "bandpass", fs=self.sampling_frequency)

        return b, a

    def apply_filter(self, signal: numpy.array, filter_name: str, **kwargs) -> numpy.array:
        """
        Apply filter to a signal.
        :param signal: matrix with the eeg recording [channels x samples]
        :param filter_name: filter type
        :param kwargs: filter specifications
        """
        b, a = getattr(self, filter_name)(**kwargs)
        signal = scipy.signal.filtfilt(b, a, signal)
        return signal


class BandEstimator():

    @classmethod
    def get_eeg_bands(cls, eeg_array: numpy.array, sampling_frequency: int) -> tuple:
        filter_bank = FilterBank(sampling_frequency)
        delta = filter_bank.apply_filter(eeg_array, "bandpass_butter", **settings["band_specifications"][0])
        theta = filter_bank.apply_filter(eeg_array, "bandpass_butter", **settings["band_specifications"][1])
        alpha = filter_bank.apply_filter(eeg_array, "bandpass_butter", **settings["band_specifications"][2])
        beta = filter_bank.apply_filter(eeg_array, "bandpass_butter", **settings["band_specifications"][3])
        return delta, theta, alpha, beta