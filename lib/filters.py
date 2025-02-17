import scipy.signal
from . import settings


class FilterBank():

    def __init__(self, sampling_frequency: int):
        self.sampling_frequency = sampling_frequency
        self.notch_qa_factor = settings["notch_qa_factor"]
        self.highpass_butter_order = settings["drift_butter_order"]
        self.lowpass_butter_order = settings["hfo_butter_order"]


    def notch(self, power_frequency: int=50, get_freqz: bool=False) -> tuple:
        b, a = scipy.signal.iirnotch(power_frequency, self.notch_qa_factor,
                                     self.sampling_frequency)
    
        if get_freqz:
            frequency_range, h = scipy.signal.freqz(b, a, fs=self.sampling_frequency)
            return frequency_range, h
        return b, a

    def highpass_butter(self, cut_frequency: int=0.5, get_freqz: bool=False) -> tuple:
        b, a = scipy.signal.butter(self.highpass_butter_order, cut_frequency,
                                   "high", fs=self.sampling_frequency)
    
        if get_freqz:
            frequency_range, h = scipy.signal.freqz(b, a, fs=self.sampling_frequency)
            return frequency_range, h
        return b, a

    def lowpass_butter(self, cut_frequency: int=80, get_freqz: bool=False) -> tuple:
        b, a = scipy.signal.butter(self.lowpass_butter_order, cut_frequency,
                                   "low", fs=self.sampling_frequency)
    
        if get_freqz:
            frequency_range, h = scipy.signal.freqz(b, a, fs=self.sampling_frequency)
            return frequency_range, h
        return b, a

    def apply_filter(self, signal, filter_name, **kwargs):
        b, a = getattr(self, filter_name)(**kwargs)
        signal = scipy.signal.filtfilt(b, a, signal)
        return signal
