import scipy.signal


class FilterBank():

    NOTCH_QA_FACTOR = 20
    HIGHPASS_BUTTER_ORDER = 5
    LOWPASS_BUTTER_ORDER = 10

    def __init__(self, sampling_frequency: int):
        self.sampling_frequency = sampling_frequency
    
    def __call__(self, filter_name, **kwargs):
        return getattr(self, filter_name)(**kwargs)
    
    def notch(self, power_frequency: int=50, get_freqz: bool=False) -> tuple:
        b, a = scipy.signal.iirnotch(power_frequency, FilterBank.NOTCH_QA_FACTOR,
                                     self.sampling_frequency)
    
        if get_freqz:
            frequency_range, h = scipy.signal.freqz(b, a, fs=self.sampling_frequency)
            return frequency_range, h
        return b, a

    def highpass_butter(self, cut_frequency: int=0.5, get_freqz: bool=False) -> tuple:
        b, a = scipy.signal.butter(FilterBank.HIGHPASS_BUTTER_ORDER, cut_frequency,
                                   "high", fs=self.sampling_frequency)
    
        if get_freqz:
            frequency_range, h = scipy.signal.freqz(b, a, fs=self.sampling_frequency)
            return frequency_range, h
        return b, a

    def lowpass_butter(self, cut_frequency: int=80, get_freqz: bool=False) -> tuple:
        b, a = scipy.signal.butter(FilterBank.LOWPASS_BUTTER_ORDER, cut_frequency,
                                   "low", fs=self.sampling_frequency)
    
        if get_freqz:
            frequency_range, h = scipy.signal.freqz(b, a, fs=self.sampling_frequency)
            return frequency_range, h
        return b, a

    def apply_filter(self, signal, filter_name, **kwargs):
        b, a = getattr(self, filter_name)(**kwargs)
        signal = scipy.signal.lfilter(b, a, self._data)
        return signal
