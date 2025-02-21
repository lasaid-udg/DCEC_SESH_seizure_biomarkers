import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def generate_seizure_line(start_time: int, end_time: int, seizure_ranges: list) -> list:
    """
    Return a horizontal line if selected eeg period contains a seizure
    :param start_time: start [sec] of the selected eeg period
    :param end_time: end [sec] of the selected eeg period
    :param seizure_ranges: list of [start_time, end_time] for each seizure
    """
    horizontal_lines = []

    for seizure_range in seizure_ranges[1:]:
        if seizure_range[0] > start_time and seizure_range[0] < end_time:
            x_coordinate = int(seizure_range[0])
            horizontal_lines.append([x_coordinate,  x_coordinate, "r"])
        if seizure_range[1] > start_time and seizure_range[1] < end_time:
            x_coordinate = int(seizure_range[1])
            horizontal_lines.append([x_coordinate,  x_coordinate, "b"])
    
    return horizontal_lines


def plot_eeg_windows(eeg_array: np.array, metadata: dict, channels_list: list,
                     sampling_frequency: int, period: list, output_file: str = None):
    """
    Plot the eeg recording
    :param eeg_array: matrix of eeg recordings [channels x samples]
    :param metadata: details about seizure ranges
    :param channel_list: list of channels
    :param sampling_frequency: eeg sampling frequency [Hz]
    :param period: selected eeg period [start_time, end_time]
    :param output_file: if specified the figure will be saved
    """
    grid_specs = GridSpec(1, 1, wspace=0.08, hspace=0.25)
    fig = plt.figure(figsize=(5.5, 4))

    time = np.linspace(period[0], period[1], int((period[1] - period[0]) * sampling_frequency))
    eeg_array = eeg_array[:, int(time[0] * sampling_frequency): int(time[-1] * sampling_frequency)]


    #########################################################
    ax = fig.add_subplot(grid_specs[0, 0])
    space = np.max(np.max(eeg_array)) / 1.5

    for count, channel in enumerate(channels_list):
        ax.plot(time, eeg_array[count, :] - space * (count + 1), label=channel,
                linewidth=1)

    #########################################################
    seizure_lines = generate_seizure_line(period[0], period[1],
                                         metadata["seizures"])
    if not seizure_lines:
        ax.set_title("Background EEG", fontsize=8)
    else:
        ax.set_title("Seizure EEG", fontsize=8)
        for line in seizure_lines:
            ax.plot(line[:2], (space, -space * (count + 2)), linewidth=1, color=line[2])

    #########################################################
    ax.set(yticks=np.arange(-space * len(channels_list), 0, space),
           yticklabels=reversed(channels_list))
    ax.set_xlabel("Time [s]", fontsize=8)
    ax.set_ylabel("Channels", fontsize=8)
    ax.xaxis.set_tick_params(labelsize=8)
    ax.yaxis.set_tick_params(labelsize=6)

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight", pad_inches=0.2,
                    transparent=False, facecolor='white')
    plt.show()


def plot_frequency_and_phase_response(frequency_range: list, response: list, output_file: str = None):
    """
    Plot the filter's frequency and phase response
    :param frequency_range: the frequencies at which ´response´ was computed
    :param h: the frequency response, as complex numbers
    """
    grid_specs = GridSpec(2, 1, wspace=0.08, hspace=0.50)
    fig = plt.figure(figsize=(5.5, 4))

    #########################################################
    ax = fig.add_subplot(grid_specs[0, 0])
    ax.plot(frequency_range, 20*np.log10(abs(response)), linewidth=1, color="r")
    ax.set_title("Filter Frequency Response", fontsize=8)
    ax.set_ylabel("Amplitude decrease [dB]", fontsize=8)
    ax.set_xlabel("Frequency [Hz]", fontsize=8)
    ax.xaxis.set_tick_params(labelsize=8)
    ax.yaxis.set_tick_params(labelsize=6)

    #########################################################
    ax = fig.add_subplot(grid_specs[1, 0])
    ax.plot(frequency_range, np.unwrap(np.angle(response))*180/np.pi, linewidth=1, color="b")
    ax.set_title("Filter Phase Response", fontsize=8)
    ax.set_ylabel("Phase shift [deg]", fontsize=8)
    ax.set_xlabel("Frequency [Hz]", fontsize=8)
    ax.xaxis.set_tick_params(labelsize=8)
    ax.yaxis.set_tick_params(labelsize=6)

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight", pad_inches=0.2,
                    transparent=False, facecolor='white')
    plt.show()


def plot_eeg_spectrum(frequency_range: np.array, spectral_components: np.array, channels_list: list,
                      channel: str, output_file: str = None):
    """
    Plot the eeg spectrum
    :param frequency_range: the frequencies at which spectral_components were computed
    :param spectral_components: spectral components as complex numbers
    :param channel_list: list of channels
    :param channel: channel of interest
    :param output_file: if specified the figure will be saved
    """
    grid_specs = GridSpec(1, 1, wspace=0.2, hspace=0.25)
    fig = plt.figure(figsize=(8.5, 4))
    selected_channel = [idx for idx, _ in enumerate(channels_list)][0]

    #########################################################
    ax = fig.add_subplot(grid_specs[0, 0])
    ax.plot(frequency_range, np.abs(spectral_components[selected_channel, :]), label=channel,
            linewidth=1, color="b")

    #########################################################
    ax.set_title(f"Spectrum (FFT), channel {channel}", fontsize=8)
    ax.set_xlabel("Frequency [Hz]", fontsize=8)
    ax.set_ylabel("Absolute amplitude", fontsize=8)
    ax.xaxis.set_tick_params(labelsize=8)
    ax.yaxis.set_tick_params(labelsize=6)

    #########################################################
    #ax = fig.add_subplot(grid_specs[0, 1])
    #ax.plot(frequency_range[:20], np.angle(spectral_components[selected_channel, :20])*180/np.pi, label=channel,
    #        linewidth=1, color="r")

    #########################################################
    #ax.set_title(f"Spectrum (FFT), channel {channel}", fontsize=8)
    #ax.set_xlabel("Frequency [Hz]", fontsize=8)
    #ax.set_ylabel("Phase [deg]", fontsize=8)
    #ax.xaxis.set_tick_params(labelsize=8)
    #ax.yaxis.set_tick_params(labelsize=6)

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight", pad_inches=0.2,
                    transparent=False, facecolor='white')
    plt.show()
