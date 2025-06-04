import pandas
import seaborn
import networkx
import numpy as np
import matplotlib
from pycirclize import Circos
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.gridspec import GridSpec
from scipy.interpolate import griddata
from . import settings


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
            horizontal_lines.append([x_coordinate, x_coordinate, "r"])
        if seizure_range[1] > start_time and seizure_range[1] < end_time:
            x_coordinate = int(seizure_range[1])
            horizontal_lines.append([x_coordinate, x_coordinate, "b"])

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
    fig = plt.figure(figsize=(5, 4))

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
    :param response: the frequency response, as complex numbers
    :param output_file: if specified the figure will be saved
    """
    grid_specs = GridSpec(2, 1, wspace=0.08, hspace=0.50)
    fig = plt.figure(figsize=(5.5, 4))

    #########################################################
    ax = fig.add_subplot(grid_specs[0, 0])
    ax.plot(frequency_range, 20 * np.log10(abs(response)), linewidth=1, color="r")
    ax.set_title("Filter Frequency Response", fontsize=8)
    ax.set_ylabel("Amplitude decrease [dB]", fontsize=8)
    ax.set_xlabel("Frequency [Hz]", fontsize=8)
    ax.xaxis.set_tick_params(labelsize=8)
    ax.yaxis.set_tick_params(labelsize=6)

    #########################################################
    ax = fig.add_subplot(grid_specs[1, 0])
    ax.plot(frequency_range, np.unwrap(np.angle(response)) * 180 / np.pi, linewidth=1, color="b")
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

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight", pad_inches=0.2,
                    transparent=False, facecolor='white')
    plt.show()


def plot_stationarity_bar_chart(stationarity_results: list, windows_lenghths: list,
                                output_file: str = None):
    """
    Plot the amount of stationary windows as per KPSS and ADF tests
    :param stationarity_results: results of White and KPSS tests
    :param windows_lenghts: window-lenghts for comparison
    :param output_file: if specified the figure will be saved
    """
    max_value = 0
    for result in stationarity_results:
        _max_value = np.max(result["count"])
        if _max_value > max_value:
            max_value = _max_value

    grid_specs = GridSpec(2, 2, wspace=0.17, hspace=0.2)
    fig = plt.figure(figsize=(5, 3.5))
    seaborn.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
    hfont = {"fontname": "Times New Roman"}

    #########################################################
    ax = fig.add_subplot(grid_specs[0, 0])
    seaborn.barplot(stationarity_results[0], x="result", y="count", ax=ax, palette="pastel")
    ax.yaxis.set_tick_params(labelsize=6)
    ax.axes.get_xaxis().set_ticks([])
    ax.set_ylabel("Count", fontsize=8)
    ax.set_xlabel("(a) 0.5s", fontsize=8, **hfont)
    ax.set_ylim([0, max_value])

    #########################################################
    ax = fig.add_subplot(grid_specs[0, 1])
    seaborn.barplot(stationarity_results[1], x="result", y="count", ax=ax, palette="pastel")
    ax.yaxis.set_tick_params(labelsize=6)
    ax.axes.get_xaxis().set_ticks([])
    ax.set_ylabel("")
    ax.set_xlabel("(b) 1s", fontsize=8, **hfont)
    ax.set_ylim([0, max_value])

    #########################################################
    ax = fig.add_subplot(grid_specs[1, 0])
    seaborn.barplot(stationarity_results[2], x="result", y="count", ax=ax, palette="pastel")
    ax.yaxis.set_tick_params(labelsize=6)
    ax.xaxis.set_tick_params(labelsize=6)
    ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_ylabel("Count", fontsize=8)
    ax.set_xlabel("(c) 3s", fontsize=8, **hfont)
    ax.set_ylim([0, max_value])

    #########################################################
    ax = fig.add_subplot(grid_specs[1, 1])
    seaborn.barplot(stationarity_results[3], x="result", y="count", ax=ax, palette="pastel")
    #ax.set_title(f"Windows length = {windows_lenghths[3]}s", fontsize=8)
    ax.yaxis.set_tick_params(labelsize=6)
    ax.xaxis.set_tick_params(labelsize=6)
    ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_ylabel("")
    ax.set_xlabel("(d) 5s", fontsize=8, **hfont)
    ax.set_ylim([0, max_value])

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight", pad_inches=0.2,
                    transparent=False, facecolor='white')
    plt.show()


def plot_univariate_intra_bar_chart(delta: pandas.DataFrame, theta: pandas.DataFrame,
                                    alpha: pandas.DataFrame, beta: pandas.DataFrame,
                                    gamma: pandas.DataFrame, all: pandas.DataFrame,
                                    feature: str, output_file: str = None):
    """
    Plot the eeg univariate features per band
    :param delta:
    :param theta:
    :param alpha:
    :param beta:
    :param all:
    :param output_file: if specified the figure will be saved
    """
    grid_specs = GridSpec(3, 2, wspace=0.15, hspace=0.20)
    fig = plt.figure(figsize=(5, 4))
    seaborn.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
    hfont = {"fontname": "Times New Roman"}
    features_map = {"hjorth_mobility": "mobility",
                    "hjorth_complexity": "complexity",
                    "katz_fractal_dimension": "katz fd",
                    "approximate_entropy": "entropy",
                    "power_spectral_density": "PSD"}
    feature = features_map.get(feature, feature)

    #########################################################
    ax = fig.add_subplot(grid_specs[0, 0])
    seaborn.barplot(delta, x="time_point", y="value", ax=ax, palette="pastel")
    ax.axes.get_xaxis().set_ticks([])
    ax.set_ylabel(f"Mean {feature}", fontsize=8)
    ax.set_xlabel("(a) delta", fontsize=8, **hfont)
    ax.yaxis.set_tick_params(labelsize=6)

    #########################################################
    ax = fig.add_subplot(grid_specs[0, 1])
    seaborn.barplot(theta, x="time_point", y="value", ax=ax, palette="pastel")
    ax.axes.get_xaxis().set_ticks([])
    ax.set_ylabel("")
    ax.set_xlabel("(b) theta", fontsize=8, **hfont)
    ax.yaxis.set_tick_params(labelsize=6)

    #########################################################
    ax = fig.add_subplot(grid_specs[1, 0])
    seaborn.barplot(alpha, x="time_point", y="value", ax=ax, palette="pastel")
    ax.set_ylabel(f"Mean {feature}", fontsize=8)
    ax.set_xlabel("(c) alpha", fontsize=8, **hfont)
    ax.axes.get_xaxis().set_ticks([])
    ax.yaxis.set_tick_params(labelsize=6)
    ax.xaxis.set_tick_params(labelsize=6)

    #########################################################
    ax = fig.add_subplot(grid_specs[1, 1])
    seaborn.barplot(beta, x="time_point", y="value", ax=ax, palette="pastel")
    ax.set_xlabel("(d) beta", fontsize=8, **hfont)
    ax.axes.get_xaxis().set_ticks([])
    ax.set_ylabel("")
    ax.xaxis.set_tick_params(labelsize=6)
    ax.yaxis.set_tick_params(labelsize=6)

    #########################################################
    ax = fig.add_subplot(grid_specs[2, 0])
    seaborn.barplot(gamma, x="time_point", y="value", ax=ax, palette="pastel")
    ax.set_ylabel(f"Mean {feature}", fontsize=8)
    ax.set_xlabel("(e) gamma", fontsize=8, **hfont)
    ax.xaxis.set_tick_params(labelsize=6)
    ax.yaxis.set_tick_params(labelsize=6)
    ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha="right")

    #########################################################
    ax = fig.add_subplot(grid_specs[2, 1])
    seaborn.barplot(all, x="time_point", y="value", ax=ax, palette="pastel")
    ax.set_xlabel("(f) Original", fontsize=8, **hfont)
    ax.set_ylabel("")
    ax.xaxis.set_tick_params(labelsize=6)
    ax.yaxis.set_tick_params(labelsize=6)
    ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha="right")

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight", pad_inches=0.2,
                    transparent=False, facecolor='white')
    plt.show()


def plot_univariate_inter_bar_chart(delta: pandas.DataFrame, theta: pandas.DataFrame,
                                    alpha: pandas.DataFrame, beta: pandas.DataFrame,
                                    gamma: pandas.DataFrame, all: pandas.DataFrame, 
                                    feature: str, output_file: str = None):
    """
    Plot the eeg univariate features per band
    :param delta:
    :param theta:
    :param alpha:
    :param beta:
    :param gamma:
    :param all:
    :param feature: name of the feature
    :param output_file: if specified the figure will be saved
    """
    grid_specs = GridSpec(3, 2, wspace=0.15, hspace=0.20)
    fig = plt.figure(figsize=(7, 6))

    #########################################################
    ax = fig.add_subplot(grid_specs[0, 0])
    seaborn.barplot(delta, x="Group", y="value", ax=ax, palette="pastel")
    ax.axes.get_xaxis().set_ticks([])
    ax.set_ylabel(f"Mean {feature}", fontsize=8)
    ax.set_xlabel("a) delta", fontsize=8)
    ax.yaxis.set_tick_params(labelsize=8)

    #########################################################
    ax = fig.add_subplot(grid_specs[0, 1])
    seaborn.barplot(theta, x="Group", y="value", ax=ax, palette="pastel")
    ax.axes.get_xaxis().set_ticks([])
    ax.set_ylabel("")
    ax.set_xlabel("b) theta", fontsize=8)
    ax.yaxis.set_tick_params(labelsize=8)

    #########################################################
    ax = fig.add_subplot(grid_specs[1, 0])
    seaborn.barplot(alpha, x="Group", y="value", ax=ax, palette="pastel")
    ax.axes.get_xaxis().set_ticks([])
    ax.set_ylabel(f"Mean {feature}", fontsize=8)
    ax.set_xlabel("c) alpha", fontsize=8)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.xaxis.set_tick_params(labelsize=8)

    #########################################################
    ax = fig.add_subplot(grid_specs[1, 1])
    seaborn.barplot(beta, x="Group", y="value", ax=ax, palette="pastel")
    ax.axes.get_xaxis().set_ticks([])
    ax.set_xlabel("d) beta", fontsize=8)
    ax.set_ylabel("")
    ax.xaxis.set_tick_params(labelsize=8)
    ax.yaxis.set_tick_params(labelsize=8)

    #########################################################
    ax = fig.add_subplot(grid_specs[2, 0])
    seaborn.barplot(gamma, x="Group", y="value", ax=ax, palette="pastel")
    ax.set_xlabel("Group \n\n e) gamma", fontsize=8)
    ax.set_ylabel(f"Mean {feature}", fontsize=8)
    ax.xaxis.set_tick_params(labelsize=8)
    ax.yaxis.set_tick_params(labelsize=8)

    #########################################################
    ax = fig.add_subplot(grid_specs[2, 1])
    seaborn.barplot(all, x="Group", y="value", ax=ax, palette="pastel")
    ax.set_xlabel("Group \n\n f) Original", fontsize=8)
    ax.set_ylabel("")
    ax.xaxis.set_tick_params(labelsize=8)
    ax.yaxis.set_tick_params(labelsize=8)

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight", pad_inches=0.2,
                    transparent=False, facecolor='white')
    plt.show()


def plot_univariate_intra_dist_chart(delta: pandas.DataFrame, theta: pandas.DataFrame,
                                     alpha: pandas.DataFrame, beta: pandas.DataFrame,
                                     gamma: pandas.DataFrame, all: pandas.DataFrame,
                                     feature: str, output_file: str = None):
    """
    Plot the eeg univariate features per band
    :param delta:
    :param theta:
    :param alpha:
    :param beta:
    :param gamma:
    :param all:
    :param feature: name of the feature
    :param output_file: if specified the figure will be saved
    """
    grid_specs = GridSpec(3, 2, wspace=0.15, hspace=0.30)
    fig = plt.figure(figsize=(7, 7))

    #########################################################
    ax = fig.add_subplot(grid_specs[0, 0])
    seaborn.kdeplot(delta, x="value", hue="Time point", ax=ax, palette="pastel")
    ax.set_ylabel("Density", fontsize=8)
    ax.set_xlabel("a) delta", fontsize=8)
    ax.get_legend().remove()
    ax.yaxis.set_tick_params(labelsize=8)
    ax.xaxis.set_tick_params(labelsize=8)

    #########################################################
    ax = fig.add_subplot(grid_specs[0, 1])
    seaborn.kdeplot(theta, x="value", hue="Time point", ax=ax, palette="pastel")
    ax.set_ylabel("")
    ax.set_xlabel("b) theta", fontsize=8)
    ax.get_legend().remove()
    ax.yaxis.set_tick_params(labelsize=8)
    ax.xaxis.set_tick_params(labelsize=8)

    #########################################################
    ax = fig.add_subplot(grid_specs[1, 0])
    seaborn.kdeplot(alpha, x="value", hue="Time point", ax=ax, palette="pastel")
    ax.set_ylabel("Density", fontsize=8)
    ax.set_xlabel("c) alpha", fontsize=8)
    ax.get_legend().remove()
    ax.yaxis.set_tick_params(labelsize=8)
    ax.xaxis.set_tick_params(labelsize=8)

    #########################################################
    ax = fig.add_subplot(grid_specs[1, 1])
    seaborn.kdeplot(beta, x="value", hue="Time point", ax=ax, palette="pastel")
    ax.set_xlabel("d) beta", fontsize=8)
    ax.get_legend().remove()
    ax.set_ylabel("")
    ax.xaxis.set_tick_params(labelsize=8)
    ax.yaxis.set_tick_params(labelsize=8)

    #########################################################
    ax = fig.add_subplot(grid_specs[2, 0])
    seaborn.kdeplot(gamma, x="value", hue="Time point", ax=ax, palette="pastel")
    seaborn.move_legend(ax, "upper right", ncol=2)
    ax.set_ylabel("Density", fontsize=8)
    ax.set_xlabel("Feature value \n\n e) gamma", fontsize=8)
    ax.get_legend().remove()
    ax.xaxis.set_tick_params(labelsize=8)
    ax.yaxis.set_tick_params(labelsize=8)

    #########################################################
    ax = fig.add_subplot(grid_specs[2, 1])
    seaborn.kdeplot(all, x="value", hue="Time point", ax=ax, palette="pastel")
    seaborn.move_legend(ax, "upper right", ncol=2)
    ax.set_ylabel("")
    ax.set_xlabel("Feature value \n\n f) Original", fontsize=8)
    ax.xaxis.set_tick_params(labelsize=8)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.get_legend().set_title(None)
    texts = ax.get_legend().get_texts()
    for t in texts:
        t.set_size('x-small')

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight", pad_inches=0.2,
                    transparent=False, facecolor='white')
    plt.show()


def plot_univariate_inter_dist_chart(delta: pandas.DataFrame, theta: pandas.DataFrame,
                                     alpha: pandas.DataFrame, beta: pandas.DataFrame,
                                     gamma: pandas.DataFrame, all: pandas.DataFrame,
                                     feature: str, output_file: str = None):
    """
    Plot the eeg univariate features per band
    :param delta:
    :param theta:
    :param alpha:
    :param beta:
    :param gamma:
    :param all:
    :param feature: name of the feature
    :param output_file: if specified the figure will be saved
    """
    grid_specs = GridSpec(3, 2, wspace=0.15, hspace=0.30)
    fig = plt.figure(figsize=(5, 5.5))
    seaborn.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
    hfont = {'fontname':'Times New Roman'}

    #########################################################
    ax = fig.add_subplot(grid_specs[0, 0])
    seaborn.violinplot(delta, x="value", y="Group", ax=ax, palette="pastel",
                       inner_kws=dict(box_width=5, whis_width=2, color="0.1"))
    ax.set_ylabel(f"{feature}".capitalize().replace("_", " "), fontsize=8)
    ax.set_xlabel("(a) delta", fontsize=8, **hfont)
    ax.xaxis.set_tick_params(labelsize=6)
    ax.yaxis.set_tick_params(labelsize=6)

    #########################################################
    ax = fig.add_subplot(grid_specs[0, 1])
    seaborn.violinplot(theta, x="value", y="Group", ax=ax, palette="pastel",
                       inner_kws=dict(box_width=5, whis_width=2, color="0.1"))
    ax.axes.get_yaxis().set_ticks([])
    ax.set_ylabel("")
    ax.set_xlabel("(b) theta", fontsize=8, **hfont)
    ax.xaxis.set_tick_params(labelsize=6)
    ax.yaxis.set_tick_params(labelsize=6)

    #########################################################
    ax = fig.add_subplot(grid_specs[1, 0])
    seaborn.violinplot(alpha, x="value", y="Group", ax=ax, palette="pastel",
                       inner_kws=dict(box_width=5, whis_width=2, color="0.1"))
    ax.set_ylabel(f"{feature}".capitalize().replace("_", " "), fontsize=8)
    ax.set_xlabel("(c) alpha", fontsize=8, **hfont)
    ax.yaxis.set_tick_params(labelsize=6)
    ax.xaxis.set_tick_params(labelsize=6)

    #########################################################
    ax = fig.add_subplot(grid_specs[1, 1])
    seaborn.violinplot(beta, x="value", y="Group", ax=ax, palette="pastel",
                       inner_kws=dict(box_width=5, whis_width=2, color="0.1"))
    ax.axes.get_yaxis().set_ticks([])
    ax.set_xlabel("(d) beta", fontsize=8, **hfont)
    ax.set_ylabel("")
    ax.xaxis.set_tick_params(labelsize=6)
    ax.yaxis.set_tick_params(labelsize=6)

    #########################################################
    ax = fig.add_subplot(grid_specs[2, 0])
    seaborn.violinplot(gamma, x="value", y="Group", ax=ax, palette="pastel",
                       inner_kws=dict(box_width=5, whis_width=2, color="0.1"))
    ax.set_ylabel(f"{feature}".capitalize().replace("_", " "), fontsize=8)
    ax.set_xlabel("Feature value \n\n (e) gamma", fontsize=8, **hfont)
    ax.xaxis.set_tick_params(labelsize=6)
    ax.yaxis.set_tick_params(labelsize=6)

    #########################################################
    ax = fig.add_subplot(grid_specs[2, 1])
    seaborn.violinplot(all, x="value", y="Group", ax=ax, palette="pastel",
                       inner_kws=dict(box_width=5, whis_width=2, color="0.1"))
    ax.axes.get_yaxis().set_ticks([])
    ax.set_xlabel("Feature value \n\n (f) Original", fontsize=8, **hfont)
    ax.set_ylabel("")
    ax.xaxis.set_tick_params(labelsize=6)
    ax.yaxis.set_tick_params(labelsize=6)
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight", pad_inches=0.2,
                    transparent=False, facecolor='white')
    plt.show()


def plot_topoplot_features_time(stage_1: np.array, stage_2: np.array,
                                stage_3: np.array, stage_4: np.array,
                                stage_5: np.array, stage_6: np.array,
                                stage_7: np.array, stage_8: np.array,
                                stage_9: np.array, stage_10: np.array,
                                stage_11: np.array,
                                output_file: str = None, is_monopolar: bool = True):
    """
    Plot the topographic map per stage
    :param stage [1-8]: each stage stands for an eeg window
    :param output_file: if specified the figure will be saved
    """
    hfont = {"fontname": "Times New Roman"}

    if is_monopolar:
        electrode_positions = settings["electrode_positions"]
        direction = "horizontal"
    else:
        electrode_positions = settings["bipolar_electrode_positions"]
        direction = "vertical"

    grid_specs = GridSpec(3, 4, wspace=0.20, hspace=0.10)
    fig = plt.figure(figsize=(8, 6))
    xi, yi = np.mgrid[-1:1:100j, -1:1:100j]

    def single_topomap(ax_object, data_array: np.array, min_color: int, max_color: int):
        ax_object.axis((-1.2, 1.2, -1.2, 1.2))
        circle = Circle([0, 0], radius=1, fill=False)
        ax_object.add_patch(circle)

        for electrode, coordinate in electrode_positions.items():
            circle = Circle(coordinate, radius=0.04, fill=True, facecolor=(1, 1, 1))
            ax_object.add_patch(circle)
            ax_object.text(coordinate[0], coordinate[1], electrode,
                           verticalalignment='center',
                           horizontalalignment='center',
                           rotation=direction,
                           size=6)
        points = []
        for channel in data_array.channel.tolist():
            point = [x * 1.2 for x in electrode_positions[channel]]
            points.append(point)

        zi = griddata(points, data_array.value, (xi, yi), method="cubic")
        colormap = plt.cm.jet
        normalize = matplotlib.colors.Normalize(vmin=min_color, vmax=max_color)
        ax_object.contourf(xi, yi, zi, 10, cmap=colormap, norm=normalize)
        ax_object.axes.get_xaxis().set_ticks([])
        ax_object.axes.get_yaxis().set_ticks([])
        ax_object.spines['top'].set_visible(False)
        ax_object.spines['right'].set_visible(False)
        ax_object.spines['bottom'].set_visible(False)
        ax_object.spines['left'].set_visible(False)

    #########################################################
    temporal_array = np.concatenate([stage_1[1].value, stage_2[1].value, stage_3[1].value, stage_4[1].value,
                                     stage_5[1].value, stage_6[1].value, stage_7[1].value, stage_8[1].value,
                                     stage_9[1].value, stage_10[1].value, stage_11[1].value])
    min_color = np.min(temporal_array)
    max_color = np.max(temporal_array)

    ax = fig.add_subplot(grid_specs[0, 0])
    single_topomap(ax, stage_1[1], min_color, max_color)
    ax.set_xlabel(f"a) {stage_1[0]}", fontsize=8, **hfont)

    #########################################################
    ax = fig.add_subplot(grid_specs[0, 1])
    single_topomap(ax, stage_2[1], min_color, max_color)
    ax.set_xlabel(f"b) {stage_2[0]}", fontsize=8, **hfont)

    #########################################################
    ax = fig.add_subplot(grid_specs[0, 2])
    single_topomap(ax, stage_3[1], min_color, max_color)
    ax.set_xlabel(f"c) {stage_3[0]}", fontsize=8, **hfont)

    #########################################################
    ax = fig.add_subplot(grid_specs[0, 3])
    single_topomap(ax, stage_4[1], min_color, max_color)
    ax.set_xlabel(f"d) {stage_4[0]}", fontsize=8, **hfont)

    #########################################################
    ax = fig.add_subplot(grid_specs[1, 0])
    single_topomap(ax, stage_5[1], min_color, max_color)
    ax.set_xlabel(f"e) {stage_5[0]}", fontsize=8, **hfont)

    #########################################################
    ax = fig.add_subplot(grid_specs[1, 1])
    single_topomap(ax, stage_6[1], min_color, max_color)
    ax.set_xlabel(f"f) {stage_6[0]}", fontsize=8, **hfont)

    #########################################################
    ax = fig.add_subplot(grid_specs[1, 2])
    single_topomap(ax, stage_7[1], min_color, max_color)
    ax.set_xlabel(f"g) {stage_7[0]}", fontsize=8, **hfont)

    #########################################################
    ax = fig.add_subplot(grid_specs[1, 3])
    single_topomap(ax, stage_11[1], min_color, max_color)
    ax.set_xlabel(f"h) {stage_11[0]}", fontsize=8, **hfont)

    #########################################################
    ax = fig.add_subplot(grid_specs[2, 0])
    single_topomap(ax, stage_10[1], min_color, max_color)
    ax.set_xlabel(f"i) {stage_10[0]}", fontsize=8, **hfont)

    #########################################################
    ax = fig.add_subplot(grid_specs[2, 1])
    single_topomap(ax, stage_9[1], min_color, max_color)
    ax.set_xlabel(f"j) {stage_9[0]}", fontsize=8, **hfont)

    #########################################################
    ax = fig.add_subplot(grid_specs[2, 2])
    single_topomap(ax, stage_8[1], min_color, max_color)
    ax.set_xlabel(f"k) {stage_8[0]}", fontsize=8, **hfont)

    norm = matplotlib.colors.Normalize(min_color, max_color)
    ax = fig.add_subplot(grid_specs[2, 3])
    ax.figure.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=plt.cm.jet),
                       ax=ax, pad=0.2)
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight", pad_inches=0.2,
                    transparent=False, facecolor='white')

    plt.show()


def plot_univariate_inter_bar_chart_psd(delta: pandas.DataFrame, theta: pandas.DataFrame,
                                        alpha: pandas.DataFrame, beta: pandas.DataFrame,
                                        gamma: pandas.DataFrame, _: pandas.DataFrame,
                                        feature: str, output_file: str = None):
    """
    Plot the eeg univariate features healthy subject vs patients (per band)
    :param delta:
    :param theta:
    :param alpha:
    :param beta:
    :param gamma:
    :param feature: name of the feature
    :param output_file: if specified the figure will be saved
    """
    grid_specs = GridSpec(3, 2, wspace=0.15, hspace=0.20)
    fig = plt.figure(figsize=(7, 6))
    features_map = {"hjorth_mobility": "mobility",
                    "hjorth_complexity": "complexity",
                    "katz_fractal_dimension": "katz fd",
                    "approximate_entropy": "entropy",
                    "power_spectral_density": "PSD"}
    feature = features_map.get(feature, feature)

    #########################################################
    ax = fig.add_subplot(grid_specs[0, 0])
    seaborn.barplot(delta, x="Group", y="value", ax=ax, palette="pastel")
    ax.axes.get_xaxis().set_ticks([])
    ax.set_ylabel(f"Mean {feature}", fontsize=8)
    ax.set_xlabel("a) delta", fontsize=8)
    ax.yaxis.set_tick_params(labelsize=8)

    #########################################################
    ax = fig.add_subplot(grid_specs[0, 1])
    seaborn.barplot(theta, x="Group", y="value", ax=ax, palette="pastel")
    ax.axes.get_xaxis().set_ticks([])
    ax.set_ylabel("")
    ax.set_xlabel("b) theta", fontsize=8)
    ax.yaxis.set_tick_params(labelsize=8)

    #########################################################
    ax = fig.add_subplot(grid_specs[1, 0])
    seaborn.barplot(alpha, x="Group", y="value", ax=ax, palette="pastel")
    ax.axes.get_xaxis().set_ticks([])
    ax.set_ylabel(f"Mean {feature}", fontsize=8)
    ax.set_xlabel("c) alpha", fontsize=8)
    ax.yaxis.set_tick_params(labelsize=8)

    #########################################################
    ax = fig.add_subplot(grid_specs[1, 1])
    seaborn.barplot(beta, x="Group", y="value", ax=ax, palette="pastel")
    ax.axes.get_xaxis().set_ticks([])
    ax.set_ylabel("")
    ax.set_xlabel("d) beta", fontsize=8)
    ax.xaxis.set_tick_params(labelsize=8)

    #########################################################
    ax = fig.add_subplot(grid_specs[2, :])
    seaborn.barplot(gamma, x="Group", y="value", ax=ax, palette="pastel")
    ax.set_ylabel(f"Mean {feature}", fontsize=8)
    ax.set_xlabel("Group \n\n e) gamma", fontsize=8)
    ax.xaxis.set_tick_params(labelsize=8)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha="right")

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight", pad_inches=0.2,
                    transparent=False, facecolor='white')
    plt.show()


def plot_univariate_inter_dist_chart_psd(delta: pandas.DataFrame, theta: pandas.DataFrame,
                                         alpha: pandas.DataFrame, beta: pandas.DataFrame,
                                         gamma: pandas.DataFrame, _: pandas.DataFrame,
                                         feature: str, output_file: str = None):
    """
    Plot the eeg univariate features healthy subject vs patients (per band)
    :param delta:
    :param theta:
    :param alpha:
    :param beta:
    :param gamma:
    :param feature: name of the feature
    :param output_file: if specified the figure will be saved
    """
    grid_specs = GridSpec(3, 2, wspace=0.15, hspace=0.30)
    fig = plt.figure(figsize=(5, 5.5))
    seaborn.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
    hfont = {"fontname": "Times New Roman"}
    features_map = {"hjorth_mobility": "mobility",
                    "hjorth_complexity": "complexity",
                    "katz_fractal_dimension": "katz fd",
                    "approximate_entropy": "entropy",
                    "power_spectral_density": "PSD"}
    feature = features_map.get(feature, feature)

    #########################################################
    ax = fig.add_subplot(grid_specs[0, 0])
    seaborn.violinplot(delta, x="value", y="Group", ax=ax, palette="pastel",
                       inner_kws=dict(box_width=5, whis_width=2, color="0.1"))
    ax.set_ylabel(feature, fontsize=8)
    ax.set_xlabel("a) delta", fontsize=8, **hfont)
    ax.yaxis.set_tick_params(labelsize=6)
    ax.xaxis.set_tick_params(labelsize=6)

    #########################################################
    ax = fig.add_subplot(grid_specs[0, 1])
    seaborn.violinplot(theta, x="value", y="Group", ax=ax, palette="pastel",
                       inner_kws=dict(box_width=5, whis_width=2, color="0.1"))
    ax.set_ylabel("")
    ax.set_xlabel("b) theta", fontsize=8, **hfont)
    ax.axes.get_yaxis().set_ticks([])
    ax.yaxis.set_tick_params(labelsize=6)
    ax.xaxis.set_tick_params(labelsize=6)

    #########################################################
    ax = fig.add_subplot(grid_specs[1, 0])
    seaborn.violinplot(alpha, x="value", y="Group", ax=ax, palette="pastel",
                       inner_kws=dict(box_width=5, whis_width=2, color="0.1"))
    ax.set_ylabel(feature, fontsize=8)
    ax.set_xlabel("c) alpha", fontsize=8, **hfont)
    ax.yaxis.set_tick_params(labelsize=6)
    ax.xaxis.set_tick_params(labelsize=6)

    #########################################################
    ax = fig.add_subplot(grid_specs[1, 1])
    seaborn.violinplot(beta, x="value", y="Group", ax=ax, palette="pastel",
                       inner_kws=dict(box_width=5, whis_width=2, color="0.1"))
    ax.set_ylabel("")
    ax.set_xlabel("d) beta", fontsize=8, **hfont)
    ax.axes.get_yaxis().set_ticks([])
    ax.xaxis.set_tick_params(labelsize=6)
    ax.yaxis.set_tick_params(labelsize=6)

    #########################################################
    ax = fig.add_subplot(grid_specs[2, :])
    seaborn.violinplot(gamma, x="value", y="Group", ax=ax, palette="pastel",
                       inner_kws=dict(box_width=5, whis_width=2, color="0.1"))
    ax.set_ylabel(feature, fontsize=8)
    ax.set_xlabel("Feature value \n\n e) gamma", fontsize=8, **hfont)
    ax.xaxis.set_tick_params(labelsize=6)
    ax.yaxis.set_tick_params(labelsize=6)

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight", pad_inches=0.2,
                    transparent=False, facecolor='white')
    plt.show()


def plot_univariate_intra_bar_chart_psd(delta: pandas.DataFrame, theta: pandas.DataFrame,
                                        alpha: pandas.DataFrame, beta: pandas.DataFrame,
                                        gamma: pandas.DataFrame, feature: str,
                                        output_file: str = None):
    """
    Plot the PSD values per band
    :param delta:
    :param theta:
    :param alpha:
    :param beta:
    :param gamma:
    ;param feature: name of the feature
    :param output_file: if specified the figure will be saved
    """
    grid_specs = GridSpec(3, 2, wspace=0.15, hspace=0.20)
    fig = plt.figure(figsize=(5, 4))
    seaborn.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
    hfont = {"fontname": "Times New Roman"}
    features_map = {"hjorth_mobility": "mobility",
                    "hjorth_complexity": "complexity",
                    "katz_fractal_dimension": "katz fd",
                    "approximate_entropy": "entropy",
                    "power_spectral_density": "PSD"}
    feature = features_map.get(feature, feature)

    #########################################################
    ax = fig.add_subplot(grid_specs[0, 0])
    seaborn.barplot(delta, x="time_point", y="value", ax=ax, palette="pastel")
    ax.axes.get_xaxis().set_ticks([])
    ax.set_ylabel(f"Mean {feature}", fontsize=8)
    ax.set_xlabel("(a) delta", fontsize=8, **hfont)
    ax.yaxis.set_tick_params(labelsize=6)
    ax.set_ylim([0, 0.7])

    #########################################################
    ax = fig.add_subplot(grid_specs[0, 1])
    seaborn.barplot(theta, x="time_point", y="value", ax=ax, palette="pastel")
    ax.axes.get_xaxis().set_ticks([])
    ax.set_ylabel("")
    ax.set_xlabel("(b) theta", fontsize=8, **hfont)
    ax.yaxis.set_tick_params(labelsize=6)
    ax.set_ylim([0, 0.7])

    #########################################################
    ax = fig.add_subplot(grid_specs[1, 0])
    seaborn.barplot(alpha, x="time_point", y="value", ax=ax, palette="pastel")
    ax.axes.get_xaxis().set_ticks([])
    ax.set_ylabel(f"Mean {feature}", fontsize=8)
    ax.set_xlabel("(c) alpha", fontsize=8, **hfont)
    ax.yaxis.set_tick_params(labelsize=6)
    ax.set_ylim([0, 0.7])

    #########################################################
    ax = fig.add_subplot(grid_specs[1, 1])
    seaborn.barplot(beta, x="time_point", y="value", ax=ax, palette="pastel")
    ax.axes.get_xaxis().set_ticks([])
    ax.set_xlabel("(d) beta", fontsize=8, **hfont)
    ax.set_ylabel("")
    ax.yaxis.set_tick_params(labelsize=6)
    ax.set_ylim([0, 0.7])

    #########################################################
    ax = fig.add_subplot(grid_specs[2, :])
    seaborn.barplot(gamma, x="time_point", y="value", ax=ax, palette="pastel")
    ax.set_ylabel(f"Mean {feature}", fontsize=8)
    ax.set_xlabel("(e) gamma", fontsize=8, **hfont)
    ax.xaxis.set_tick_params(labelsize=6)
    ax.yaxis.set_tick_params(labelsize=6)
    ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_ylim([0, 0.7])

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight", pad_inches=0.2,
                    transparent=False, facecolor='white')
    plt.show()


def plot_univariate_intra_dist_chart_psd(delta: pandas.DataFrame, theta: pandas.DataFrame,
                                         alpha: pandas.DataFrame, beta: pandas.DataFrame,
                                         gamma: pandas.DataFrame, feature: str,
                                         output_file: str = None):
    """
    Plot the PDS values per band
    :param delta:
    :param theta:
    :param alpha:
    :param beta:
    :param gamma:
    :param feature: name of the feature
    :param output_file: if specified the figure will be saved
    """
    grid_specs = GridSpec(3, 2, wspace=0.15, hspace=0.20)
    fig = plt.figure(figsize=(7, 7))

    #########################################################
    ax = fig.add_subplot(grid_specs[0, 0])
    seaborn.kdeplot(delta, x="value", hue="Time point", ax=ax, palette="pastel")
    ax.axes.get_xaxis().set_ticks([])
    ax.set_ylabel("Density", fontsize=8)
    ax.set_xlabel("a) delta", fontsize=8)
    ax.get_legend().remove()
    ax.yaxis.set_tick_params(labelsize=8)

    #########################################################
    ax = fig.add_subplot(grid_specs[0, 1])
    seaborn.kdeplot(theta, x="value", hue="Time point", ax=ax, palette="pastel")
    ax.axes.get_xaxis().set_ticks([])
    ax.set_ylabel("")
    ax.set_xlabel("b) theta", fontsize=8)
    ax.get_legend().remove()
    ax.yaxis.set_tick_params(labelsize=8)

    #########################################################
    ax = fig.add_subplot(grid_specs[1, 0])
    seaborn.kdeplot(alpha, x="value", hue="Time point", ax=ax, palette="pastel")
    ax.axes.get_xaxis().set_ticks([])
    ax.set_ylabel("Density", fontsize=8)
    ax.set_xlabel("c) alpha", fontsize=8)
    ax.get_legend().remove()
    ax.yaxis.set_tick_params(labelsize=8)
    ax.xaxis.set_tick_params(labelsize=8)

    #########################################################
    ax = fig.add_subplot(grid_specs[1, 1])
    seaborn.kdeplot(beta, x="value", hue="Time point", ax=ax, palette="pastel")
    ax.axes.get_xaxis().set_ticks([])
    ax.set_ylabel("")
    ax.set_xlabel("d) beta", fontsize=8)
    ax.get_legend().remove()
    ax.yaxis.set_tick_params(labelsize=8)

    #########################################################
    ax = fig.add_subplot(grid_specs[2, :])
    seaborn.kdeplot(gamma, x="value", hue="Time point", ax=ax, palette="pastel")
    seaborn.move_legend(ax, "lower right", ncol=2)
    ax.set_xlabel("Feature value \n\n e) gamma", fontsize=8)
    ax.set_ylabel("")
    ax.xaxis.set_tick_params(labelsize=8)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.get_legend().set_title(None)
    texts = ax.get_legend().get_texts()
    for t in texts:
        t.set_size('xx-small')

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight", pad_inches=0.2,
                    transparent=False, facecolor='white')
    plt.show()


def plot_network_features_time(stage_1: np.array, stage_2: np.array,
                               stage_3: np.array, stage_4: np.array,
                               stage_5: np.array, stage_6: np.array,
                               stage_7: np.array, stage_8: np.array,
                               stage_9: np.array, stage_10: np.array,
                               stage_11: np.array, line_widths: list,
                               output_file: str = None, is_monopolar: bool = True):
    """
    Plot the network connectivity per stage
    :param stage [1-8]: each stage stands for an eeg window
    :param line_widths: weight for each connection
    :param output_file: if specified the figure will be saved
    :param is_monopolar: type of channel, monopolar or bipolar
    """
    if is_monopolar:
        electrode_positions = settings["electrode_positions"]
    else:
        electrode_positions = settings["bipolar_electrode_positions"]

    grid_specs = GridSpec(3, 4, wspace=0.20, hspace=0.10)
    fig = plt.figure(figsize=(12, 7))

    def single_network_map(ax_object, network_graph: np.array):
        positions = {}

        for electrode, coordinate in electrode_positions.items():
            network_graph.add_node(electrode)
            positions.update({electrode: tuple(coordinate)})

        networkx.draw_networkx_nodes(network_graph, ax=ax_object, pos=positions, node_size=100)
        networkx.draw_networkx_edges(network_graph, pos=positions, width=line_widths)
        ax_object.spines['top'].set_visible(False)
        ax_object.spines['right'].set_visible(False)
        ax_object.spines['bottom'].set_visible(False)
        ax_object.spines['left'].set_visible(False)

    #########################################################
    ax = fig.add_subplot(grid_specs[0, 0])
    single_network_map(ax, stage_1[1])
    ax.set_xlabel(f"a) {stage_1[0]}", fontsize=8)

    #########################################################
    ax = fig.add_subplot(grid_specs[0, 1])
    single_network_map(ax, stage_2[1])
    ax.set_xlabel(f"b) {stage_2[0]}", fontsize=8)

    #########################################################
    ax = fig.add_subplot(grid_specs[0, 2])
    single_network_map(ax, stage_3[1])
    ax.set_xlabel(f"c) {stage_3[0]}", fontsize=8)

    #########################################################
    ax = fig.add_subplot(grid_specs[0, 3])
    single_network_map(ax, stage_4[1])
    ax.set_xlabel(f"d) {stage_4[0]}", fontsize=8)

    #########################################################
    ax = fig.add_subplot(grid_specs[1, 0])
    single_network_map(ax, stage_5[1])
    ax.set_xlabel(f"e) {stage_5[0]}", fontsize=8)

    #########################################################
    ax = fig.add_subplot(grid_specs[1, 1])
    single_network_map(ax, stage_6[1])
    ax.set_xlabel(f"f) {stage_6[0]}", fontsize=8)

    #########################################################
    ax = fig.add_subplot(grid_specs[1, 2])
    single_network_map(ax, stage_7[1])
    ax.set_xlabel(f"g) {stage_7[0]}", fontsize=8)

    #########################################################
    ax = fig.add_subplot(grid_specs[1, 3])
    single_network_map(ax, stage_11[1])
    ax.set_xlabel(f"h) {stage_11[0]}", fontsize=8)

    #########################################################
    ax = fig.add_subplot(grid_specs[2, 0])
    single_network_map(ax, stage_10[1])
    ax.set_xlabel(f"i) {stage_10[0]}", fontsize=8)

    #########################################################
    ax = fig.add_subplot(grid_specs[2, 1])
    single_network_map(ax, stage_9[1])
    ax.set_xlabel(f"j) {stage_9[0]}", fontsize=8)

    #########################################################
    ax = fig.add_subplot(grid_specs[2, 2])
    single_network_map(ax, stage_8[1])
    ax.set_xlabel(f"k) {stage_8[0]}", fontsize=8)

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight", pad_inches=0.2,
                    transparent=False, facecolor='white')

    plt.show()


def plot_graph_striplot_chart(delta: pandas.DataFrame, theta: pandas.DataFrame,
                              alpha: pandas.DataFrame, beta: pandas.DataFrame,
                              output_file: str = None):
    """
    Plot the global efficiency values per band
    :param delta:
    :param theta:
    :param alpha:
    :param beta:
    :param output_file: if specified the figure will be saved
    """
    grid_specs = GridSpec(2, 2, wspace=0.12, hspace=0.17)
    fig = plt.figure(figsize=(7, 5))
    order = ["61s before", "31s before", "11s before", "1s before", "Start of seizure",
             "Middle point", "End of seizure", "1s after", "11s after", "31s after", "61s after"]

    #########################################################
    ax = fig.add_subplot(grid_specs[0, 0])
    seaborn.stripplot(delta, x="value", y="Time point", ax=ax, palette="pastel", order=order, dodge=True,
                      alpha=.25, zorder=1)
    seaborn.pointplot(delta, x="value", y="Time point", ax=ax, dodge=0.8 - 0.8 / 3, palette="dark",
                      errorbar=None, markers="d", markersize=4, linestyle="none", order=order)
    ax.axes.get_xaxis().set_ticks([])
    ax.set_xlabel("a) delta", fontsize=8)
    ax.set_ylabel("")
    ax.yaxis.set_tick_params(labelsize=8)

    #########################################################
    ax = fig.add_subplot(grid_specs[0, 1])
    seaborn.stripplot(theta, x="value", y="Time point", ax=ax, palette="pastel", order=order, dodge=True,
                      alpha=.25, zorder=1)
    seaborn.pointplot(theta, x="value", y="Time point", ax=ax, dodge=0.8 - 0.8 / 3, palette="dark",
                      errorbar=None, markers="d", markersize=4, linestyle="none", order=order)
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    ax.set_xlabel("b) theta", fontsize=8)
    ax.set_ylabel("")
    ax.yaxis.set_tick_params(labelsize=8)

    #########################################################
    ax = fig.add_subplot(grid_specs[1, 0])
    seaborn.stripplot(alpha, x="value", y="Time point", ax=ax, palette="pastel", order=order, dodge=True,
                      alpha=.25, zorder=1)
    seaborn.pointplot(alpha, x="value", y="Time point", ax=ax, dodge=0.8 - 0.8 / 3, palette="dark",
                      errorbar=None, markers="d", markersize=4, linestyle="none", order=order)
    ax.set_xlabel("Feature value \n\n c) alpha", fontsize=8)
    ax.set_ylabel("")
    ax.yaxis.set_tick_params(labelsize=8)
    ax.xaxis.set_tick_params(labelsize=8)

    #########################################################
    ax = fig.add_subplot(grid_specs[1, 1])
    seaborn.stripplot(beta, x="value", y="Time point", ax=ax, palette="pastel", order=order, dodge=True,
                      alpha=.25, zorder=1)
    seaborn.pointplot(beta, x="value", y="Time point", ax=ax, dodge=0.8 - 0.8 / 3, palette="dark",
                      errorbar=None, markers="d", markersize=4, linestyle="none", order=order)
    ax.set_xlabel("Feature value \n\n d) beta", fontsize=8)
    ax.set_ylabel("")
    ax.axes.get_yaxis().set_ticks([])
    ax.xaxis.set_tick_params(labelsize=8)
    ax.yaxis.set_tick_params(labelsize=8)

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight", pad_inches=0.2,
                    transparent=False, facecolor='white')
    plt.show()


def plot_graph_pointplot_chart(delta: pandas.DataFrame, theta: pandas.DataFrame,
                               alpha: pandas.DataFrame, beta: pandas.DataFrame,
                               ylim: list, output_file: str = None):
    """
    Plot the global efficiency values per band
    :param delta:
    :param theta:
    :param alpha:
    :param beta:
    :param output_file: if specified the figure will be saved
    """
    grid_specs = GridSpec(2, 2, wspace=0.12, hspace=0.17)
    fig = plt.figure(figsize=(7, 5))
    order = ["61s before", "31s before", "11s before", "1s before", "Start of seizure",
             "Middle point", "End of seizure", "1s after", "11s after", "31s after", "61s after"]

    #########################################################
    ax = fig.add_subplot(grid_specs[0, 0])
    seaborn.pointplot(delta, x="Time point", y="value", hue="seizure_number", ax=ax, palette="dark",
                      errorbar=None, markers="s", markersize=4, order=order, legend=False)
    ax.axvline(x=4)
    ax.axes.get_xaxis().set_ticks([])
    ax.set_xlabel("a) delta", fontsize=8)
    ax.set_ylabel("")
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_ylim(ylim)

    #########################################################
    ax = fig.add_subplot(grid_specs[0, 1])
    seaborn.pointplot(theta, x="Time point", y="value", hue="seizure_number", ax=ax, palette="dark",
                      errorbar=None, markers="s", markersize=4, order=order, legend=False)
    ax.axvline(x=4)
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    ax.set_xlabel("b) theta", fontsize=8)
    ax.set_ylabel("")
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_ylim(ylim)

    #########################################################
    ax = fig.add_subplot(grid_specs[1, 0])
    seaborn.pointplot(alpha, x="Time point", y="value", hue="seizure_number", ax=ax, palette="dark",
                      errorbar=None, markers="s", markersize=4, order=order, legend=False)
    ax.axvline(x=4)
    ax.set_xlabel("Feature value \n\n c) alpha", fontsize=8)
    ax.set_ylabel("")
    ax.yaxis.set_tick_params(labelsize=8)
    ax.xaxis.set_tick_params(labelsize=8)
    ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_ylim(ylim)

    #########################################################
    ax = fig.add_subplot(grid_specs[1, 1])
    seaborn.pointplot(beta, x="Time point", y="value", hue="seizure_number", ax=ax, palette="dark",
                      errorbar=None, markers="s", markersize=4, order=order, legend=False)
    ax.axvline(x=4)
    ax.set_xlabel("Feature value \n\n d) beta", fontsize=8)
    ax.set_ylabel("")
    ax.axes.get_yaxis().set_ticks([])
    ax.xaxis.set_tick_params(labelsize=8)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_ylim(ylim)

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight", pad_inches=0.2,
                    transparent=False, facecolor='white')
    plt.show()


def plot_topoplot_features_frame(impacted_channels: list, entropy_features: list, theta_psd_features: list,
                                 alpha_psd_features: list, entropy_min_value: float, entropy_max_value: float,
                                 theta_psd_min_value: float, theta_psd_max_value: float,
                                 alpha_psd_min_value: float, alpha_psd_max_value: float,
                                 output_file: str = None):
    """
    Plot a single frame of the topographic video
    :param impacted_chanles: channels displaying a ictal event
    :param features [entropy, theta psd, alpha psd]: feature value per channel
    :param min_value [entropy, theta psd, alpha psd]: minimum value across all the frames
    :param max_value [entropy, theta psd, alpha psd]: maximum value across all the frames
    :param output_file: if specified the figure will be saved
    """
    electrode_positions = settings["electrode_positions"]

    grid_specs = GridSpec(1, 3)
    fig = plt.figure(figsize=(9, 3))
    xi, yi = np.mgrid[-1:1:100j, -1:1:100j]

    def single_topomap(ax_object, data_array: list, min_value: float, max_value: float):
        ax_object.axis((-1.2, 1.2, -1.2, 1.2))
        circle = Circle([0, 0], radius=1, fill=False)
        ax_object.add_patch(circle)

        points = []
        channel_names = [x[0] for x in data_array]
        channel_values = [x[1] for x in data_array]
        for channel in channel_names:
            point = [x * 1.2 for x in electrode_positions[channel]]
            points.append(point)

        zi = griddata(points, channel_values, (xi, yi), method="cubic")
        colormap = plt.cm.jet
        normalize = matplotlib.colors.Normalize(vmin=min_value, vmax=max_value)
        ax_object.contourf(xi, yi, zi, 10, cmap=colormap, norm=normalize)

        for electrode, coordinate in electrode_positions.items():
            ax_object.text(coordinate[0], coordinate[1], electrode,
                           verticalalignment='center',
                           horizontalalignment='center',
                           size=12)

        ax_object.axes.get_xaxis().set_ticks([])
        ax_object.axes.get_yaxis().set_ticks([])
        ax_object.spines['top'].set_visible(False)
        ax_object.spines['right'].set_visible(False)
        ax_object.spines['bottom'].set_visible(False)
        ax_object.spines['left'].set_visible(False)

    #########################################################
    ax = fig.add_subplot(grid_specs[0, 0])
    single_topomap(ax, entropy_features, entropy_min_value, entropy_max_value)
    ax.set_xlabel("Approximate entropy", fontsize=12)

    #########################################################
    ax = fig.add_subplot(grid_specs[0, 1])
    single_topomap(ax, theta_psd_features, theta_psd_min_value, theta_psd_max_value)
    ax.set_xlabel("PSD theta band", fontsize=12)

    #########################################################
    ax = fig.add_subplot(grid_specs[0, 2])
    single_topomap(ax, alpha_psd_features, alpha_psd_min_value, alpha_psd_max_value)
    ax.set_xlabel("PSD alpha band", fontsize=12)

    if output_file:
        plt.savefig(output_file, dpi=100, bbox_inches="tight", pad_inches=0.2,
                    transparent=False, facecolor='white')

    plt.show()


def plot_chord_diagram_windows(chb_dataset: tuple, siena_dataset: tuple,
                               tusz_gnsz_dataset: tuple, tusz_fnsz_dataset: tuple,
                               output_file: str):
    """
    Plot a chord diagram for each database
    :param chb_dataset: [database name, seizure type, dataframe, invalid links]
    :param siena_dataset: [database name, seizure type, dataframe, invalid links]
    :param tusz_gnsz_dataset: [database name, seizure type, dataframe, invalid links]
    :param tusz_gnsz_dataset: [database name, seizure type, dataframe, invalid links]
    :param output_file: if specified the figure will be saved
    """
    class LinkHandler():
        def __init__(self, invalid_links):
            self.invalid_links = invalid_links
        
        def __call__(self, source_sector, destine_sector):
            if (source_sector, destine_sector) in self.invalid_links:
                return dict(alpha=0)

    def build_circos_object(dataset: pandas.DataFrame, invalid_links: set, label: str):
        circos = Circos.chord_diagram(dataset,
                                      space=3,
                                      r_lim=(85, 100),
                                      cmap="Set2",
                                      ticks_interval=500,
                                      label_kws=dict(r=90, size=8, color="black"),
                                      link_kws=dict(ec="black", lw=0.5),
                                      link_kws_handler=LinkHandler(invalid_links))
        circos.text(label, fontsize=8, r=115, deg=180, fontname="Times New Roman")
        return circos

    grid_specs = GridSpec(2, 2, wspace=0.1, hspace=0.12)
    fig = plt.figure(figsize=(8, 8))

    circos_1 = build_circos_object(chb_dataset[-1], chb_dataset[-2], "(a) CHB-MIT")
    ax = fig.add_subplot(grid_specs[0, 0], polar=True)
    fig = circos_1.plotfig(ax=ax)

    circos_2 = build_circos_object(siena_dataset[-1], siena_dataset[-2], "(b) Siena (IAS)")
    ax = fig.add_subplot(grid_specs[0, 1], polar=True)
    fig = circos_2.plotfig(ax=ax)

    circos_3 = build_circos_object(tusz_gnsz_dataset[-1], tusz_gnsz_dataset[-2], "(c) TUZ (gnsz)")
    ax = fig.add_subplot(grid_specs[1, 0], polar=True)
    fig = circos_3.plotfig(ax=ax)

    circos_4 = build_circos_object(tusz_fnsz_dataset[-1], tusz_fnsz_dataset[-2], "(d) TUSZ (fnsz)")
    ax = fig.add_subplot(grid_specs[1, 1], polar=True)
    fig = circos_4.plotfig(ax=ax)

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight", pad_inches=0.2,
                    transparent=False, facecolor='white')
    plt.show()


def plot_chord_diagram_windows_inter(siena_dataset: tuple, tusz_dataset: tuple, output_file: str):
    """
    Plot a chord diagram for each database
    :param siena_dataset: [database name, seizure type, dataframe, invalid links]
    :param tusz_dataset: [database name, seizure type, dataframe, invalid links]
    :param output_file: if specified the figure will be saved
    """
    class LinkHandler():
        def __init__(self, invalid_links):
            self.invalid_links = invalid_links
        
        def __call__(self, source_sector, destine_sector):
            if (source_sector, destine_sector) in self.invalid_links:
                return dict(alpha=0)

    def build_circos_object(dataset: pandas.DataFrame, invalid_links: set, label: str):
        circos = Circos.chord_diagram(dataset,
                                      space=3,
                                      r_lim=(85, 100),
                                      cmap="Set2",
                                      ticks_interval=500,
                                      label_kws=dict(r=90, size=8, color="black"),
                                      link_kws=dict(ec="black", lw=0.5),
                                      link_kws_handler=LinkHandler(invalid_links))
        circos.text(label, fontsize=8, r=115, deg=180, fontname="Times New Roman")
        return circos

    grid_specs = GridSpec(1, 2, wspace=0.1, hspace=0.12)
    fig = plt.figure(figsize=(8, 4))

    circos_1 = build_circos_object(siena_dataset[-1], siena_dataset[-2], "(a) Siena vs TUEP")
    ax = fig.add_subplot(grid_specs[0, 0], polar=True)
    fig = circos_1.plotfig(ax=ax)

    circos_2 = build_circos_object(tusz_dataset[-1], tusz_dataset[-2], "(c) TUSZ vs TUEP")
    ax = fig.add_subplot(grid_specs[0, 1], polar=True)
    fig = circos_2.plotfig(ax=ax)

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight", pad_inches=0.2,
                    transparent=False, facecolor='white')
    plt.show()