import os
import glob
import numpy
import pandas
import logging
import networkx
from typing import Iterable
from pandasql import sqldf
from . import settings
from .visuals import plot_univariate_intra_bar_chart, plot_univariate_intra_dist_chart, \
                     plot_topoplot_features_time, \
                     plot_univariate_inter_bar_chart, plot_univariate_inter_dist_chart, \
                     plot_univariate_inter_bar_chart_psd, \
                     plot_univariate_inter_dist_chart_psd, plot_univariate_intra_bar_chart_psd, \
                     plot_univariate_intra_dist_chart_psd, \
                     plot_network_features_time, plot_graph_striplot_chart, plot_graph_pointplot_chart, \
                     plot_chord_diagram_windows, plot_chord_diagram_windows_inter


class IntraUnivariateFeatureAnalyzer():

    def __init__(self, feature_name: str):
        """
        :param feature_name: name of the feature for analysis
        """
        base_path = os.getenv("BIOMARKERS_PROJECT_HOME")
        self.base_path = os.path.join(base_path, "features", self.DATASET, f"{feature_name}.csv")
        self.raw_features = self.base_path
        self.feature_name = feature_name
        self.univariate_channels_groups = settings[self.DATASET]["univariate_channels_groups"]
        self.channels = settings[self.DATASET]["channels"]

    @property
    def raw_features(self) -> pandas.DataFrame:
        return self._raw_features

    @raw_features.setter
    def raw_features(self, filename: str) -> None:
        """
        Read the CSV file for the selected feature
        :param filename: full path to the feature file
        """
        self._raw_features = pandas.read_csv(filename)

    def univariate_zone_bar_chart(self, zone: str, seizure_type: str) -> None:
        """
        Separate data into groups and plot bar chart
        :param zone: name of the brain zone
        :param seizure_type: type of seizure
        """
        features = self.raw_features
        if seizure_type != "unknown":
            features = features[features.seizure_type == seizure_type]

        channel_groups = self.univariate_channels_groups[zone]
        features = features[features.channel.isin(channel_groups)]

        for stage in settings["intra_windows_categories"]:
            features.loc[((features.time_point == stage[1]) &
                          (features.seizure_stage == stage[0])), "time_point"] = stage[3]

        delta_features = features[features.band == "delta"]
        theta_features = features[features.band == "theta"]
        alpha_features = features[features.band == "alpha"]
        beta_features = features[features.band == "beta"]
        gamma_features = features[features.band == "gamma"]
        all_features = features[features.band == "all"]

        base_path = os.getenv("BIOMARKERS_PROJECT_HOME")
        output_file = os.path.join(base_path, "images", self.DATASET,
                                   f"bar_intra_{self.feature_name}_{zone}_{seizure_type}.png")
        if self.feature_name != "power_spectral_density":
            plot_univariate_intra_bar_chart(delta_features, theta_features,
                                            alpha_features, beta_features,
                                            gamma_features, all_features,
                                            self.feature_name, output_file)
        else:
            plot_univariate_intra_bar_chart_psd(delta_features, theta_features,
                                                alpha_features, beta_features,
                                                gamma_features, self.feature_name,
                                                output_file)

    def univariate_zone_dist_chart(self, zone: str, seizure_type: str) -> None:
        """
        Separate data into groups and plot KDE chart
        :param zone: name of the brain zone
        :param seizure_type: type of seizure
        """
        features = self.raw_features
        if seizure_type != "unknown":
            features = features[features.seizure_type == seizure_type]

        channel_groups = self.univariate_channels_groups[zone]
        features = features[features.channel.isin(channel_groups)]
        features["Time point"] = "unknown"
        for stage in settings["intra_windows_categories"]:
            features.loc[((features.time_point == stage[1]) &
                          (features.seizure_stage == stage[0])), "Time point"] = stage[3]

        delta_features = features[features.band == "delta"]
        theta_features = features[features.band == "theta"]
        alpha_features = features[features.band == "alpha"]
        beta_features = features[features.band == "beta"]
        gamma_features = features[features.band == "gamma"]
        all_features = features[features.band == "all"]

        base_path = os.getenv("BIOMARKERS_PROJECT_HOME")
        output_file = os.path.join(base_path, "images", self.DATASET,
                                   f"dist_intra_{self.feature_name}_{zone}_{seizure_type}.png")
        if self.feature_name != "power_spectral_density":
            plot_univariate_intra_dist_chart(delta_features, theta_features,
                                             alpha_features, beta_features,
                                             gamma_features, all_features,
                                             self.feature_name, output_file)
        else:
            plot_univariate_intra_dist_chart_psd(delta_features, theta_features,
                                                 alpha_features, beta_features,
                                                 gamma_features, self.feature_name,
                                                 output_file)

    def univariate_topo_plot_average(self, seizure_type: int, band: int) -> None:
        """
        Separate data into groups for an averaged topographic chart across ictal events
        :param seizure_type: type of seizure
        :param band: any of [delta, theta, alpha, beta, gamma]
        """
        features = self.raw_features

        if seizure_type != "unknown":
            features = features[features.seizure_type == seizure_type]

        groups = []
        for stage in settings["intra_windows_categories"]:
            group = sqldf(f"""SELECT channel, AVG(value) as value FROM features
                              WHERE band='{band}'
                              AND seizure_stage='{stage[0]}'
                              AND time_point='{stage[1]}'
                              GROUP BY channel""")
            groups.append([stage[3], group])

        base_path = os.getenv("BIOMARKERS_PROJECT_HOME")
        output_file = os.path.join(base_path, "images", self.DATASET,
                                   f"topo_time_average_{self.feature_name}_{seizure_type}_{band}.png")

        if self.DATASET == "chb-mit":
            plot_topoplot_features_time(*tuple(groups), output_file, is_monopolar=False)
        else:
            plot_topoplot_features_time(*tuple(groups), output_file)

    def univariate_topo_plot_individual(self, seizure_number: int, band: int) -> None:
        """
        Separate data into groups for topographic chart for single ictal event
        :param seizure_number: index of the seizure
        :param band: any of [delta, theta, alpha, beta, gamma]
        """
        features = self.raw_features
        groups = []
        for stage in settings["intra_windows_categories"]:
            group = features[(features.band == band) &
                             (features.seizure_number == seizure_number) &
                             (features.seizure_stage == stage[0]) & (features.time_point == stage[1])]
            logging.info(f"Seizure type = {group['seizure_type'].iloc[-1]}")
            groups.append([stage[3], group])

        subject = groups[-1][-1]['patient'].iloc[-1]
        base_path = os.getenv("BIOMARKERS_PROJECT_HOME")
        output_file = os.path.join(base_path, "images", self.DATASET,
                                   f"topo_time_{subject}_{self.feature_name}_{band}.png")

        if self.DATASET == "chb-mit":
            plot_topoplot_features_time(*tuple(groups), output_file, is_monopolar=False)
        else:
            plot_topoplot_features_time(*tuple(groups), output_file)

    def processed_data_for_lilliefors(self, zone: str, seizure_type: str) -> list:
        """
        Separate data into groups for normality evaluation
        :param zone: name of the brain zone
        :param seizure_type: type of seizure
        """
        groups = []
        features = self.raw_features
        if seizure_type != "unknown":
            features = features[features.seizure_type == seizure_type]

        channel_groups = self.univariate_channels_groups[zone]
        features = features[features.channel.isin(channel_groups)]
        for band in ["delta", "theta", "alpha", "beta", "gamma", "all"]:
            for stage in settings["intra_windows_categories"]:
                values = features[(features.band == band) &
                                  (features.seizure_stage == stage[0]) &
                                  (features.time_point == stage[1])]
                groups.append([band, stage[1], values.value])

        return groups

    def processed_data_for_friedman(self, zone: str, seizure_type: str) -> list:
        """
        Separate data into groups for friedman evaluation
        :param zone: name of the brain zone
        :param seizure_type: type of seizure
        """
        groups = []
        features = self.raw_features
        if seizure_type != "unknown":
            features = features[features.seizure_type == seizure_type]
        channel_groups = self.univariate_channels_groups[zone]
        features = features[features.channel.isin(channel_groups)]

        for band in ["delta", "theta", "alpha", "beta", "gamma", "all"]:
            rank_arrays = [[] for _ in range(len(settings["intra_windows_categories"]))]
            for unique_seizure in features.seizure_number.unique():
                seizure_features = []
                for stage in settings["intra_windows_categories"]:
                    values = features[(features.band == band) &
                                      (features.seizure_stage == stage[0]) &
                                      (features.time_point == stage[1]) &
                                      (features.seizure_number == unique_seizure)]
                    seizure_features.append(float(numpy.mean(values.value)))

                rank_array = [sorted(seizure_features).index(x) for x in seizure_features]
                for idx, value in enumerate(rank_array):
                    rank_arrays[idx].append(value)

            groups.append([band, rank_arrays])

        return groups


class IntraUnivariateChbAnalyzer(IntraUnivariateFeatureAnalyzer):

    DATASET = "chb-mit"

    def __init__(self, feature: str):
        """
        :param feature: name of the feature file
        """
        super().__init__(feature)

    def hemisfere_bar_chart(self, hemisfere: str, seizure_type: str) -> None:
        """
        :param hemisfere: any of [left, right]
        :param seizure_type: any of [unknown]
        """
        self.univariate_zone_bar_chart(hemisfere, seizure_type)

    def hemisfere_dist_chart(self, hemisfere: str, seizure_type: str) -> None:
        """
        :param hemisfere: any of [left, right]
        :param seizure_type: any of [unknown]
        """
        self.univariate_zone_dist_chart(hemisfere, seizure_type)


class IntraUnivariateSienaAnalyzer(IntraUnivariateFeatureAnalyzer):

    DATASET = "siena"

    def __init__(self, feature: str):
        """
        :param feature: name of the feature file
        """
        super().__init__(feature)

    def region_bar_chart(self, region: str, seizure_type: str) -> None:
        """
        :param region: any of [temporal, parietal, frontal, occipital]
        :param seizure_type: any of [IAS]
        """
        self.univariate_zone_bar_chart(region, seizure_type)

    def region_dist_chart(self, region: str, seizure_type: str) -> None:
        """
        :param region: any of [temporal, parietal, frontal, occipital]
        :param seizure_type: any of [IAS]
        """
        self.univariate_zone_dist_chart(region, seizure_type)


class IntraUnivariateTuszAnalyzer(IntraUnivariateFeatureAnalyzer):

    DATASET = "tusz"

    def __init__(self, feature: str):
        """
        :param feature: name of the feature file
        """
        super().__init__(feature)

    def region_bar_chart(self, region: str, seizure_type: str) -> None:
        """
        :param region: any of [temporal, parietal, frontal, occipital]
        :param seizure_type: any of [tnsz, cpsz, gnsz, fnsz]
        """
        self.univariate_zone_bar_chart(region, seizure_type)

    def region_dist_chart(self, region: str, seizure_type: str) -> None:
        """
        :param region: any of [temporal, parietal, frontal, occipital]
        :param seizure_type: any of [tnsz, cpsz, gnsz, fnsz]
        """
        self.univariate_zone_dist_chart(region, seizure_type)


class IntraUnivariateTuepAnalyzer(IntraUnivariateFeatureAnalyzer):

    DATASET = "tuep"

    def __init__(self, feature: str):
        """
        :param feature: name of the feature file
        """
        super().__init__(feature)

    def processed_data_for_lilliefors(self, zone: str) -> list:
        """
        Separate data into groups for normality evaluation
        :param zone: name of the brain zone
        """
        groups = []
        features = self.raw_features

        channel_groups = self.univariate_channels_groups[zone]
        features = features[features.channel.isin(channel_groups)]
        for band in ["delta", "theta", "alpha", "beta", "gamma", "all"]:
            values = features[(features.band == band)]
            groups.append([band, values.value])

        return groups


class InterUnivariateFeatureAnalyzer():

    def __init__(self, feature_name: str):
        """
        :param feature_name: name of the feature file
        """
        base_path = os.getenv("BIOMARKERS_PROJECT_HOME")
        patients_base_path = os.path.join(base_path, "features", self.DATASET, f"{feature_name}.csv")
        self.patients_raw_features = pandas.read_csv(patients_base_path)

        healthy_base_path = os.path.join(base_path, "features", "TUEP", f"{feature_name}.csv")
        self.healthy_raw_features = pandas.read_csv(healthy_base_path)

        self.feature_name = feature_name
        self.univariate_channels_groups = settings[self.DATASET]["univariate_channels_groups"]

    def univariate_zone_bar_chart(self, zone: str) -> None:
        """
        Separate data into groups for bar evaluation
        :param region: name of the brain zone
        """
        preictal_features = self.patients_raw_features
        channel_groups = self.univariate_channels_groups[zone]
        preictal_features = preictal_features[preictal_features.channel.isin(channel_groups)]
        preictal_features = preictal_features[preictal_features.seizure_stage == "preictal"]

        ictal_features = self.patients_raw_features
        channel_groups = self.univariate_channels_groups[zone]
        ictal_features = ictal_features[ictal_features.channel.isin(channel_groups)]
        ictal_features = ictal_features[ictal_features.seizure_stage == "ictal"]

        healthy_features = self.healthy_raw_features
        channel_groups = self.univariate_channels_groups[zone]
        healthy_features = healthy_features[healthy_features.channel.isin(channel_groups)]

        bands = []
        for band in ["delta", "theta", "alpha", "beta", "gamma", "all"]:
            preictal_band_array = preictal_features[preictal_features.band == band][["band",
                                                                                     "feature",
                                                                                     "value"]]
            preictal_band_array["Group"] = "Preictal"
            ictal_band_array = ictal_features[ictal_features.band == band][["band", "feature", "value"]]
            ictal_band_array["Group"] = "Ictal"
            healthy_band_array = healthy_features[healthy_features.band == band][["band", "feature", "value"]]
            healthy_band_array["Group"] = "Healthy"
            merged_band_array = pandas.concat([preictal_band_array,
                                               ictal_band_array,
                                               healthy_band_array],
                                              ignore_index=True)
            bands.append(merged_band_array)

        base_path = os.getenv("BIOMARKERS_PROJECT_HOME")
        output_file = os.path.join(base_path, "images", self.DATASET,
                                   f"bar_inter_{self.feature_name}_{zone}.png")
        if self.feature_name != "power_spectral_density":
            plot_univariate_inter_bar_chart(*bands, self.feature_name, output_file)
        else:
            plot_univariate_inter_bar_chart_psd(*bands, self.feature_name, output_file)

    def univariate_zone_violin_chart(self, zone: str) -> None:
        """
        Separate data into groups for distribution chart
        :param region: name of the brain zone
        """
        preictal_features = self.patients_raw_features
        channel_groups = self.univariate_channels_groups[zone]
        preictal_features = preictal_features[preictal_features.channel.isin(channel_groups)]
        preictal_features = preictal_features[preictal_features.seizure_stage == "preictal"]

        ictal_features = self.patients_raw_features
        channel_groups = self.univariate_channels_groups[zone]
        ictal_features = ictal_features[ictal_features.channel.isin(channel_groups)]
        ictal_features = ictal_features[ictal_features.seizure_stage == "ictal"]

        healthy_features = self.healthy_raw_features
        channel_groups = self.univariate_channels_groups[zone]
        healthy_features = healthy_features[healthy_features.channel.isin(channel_groups)]

        bands = []
        for band in ["delta", "theta", "alpha", "beta", "gamma", "all"]:
            preictal_band_array = preictal_features[preictal_features.band == band][["band",
                                                                                     "feature",
                                                                                     "value"]]
            preictal_band_array["Group"] = "Preictal"
            ictal_band_array = ictal_features[ictal_features.band == band][["band", "feature", "value"]]
            ictal_band_array["Group"] = "Ictal"
            healthy_band_array = healthy_features[healthy_features.band == band][["band", "feature", "value"]]
            healthy_band_array["Group"] = "Healthy"
            merged_band_array = pandas.concat([preictal_band_array,
                                               ictal_band_array,
                                               healthy_band_array], ignore_index=True)
            bands.append(merged_band_array)

        base_path = os.getenv("BIOMARKERS_PROJECT_HOME")
        output_file = os.path.join(base_path, "images", self.DATASET,
                                   f"dist_inter_{self.feature_name}_{zone}.png")
        if self.feature_name != "power_spectral_density":
            plot_univariate_inter_dist_chart(*bands, self.feature_name, output_file)
        else:
            plot_univariate_inter_dist_chart_psd(*bands, self.feature_name, output_file)

    def processed_data_for_kruskal_wallis(self, zone: str) -> list:
        """
        Separate data into groups for kruskal evaluation
        :param zone: name of the brain zone
        """
        preictal_features = self.patients_raw_features
        channel_groups = self.univariate_channels_groups[zone]
        preictal_features = preictal_features[preictal_features.channel.isin(channel_groups)]
        preictal_features = preictal_features[preictal_features.seizure_stage == "preictal"]

        ictal_features = self.patients_raw_features
        channel_groups = self.univariate_channels_groups[zone]
        ictal_features = ictal_features[ictal_features.channel.isin(channel_groups)]
        ictal_features = ictal_features[ictal_features.seizure_stage == "ictal"]

        healthy_features = self.healthy_raw_features
        channel_groups = self.univariate_channels_groups[zone]
        healthy_features = healthy_features[healthy_features.channel.isin(channel_groups)]

        groups = []
        for band in ["delta", "theta", "alpha", "beta", "gamma", "all"]:
            preictal_band_array = preictal_features[preictal_features.band == band]["value"]
            ictal_band_array = ictal_features[ictal_features.band == band]["value"]
            healthy_band_array = healthy_features[healthy_features.band == band]["value"]
            # Keep the order - preictal, ictal, healthy
            groups.append([band, [preictal_band_array, ictal_band_array, healthy_band_array]])

        return groups


class InterUnivariateSienaAnalyzer(InterUnivariateFeatureAnalyzer):

    DATASET = "siena"

    def __init__(self, feature: str):
        """
        :param feature: name of the feature file
        :param feature_type: any of [univariate, bivariate]
        """
        super().__init__(feature)


class InterUnivariateTuszAnalyzer(InterUnivariateFeatureAnalyzer):

    DATASET = "tusz"

    def __init__(self, feature: str):
        """
        :param feature: name of the feature file
        :param feature_type: any of [univariate, bivariate]
        """
        super().__init__(feature)


class IntraBivariateFeatureAnalyzer():

    def __init__(self, feature_name: str):
        """
        :param feature_name: name of the feature file
        """
        base_path = os.getenv("BIOMARKERS_PROJECT_HOME")
        self.base_path = os.path.join(base_path, "features", self.DATASET, f"{feature_name}.csv")
        self.raw_features = self.base_path
        self.feature_name = feature_name
        self.channels = settings[self.DATASET]["channels"]
        self.univariate_channels_groups = settings[self.DATASET]["univariate_channels_groups"]

    @property
    def raw_features(self) -> pandas.DataFrame:
        return self._raw_features

    @raw_features.setter
    def raw_features(self, filename: str) -> None:
        """
        :param feature: full path to the feature file
        """
        self._raw_features = pandas.read_csv(filename)

    def bivariate_zone_bar_chart(self, zone: str, seizure_type: str) -> None:
        """
        Separate data into groups and plot bar chart
        :param region: name of the brain zone
        :param seizure_type: type of seizure
        """
        features = self.raw_features
        if seizure_type != "unknown":
            features = features[features.seizure_type == seizure_type]

        selected_channels = []
        channels = self.univariate_channels_groups[zone]
        for channel_1 in channels:
            for channel_2 in channels:
                selected_channels.append(f"{channel_1}_{channel_2}")

        features = features[features.channels.isin(selected_channels)]

        for stage in settings["intra_windows_categories"]:
            features.loc[((features.time_point == stage[1]) &
                          (features.seizure_stage == stage[0])), "time_point"] = stage[3]

        delta_features = features[features.band == "delta"]
        theta_features = features[features.band == "theta"]
        alpha_features = features[features.band == "alpha"]
        beta_features = features[features.band == "beta"]
        gamma_features = features[features.band == "gamma"]
        all_features = features[features.band == "all"]

        base_path = os.getenv("BIOMARKERS_PROJECT_HOME")
        output_file = os.path.join(base_path, "images", self.DATASET,
                                   f"bar_intra_{self.feature_name}_{zone}_{seizure_type}.png")
        if self.feature_name != "power_spectral_density":
            plot_univariate_intra_bar_chart(delta_features, theta_features, alpha_features,
                                            beta_features, gamma_features, all_features,
                                            output_file)
        else:
            plot_univariate_intra_bar_chart_psd(delta_features, theta_features, alpha_features,
                                                beta_features, gamma_features, output_file)

    def bivariate_zone_dist_chart(self, zone: str, seizure_type: str) -> None:
        """
        Separate data into groups and plot KDE chart
        :param region: name of the brain zone
        :param seizure_type: type of seizure
        """
        features = self.raw_features
        if seizure_type != "unknown":
            features = features[features.seizure_type == seizure_type]

        selected_channels = []
        channels = self.univariate_channels_groups[zone]
        for channel_1 in channels:
            for channel_2 in channels:
                selected_channels.append(f"{channel_1}_{channel_2}")
        features = features[features.channels.isin(selected_channels)]

        features["Time point"] = "unknown"
        for stage in settings["intra_windows_categories"]:
            features.loc[((features.time_point == stage[1]) &
                          (features.seizure_stage == stage[0])), "Time point"] = stage[3]

        delta_features = features[features.band == "delta"]
        theta_features = features[features.band == "theta"]
        alpha_features = features[features.band == "alpha"]
        beta_features = features[features.band == "beta"]
        gamma_features = features[features.band == "gamma"]
        all_features = features[features.band == "all"]

        base_path = os.getenv("BIOMARKERS_PROJECT_HOME")
        output_file = os.path.join(base_path, "images", self.DATASET,
                                   f"dist_intra_{self.feature_name}_{zone}_{seizure_type}.png")
        if self.feature_name != "power_spectral_density":
            plot_univariate_intra_dist_chart(delta_features, theta_features, alpha_features,
                                             beta_features, gamma_features, all_features,
                                             output_file)
        else:
            plot_univariate_intra_dist_chart_psd(delta_features, theta_features, alpha_features,
                                                 beta_features, gamma_features, output_file)

    def bivariate_network_plot_average(self, seizure_type: int, band: int) -> None:
        """
        Separate data into groups for an averaged network chart across ictal events
        :param seizure_type: type of seizure
        :param band: any of [delta, theta, alpha, beta]
        """
        features = self.raw_features

        if seizure_type != "unknown":
            features = features[features.seizure_type == seizure_type]

        networks = []
        threshold = features[features.band == band].value.mean() * 1.1
        averages = sqldf(f"""SELECT channels, AVG(value) as value FROM features
                             WHERE band='{band}'
                             GROUP BY channels, seizure_stage, time_point""")
        maximum = averages.value.max()

        for stage in settings["intra_windows_categories"]:
            group = sqldf(f"""SELECT channels, AVG(value) as value FROM features
                              WHERE band='{band}'
                              AND seizure_stage='{stage[0]}'
                              AND time_point='{stage[1]}'
                              GROUP BY channels""")

            network = networkx.Graph()
            line_widths = []

            for channel_1 in self.channels:
                for channel_2 in self.channels:
                    selected_group = group[group.channels == f"{channel_1}_{channel_2}"]
                    if (not len(selected_group) or selected_group.value.tolist()[0] < threshold):
                        continue
                    line_width = numpy.round(4 * ((selected_group.value.tolist()[0] - threshold) / (maximum - threshold)), 2)
                    line_widths.append(line_width)
                    network.add_edge(channel_1, channel_2, weight=selected_group.value.tolist()[0])

            logging.info(f"Seizure type = {seizure_type}")
            networks.append([stage[3], network])

        base_path = os.getenv("BIOMARKERS_PROJECT_HOME")
        output_file = os.path.join(base_path, "images", self.DATASET,
                                   f"network_avg_{self.feature_name}_{seizure_type}_{band}.png")

        if self.DATASET == "chb-mit":
            plot_network_features_time(*tuple(networks), line_widths, output_file, is_monopolar=False)
        else:
            plot_network_features_time(*tuple(networks), line_widths, output_file)

    def bivariate_network_plot_individual(self, seizure_number: int, band: int) -> None:
        """
        Separate data into groups for network chart for single ictal event
        :param seizure_number: index of the seizure
        :param band: any of [delta, theta, alpha, beta, gamma]
        """
        features = self.raw_features
        networks = []
        threshold = features.value.median() * 1.5
        maximum = features[features.band == band].value.max()

        for stage in settings["intra_windows_categories"]:
            group = features[(features.band == band) &
                             (features.seizure_number == seizure_number) &
                             (features.seizure_stage == stage[0]) & (features.time_point == stage[1])]

            network = networkx.Graph()
            line_widths = []
            for channel_1 in self.channels:
                for channel_2 in self.channels:
                    selected_group = group[group.channels == f"{channel_1}_{channel_2}"]
                    if (not len(selected_group) or selected_group.value.tolist()[0] < threshold):
                        continue
                    line_width = numpy.round(4 * ((selected_group.value.tolist()[0] - threshold) / (maximum - threshold)), 1)
                    line_widths.append(line_width)
                    network.add_edge(channel_1, channel_2, weight=selected_group.value.tolist()[0])

            logging.info(f"Seizure type = {group['seizure_type'].iloc[-1]}")
            networks.append([stage[3], network])

        subject = features.patient.tolist()[0]
        base_path = os.getenv("BIOMARKERS_PROJECT_HOME")
        output_file = os.path.join(base_path, "images", self.DATASET,
                                   f"network_{subject}_{self.feature_name}_{band}.png")

        if self.DATASET == "chb-mit":
            plot_network_features_time(*tuple(networks), line_widths, output_file, is_monopolar=False)
        else:
            plot_network_features_time(*tuple(networks), line_widths, output_file)

    def processed_data_for_friedman(self, zone: str, seizure_type: str) -> list:
        """
        Separate data into groups for friedman evaluation
        :param zone: name of the brain zone
        :param seizure_type: type of seizure
        """
        groups = []
        features = self.raw_features
        if seizure_type != "unknown":
            features = features[features.seizure_type == seizure_type]

        selected_channels = []
        channels = self.univariate_channels_groups[zone]
        for channel_1 in channels:
            for channel_2 in channels:
                selected_channels.append(f"{channel_1}_{channel_2}")

        features = features[features.channels.isin(selected_channels)]

        for band in ["delta", "theta", "alpha", "beta", "all"]:
            rank_arrays = [[] for _ in range(len(settings["intra_windows_categories"]))]
            for unique_seizure in features.seizure_number.unique():
                seizure_features = []
                for stage in settings["intra_windows_categories"]:
                    values = features[(features.band == band) &
                                      (features.seizure_stage == stage[0]) &
                                      (features.time_point == stage[1]) &
                                      (features.seizure_number == unique_seizure)]
                    seizure_features.append(float(numpy.mean(values.value)))

                rank_array = [sorted(seizure_features).index(x) for x in seizure_features]
                for idx, value in enumerate(rank_array):
                    rank_arrays[idx].append(value)

            groups.append([band, rank_arrays])

        return groups

    def processed_data_for_efficiency(self, seizure_type: str, threshold: float) -> Iterable[tuple]:
        """
        Compute adjacency matrices for efficiency analysis
        :param seizure_type: type of seizure
        :param threshold: if connection weight is under 'threshold',
                          connection weight is set to 0,
                          otherwise it is set to 1
        """
        features = self.raw_features
        if seizure_type != "unknown":
            features = features[features.seizure_type == seizure_type]

        for band in ["delta", "theta", "alpha", "beta", "gamma", "all"]:
            logging.info(f"Processing band = {band}")
            unique_seizures = features.seizure_number.unique()
            for unique_seizure in unique_seizures:
                for stage in settings["intra_windows_categories"]:
                    weighted_adj_matrix = features[(features.seizure_number == unique_seizure) &
                                                   (features.band == band) &
                                                   (features.time_point == stage[1]) &
                                                   (features.seizure_stage == stage[0])]

                    undirected_graph = []
                    unique_channels = set()

                    for _, row in weighted_adj_matrix.iterrows():
                        if row["value"] < threshold:
                            continue
                        node_1, node_2 = row["index_channels"].split("_")
                        unique_channels.add(node_1)
                        unique_channels.add(node_2)
                        undirected_graph.append([int(node_1), int(node_2)])

                    network = networkx.Graph()
                    for node in unique_channels:
                        network.add_node(node)
                    for edge in undirected_graph:
                        network.add_edge(edge[0], edge[1])
                    yield unique_seizure, band, stage[0], stage[1], network

    def network_stripplot_chart(self, seizure_type: str) -> None:
        """
        Separate efficiency data into groups and plot striplot chart
        :param seizure_type: type of seizure
        """
        features = self.raw_features

        if seizure_type != "unknown":
            features = features[features.seizure_type == seizure_type]

        features["Time point"] = "unknown"
        for stage in settings["intra_windows_categories"]:
            features.loc[((features.time_point == stage[1]) &
                          (features.seizure_stage == stage[0])), "Time point"] = stage[3]

        delta_features = features[features.band == "delta"]
        theta_features = features[features.band == "theta"]
        alpha_features = features[features.band == "alpha"]
        beta_features = features[features.band == "beta"]
        gamma_features = features[features.band == "gamma"]

        base_path = os.getenv("BIOMARKERS_PROJECT_HOME")
        output_file = os.path.join(base_path, "images", self.DATASET,
                                   f"scatter_{self.feature_name}_{seizure_type}.png")
        plot_graph_striplot_chart(delta_features, theta_features, alpha_features,
                                  beta_features, gamma_features, output_file)

    def network_lineplot_chart(self, seizure_list: list) -> None:
        """
        Separate efficiency data into single evetns and plot lineplot chart
        :param seizure_type: type of seizure
        """
        features = self.raw_features

        features = features[features.seizure_number.isin(seizure_list)]

        features["Time point"] = "unknown"
        for stage in settings["intra_windows_categories"]:
            features.loc[((features.time_point == stage[1]) &
                          (features.seizure_stage == stage[0])), "Time point"] = stage[3]

        minimum = features.value.max()
        maximum = features.value.min()
        delta_features = features[features.band == "delta"]
        theta_features = features[features.band == "theta"]
        alpha_features = features[features.band == "alpha"]
        beta_features = features[features.band == "beta"]
        gamma_features = features[features.band == "gamma"]

        base_path = os.getenv("BIOMARKERS_PROJECT_HOME")
        output_file = os.path.join(base_path, "images", self.DATASET,
                                   f"lineplot_{self.feature_name}_individual.png")
        plot_graph_pointplot_chart(delta_features, theta_features, alpha_features,
                                   beta_features, gamma_features,
                                   [maximum, minimum], output_file)


class IntraBivariateChbAnalyzer(IntraBivariateFeatureAnalyzer):

    DATASET = "chb-mit"

    def __init__(self, feature: str):
        """
        :param feature: name of the feature file
        :param feature_type: any of [univariate, bivariate]
        """
        super().__init__(feature)


class IntraBivariateSienaAnalyzer(IntraBivariateFeatureAnalyzer):

    DATASET = "siena"

    def __init__(self, feature: str):
        """
        :param feature: name of the feature file
        """
        super().__init__(feature)


class IntraBivariateTuszAnalyzer(IntraBivariateFeatureAnalyzer):

    DATASET = "tusz"

    def __init__(self, feature: str):
        """
        :param feature: name of the feature file
        """
        super().__init__(feature)


class InterBivariateFeatureAnalyzer():

    def __init__(self, feature_name: str):
        """
        :param feature_name: name of the feature file
        """
        base_path = os.getenv("BIOMARKERS_PROJECT_HOME")
        patients_base_path = os.path.join(base_path, "features", self.DATASET, f"{feature_name}.csv")
        self.patients_raw_features = pandas.read_csv(patients_base_path)

        healthy_base_path = os.path.join(base_path, "features", "TUEP", f"{feature_name}.csv")
        self.healthy_raw_features = pandas.read_csv(healthy_base_path)

        self.feature_name = feature_name
        self.univariate_channels_groups = settings[self.DATASET]["univariate_channels_groups"]

    def bivariate_zone_bar_chart(self, zone: str) -> None:
        """
        Separate data into groups and plot bar chart
        :param zone: name of the brain zone
        """
        selected_channels = []
        for channel_1 in self.univariate_channels_groups[zone]:
            for channel_2 in self.univariate_channels_groups[zone]:
                selected_channels.append(f"{channel_1}_{channel_2}")

        preictal_features = self.patients_raw_features
        preictal_features = preictal_features[preictal_features.channels.isin(selected_channels)]
        preictal_features = preictal_features[preictal_features.seizure_stage == "preictal"]

        ictal_features = self.patients_raw_features
        ictal_features = ictal_features[ictal_features.channels.isin(selected_channels)]
        ictal_features = ictal_features[ictal_features.seizure_stage == "ictal"]

        healthy_features = self.healthy_raw_features
        healthy_features = healthy_features[healthy_features.channels.isin(selected_channels)]

        bands = []
        for band in ["delta", "theta", "alpha", "beta", "gamma", "all"]:
            preictal_band_array = preictal_features[preictal_features.band == band][["band",
                                                                                     "feature",
                                                                                     "value"]]
            preictal_band_array["Group"] = "Preictal"
            ictal_band_array = ictal_features[ictal_features.band == band][["band", "feature", "value"]]
            ictal_band_array["Group"] = "Ictal"
            healthy_band_array = healthy_features[healthy_features.band == band][["band", "feature", "value"]]
            healthy_band_array["Group"] = "Healthy"
            merged_band_array = pandas.concat([preictal_band_array,
                                               ictal_band_array,
                                               healthy_band_array], ignore_index=True)
            bands.append(merged_band_array)

        base_path = os.getenv("BIOMARKERS_PROJECT_HOME")
        output_file = os.path.join(base_path, "images", self.DATASET,
                                   f"bar_inter_{self.feature_name}_{zone}.png")
        plot_univariate_inter_bar_chart(*bands, output_file)

    def bivariate_zone_violin_chart(self, zone: str) -> None:
        """
        Separate data into groups and plot distribution chart
        :param zone: name of the brain zone
        """
        selected_channels = []
        for channel_1 in self.univariate_channels_groups[zone]:
            for channel_2 in self.univariate_channels_groups[zone]:
                selected_channels.append(f"{channel_1}_{channel_2}")

        preictal_features = self.patients_raw_features
        preictal_features = preictal_features[preictal_features.channels.isin(selected_channels)]
        preictal_features = preictal_features[preictal_features.seizure_stage == "preictal"]

        ictal_features = self.patients_raw_features
        ictal_features = ictal_features[ictal_features.channels.isin(selected_channels)]
        ictal_features = ictal_features[ictal_features.seizure_stage == "ictal"]

        healthy_features = self.healthy_raw_features
        healthy_features = healthy_features[healthy_features.channels.isin(selected_channels)]

        bands = []
        for band in ["delta", "theta", "alpha", "beta", "gamma", "all"]:
            preictal_band_array = preictal_features[preictal_features.band == band][["band",
                                                                                     "feature",
                                                                                     "value"]]
            preictal_band_array["Group"] = "Preictal"
            ictal_band_array = ictal_features[ictal_features.band == band][["band", "feature", "value"]]
            ictal_band_array["Group"] = "Ictal"
            healthy_band_array = healthy_features[healthy_features.band == band][["band", "feature", "value"]]
            healthy_band_array["Group"] = "Healthy"
            merged_band_array = pandas.concat([preictal_band_array,
                                               ictal_band_array,
                                               healthy_band_array], ignore_index=True)
            bands.append(merged_band_array)

        base_path = os.getenv("BIOMARKERS_PROJECT_HOME")
        output_file = os.path.join(base_path, "images", self.DATASET,
                                   f"dist_inter_{self.feature_name}_{zone}.png")
        plot_univariate_inter_dist_chart(*bands, output_file)

    def processed_data_for_kruskal_wallis(self, zone: str) -> list:
        """
        Separate data into groups for kruskal evaluation
        :param zone: name of the brain zone
        """
        selected_channels = []
        channels = self.univariate_channels_groups[zone]
        for channel_1 in channels:
            for channel_2 in channels:
                selected_channels.append(f"{channel_1}_{channel_2}")

        preictal_features = self.patients_raw_features
        preictal_features = preictal_features[preictal_features.channels.isin(selected_channels)]
        preictal_features = preictal_features[preictal_features.seizure_stage == "preictal"]

        ictal_features = self.patients_raw_features
        ictal_features = ictal_features[ictal_features.channels.isin(selected_channels)]
        ictal_features = ictal_features[ictal_features.seizure_stage == "ictal"]

        healthy_features = self.healthy_raw_features
        healthy_features = healthy_features[healthy_features.channels.isin(selected_channels)]

        groups = []
        for band in ["delta", "theta", "alpha", "beta", "gamma", "all"]:
            preictal_band_array = preictal_features[preictal_features.band == band]["value"]
            ictal_band_array = ictal_features[ictal_features.band == band]["value"]
            healthy_band_array = healthy_features[healthy_features.band == band]["value"]
            # Keep the order - preictal, ictal, healthy
            groups.append([band, [preictal_band_array, ictal_band_array, healthy_band_array]])

        return groups


class InterBivariateSienaAnalyzer(InterBivariateFeatureAnalyzer):

    DATASET = "siena"

    def __init__(self, feature: str):
        """
        :param feature: name of the feature file
        """
        super().__init__(feature)


class InterBivariateTuszAnalyzer(InterBivariateFeatureAnalyzer):

    DATASET = "tusz"

    def __init__(self, feature: str):
        """
        :param feature: name of the feature file
        """
        super().__init__(feature)


class CirclizerCharts():

    def __init__(self):
        self.groups = (("chb", ("unknown",)),
                       ("siena", ("IAS",)),
                       ("tusz", ("gnsz", "fnsz")))
        self.base_path = os.path.join(os.getenv("BIOMARKERS_PROJECT_HOME"), "reports", "*")
    
    def aggregate_single_group(self, database: str, seizure_type: str) -> tuple:
        """
        Count the number of significant differences per pair-wise
        comparison (Nemenyi's test, intra analysis). It aggregates across features, brain
        regions and bands:
        :param database: name of the database
        :param seizure_type: type of the seizure
        """
        report_file = glob.glob(f"{self.base_path}/friedman_{database}*.csv")
        matrix_length = len(settings["intra_windows_categories"])
        aggregated_values = [[0.1 for _ in range(matrix_length)] for _ in range(matrix_length)]

        for file in report_file:
            report_dataframe = pandas.read_csv(file)
            report_dataframe = report_dataframe[report_dataframe.seizure_type == seizure_type]

            for idx1, stage1 in enumerate(settings["intra_windows_categories"]):
                for idx2, stage2 in enumerate(settings["intra_windows_categories"]):
                    column_name = f"{stage1[-1]}-{stage2[-1]}"

                    for _, row in report_dataframe.iterrows():
                        p_value = row.get(column_name)
                        if p_value is None or row["feature"] == "Aggregation":
                            continue
                        if float(p_value) < 0.05:
                            aggregated_values[idx1][idx2] += 1
            
        invalid_links = set()
        for idx1, row in enumerate(aggregated_values):
            for idx2, value in enumerate(row):
                if value == 0.1:
                    source = settings["intra_windows_abbreviations"][settings["intra_windows_categories"][idx1][-1]]
                    target = settings["intra_windows_abbreviations"][settings["intra_windows_categories"][idx2][-1]]
                    invalid_links.add((source, target))

        columns = [settings["intra_windows_abbreviations"][x[-1]] for x in settings["intra_windows_categories"]]
        matrix_dataframe = pandas.DataFrame(aggregated_values, index=columns, columns=columns)
        return invalid_links, matrix_dataframe

    def windows_based_charts(self) -> None:
        """
        Aggregate statistical test results and plot chord diagram
        """
        aggregated_matrices = list()
        output_directory = os.getenv("BIOMARKERS_PROJECT_HOME")
        output_file = os.path.join(output_directory, "images", "chord_diagram.png")

        for (database, seizure_types) in self.groups:
            for seizure_type in seizure_types:
                invalid_link, aggregated_matrix = self.aggregate_single_group(database, seizure_type)
                aggregated_matrices.append((database, seizure_type, invalid_link, aggregated_matrix))

        plot_chord_diagram_windows(*tuple(aggregated_matrices), output_file)

    def aggregate_single_inter_group(self, database: str) -> tuple:
        """
        Count the number of significant differences per pair-wise
        comparison (Duns's test, inter analysis). It aggregates across features, brain
        regions and bands:
        :param database: name of the database
        """
        report_file = glob.glob(f"{self.base_path}/kruskal_{database}*.csv")
        matrix_length = len(settings["inter_windows_categories"])
        aggregated_values = [[0.1 for _ in range(matrix_length)] for _ in range(matrix_length)]

        for file in report_file:
            report_dataframe = pandas.read_csv(file)

            for idx1, stage1 in enumerate(settings["inter_windows_categories"]):
                for idx2, stage2 in enumerate(settings["inter_windows_categories"]):
                    column_name = f"{stage1[-1]}-{stage2[-1]}"

                    for _, row in report_dataframe.iterrows():
                        p_value = row.get(column_name)
                        if p_value is None or row["feature"] == "Aggregation":
                            continue
                        if float(p_value) < 0.05:
                            aggregated_values[idx1][idx2] += 1

        invalid_links = set()
        for idx1, row in enumerate(aggregated_values):
            for idx2, value in enumerate(row):
                if value == 0.1:
                    source = settings["inter_windows_categories"][idx1][-1]
                    target = settings["inter_windows_categories"][idx2][-1]
                    invalid_links.add((source, target))

        columns = [x[-1] for x in settings["inter_windows_categories"]]
        matrix_dataframe = pandas.DataFrame(aggregated_values, index=columns, columns=columns)
        return invalid_links, matrix_dataframe

    def windows_based_charts_inter(self) -> None:
        """
        Aggregate statistical test results and plot chord diagram
        """
        aggregated_matrices = list()
        output_directory = os.getenv("BIOMARKERS_PROJECT_HOME")
        output_file = os.path.join(output_directory, "images", "chord_diagram_inter.png")

        for (database, _) in self.groups[1:]:
            invalid_link, aggregated_matrix = self.aggregate_single_inter_group(database)
            aggregated_matrices.append((database, invalid_link, aggregated_matrix))

        plot_chord_diagram_windows_inter(*tuple(aggregated_matrices), output_file)

