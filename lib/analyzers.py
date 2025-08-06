import os
import glob
import numpy
import pandas
import logging
import networkx
from pandasql import sqldf
from . import settings
from .visuals import plot_univariate_intra_bar_chart, plot_univariate_intra_dist_chart, \
                     plot_topoplot_features_time, \
                     plot_univariate_inter_bar_chart, plot_univariate_inter_dist_chart, \
                     plot_univariate_inter_bar_chart_psd, \
                     plot_univariate_inter_dist_chart_psd, plot_univariate_intra_bar_chart_psd, \
                     plot_univariate_intra_dist_chart_psd, \
                     plot_chord_diagram_windows, \
                     plot_heatmap_diagram_windows


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

    def univariate_zone_bar_chart(self) -> None:
        """
        Separate data into groups and plot bar chart
        """
        features = self.raw_features

        features = sqldf(f"""SELECT band, seizure_stage, time_point, AVG(value) as value
                             FROM features
                             GROUP BY patient, seizure_number, hemisphere, band,
                             seizure_stage, time_point""")

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
                                   f"bar_intra_{self.feature_name}.png")
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

    def univariate_zone_dist_chart(self) -> None:
        """
        Separate data into groups and plot KDE chart
        """
        features = self.raw_features

        features = sqldf(f"""SELECT band, seizure_stage, time_point, AVG(value) as value
                             FROM features
                             GROUP BY patient, seizure_number, hemisphere, band,
                             seizure_stage, time_point""")

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
                                   f"dist_intra_{self.feature_name}.png")
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

    def univariate_topo_plot_average(self, band: int) -> None:
        """
        Separate data into groups for an averaged topographic chart across ictal events
        :param seizure_type: type of seizure
        :param band: any of [delta, theta, alpha, beta, gamma]
        """
        features = self.raw_features
        dir(features)

        groups = []
        for stage in settings["intra_windows_categories"]:
            group = sqldf(f"""SELECT channel, AVG(value) as value FROM features
                              WHERE band='{band}'
                              AND seizure_stage='{stage[0]}'
                              AND time_point='{stage[1]}'
                              GROUP BY channel""")
            groups.append([stage[3], group])

        base_path = os.getenv("BIOMARKERS_PROJECT_HOME")
        channels = group.channel.unique()
        output_file = os.path.join(base_path, "images", self.DATASET,
                                   f"topo_time_average_{self.feature_name}_{band}.png")

        plot_topoplot_features_time(*tuple(groups), channels, output_file)

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
            groups.append([stage[3], group])
            channels = group.channel.unique()

        subject = groups[-1][-1]['patient'].iloc[-1]
        base_path = os.getenv("BIOMARKERS_PROJECT_HOME")
        output_file = os.path.join(base_path, "images", self.DATASET,
                                   f"topo_time_{subject}_{self.feature_name}_{band}.png")

        plot_topoplot_features_time(*tuple(groups), channels, output_file)

    def processed_data_for_lilliefors(self) -> list:
        """
        Separate data into groups for normality evaluation
        """
        groups = []
        features = self.raw_features
        _ = repr(features)
        averaged_features = sqldf(f"""SELECT patient, seizure_number, hemisphere, band,
                                             seizure_stage, time_point, AVG(value) as value
                                      FROM features
                                      GROUP BY patient, seizure_number, hemisphere, band,
                                               seizure_stage, time_point""")
        grouped_features = averaged_features.groupby(["band", "seizure_stage", "time_point"])
        for indices, group in grouped_features:
            groups.append([indices[0], indices[1], int(indices[2]), group.value])

        return groups

    def processed_data_for_friedman(self) -> list:
        """
        Separate data into groups for friedman evaluation
        """
        groups = []
        features = self.raw_features
        _ = repr(features)
        averaged_features = sqldf(f"""SELECT patient, seizure_number, hemisphere, band,
                                             seizure_stage, time_point, AVG(value) as value
                                      FROM features
                                      GROUP BY patient, seizure_number, hemisphere, band,
                                               seizure_stage, time_point""")

        for band in ["delta", "theta", "alpha", "beta", "gamma", "all"]:
            rank_arrays = [[] for _ in range(len(settings["intra_windows_categories"]))]
            for hemisphere in averaged_features.hemisphere.unique():
                hemisphere_features = averaged_features[averaged_features.hemisphere == hemisphere]

                for unique_seizure in hemisphere_features.seizure_number.unique():
                    seizure_features = []
                    for stage in settings["intra_windows_categories"]:
                        values = hemisphere_features[(hemisphere_features.band == band) &
                                                     (hemisphere_features.seizure_stage == stage[0]) &
                                                     (hemisphere_features.time_point == stage[1]) &
                                                     (hemisphere_features.seizure_number == unique_seizure)]
                        seizure_features.extend([float(x) for x in values.value])

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


class IntraUnivariateSienaAnalyzer(IntraUnivariateFeatureAnalyzer):

    DATASET = "siena"

    def __init__(self, feature: str):
        """
        :param feature: name of the feature file
        """
        super().__init__(feature)


class IntraUnivariateTuszAnalyzer(IntraUnivariateFeatureAnalyzer):

    DATASET = "tusz"

    def __init__(self, feature: str):
        """
        :param feature: name of the feature file
        """
        super().__init__(feature)


class IntraUnivariateTuepAnalyzer(IntraUnivariateFeatureAnalyzer):

    DATASET = "tuep"

    def __init__(self, feature: str):
        """
        :param feature: name of the feature file
        """
        super().__init__(feature)

    def processed_data_for_lilliefors(self) -> list:
        """
        Separate data into groups for normality evaluation
        """
        groups = []
        features = self.raw_features
        _ = repr(features)
        averaged_features = sqldf(f"""SELECT patient, id, hemisphere, band, AVG(value) as value
                                      FROM features
                                      GROUP BY patient, id, hemisphere, band""")
        grouped_features = averaged_features.groupby(["band"])
        for indices, group in grouped_features:
            groups.append([indices[0], None, None, group.value])

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

    def univariate_zone_bar_chart(self) -> None:
        """
        Separate data into groups for bar evaluation
        """
        features = self.patients_raw_features
        dir(features)
        averaged_features = sqldf(f"""SELECT seizure_stage, feature, band, AVG(value) as value
                                      FROM features
                                      GROUP BY patient, seizure_number, hemisphere, band,
                                               seizure_stage, time_point""")

        preictal_features = averaged_features[averaged_features.seizure_stage == "preictal"]
        ictal_features = averaged_features[averaged_features.seizure_stage == "ictal"]

        features = self.healthy_raw_features
        averaged_features = sqldf(f"""SELECT feature, band, AVG(value) as value
                                      FROM features
                                      GROUP BY patient, id, hemisphere, band""")
        healthy_features = averaged_features

        bands = []
        for band in ["delta", "theta", "alpha", "beta", "gamma", "all"]:
            preictal_band_array = preictal_features[preictal_features.band == band]
            preictal_band_array["Group"] = "Preictal"
            ictal_band_array = ictal_features[ictal_features.band == band]
            ictal_band_array["Group"] = "Ictal"
            healthy_band_array = healthy_features[healthy_features.band == band]
            healthy_band_array["Group"] = "Healthy"
            merged_band_array = pandas.concat([preictal_band_array,
                                               ictal_band_array,
                                               healthy_band_array],
                                               ignore_index=True)
            bands.append(merged_band_array)

        base_path = os.getenv("BIOMARKERS_PROJECT_HOME")
        output_file = os.path.join(base_path, "images", self.DATASET,
                                   f"bar_inter_{self.feature_name}.png")
        if self.feature_name != "power_spectral_density":
            plot_univariate_inter_bar_chart(*bands, self.feature_name, output_file)
        else:
            plot_univariate_inter_bar_chart_psd(*bands, self.feature_name, output_file)

    def univariate_zone_violin_chart(self) -> None:
        """
        Separate data into groups for distribution chart
        """
        groups = self.processed_data_for_kruskal_wallis()
        bands = []

        for band, features in groups.items():
            features_array = []
            for idx in range(features[0].shape[0]):
                features_array.append(["Preictal", band, self.feature_name, features[0][idx]])
            for idx in range(features[1].shape[0]):
                features_array.append(["Ictal", band, self.feature_name, features[1][idx]])
            for idx in range(features[2].shape[0]):
                features_array.append(["Healthy", band, self.feature_name, features[2][idx]])
        
            band_df = pandas.DataFrame(features_array, columns=["Group", "band", "feature", "value"])
            bands.append(band_df)

        base_path = os.getenv("BIOMARKERS_PROJECT_HOME")
        output_file = os.path.join(base_path, "images", self.DATASET,
                                   f"dist_inter_{self.feature_name}.png")
        if self.feature_name != "power_spectral_density":
            plot_univariate_inter_dist_chart(*bands, self.feature_name, output_file)
        else:
            plot_univariate_inter_dist_chart_psd(*bands, self.feature_name, output_file)

    def processed_data_for_kruskal_wallis(self) -> list:
        """
        Separate data into groups for kruskal evaluation
        """
        features = self.patients_raw_features
        dir(features)
        averaged_features = sqldf(f"""SELECT patient, seizure_number, hemisphere, band,
                                             seizure_stage, time_point, AVG(value) as value
                                      FROM features
                                      GROUP BY patient, seizure_number, hemisphere, band,
                                               seizure_stage, time_point""")

        preictal_features = averaged_features[averaged_features.seizure_stage == "preictal"]
        ictal_features = averaged_features[averaged_features.seizure_stage == "ictal"]

        features = self.healthy_raw_features
        averaged_features = sqldf(f"""SELECT patient, id, hemisphere, band,
                                             AVG(value) as value
                                      FROM features
                                      GROUP BY patient, id, hemisphere, band""")
        healthy_features = averaged_features

        groups = {}
        for band in ["delta", "theta", "alpha", "beta", "gamma", "all"]:
            preictal_band_array = preictal_features[preictal_features.band == band].value.tolist()
            ictal_band_array = ictal_features[ictal_features.band == band].value.tolist()
            healthy_band_array = healthy_features[healthy_features.band == band].value.tolist()

            # Keep the order - preictal, ictal, healthy
            groups[band] =  [numpy.array(preictal_band_array),
                             numpy.array(ictal_band_array),
                             numpy.array(healthy_band_array)]
        return groups


class InterUnivariateSienaAnalyzer(InterUnivariateFeatureAnalyzer):

    DATASET = "siena"

    def __init__(self, feature: str):
        """
        :param feature: name of the feature file
        """
        super().__init__(feature)


class InterUnivariateTuszAnalyzer(InterUnivariateFeatureAnalyzer):

    DATASET = "tusz"

    def __init__(self, feature: str):
        """
        :param feature: name of the feature file
        """
        super().__init__(feature)


class InterUnivariateChbAnalyzer(InterUnivariateFeatureAnalyzer):

    DATASET = "chb-mit"

    def __init__(self, feature: str):
        """
        :param feature: name of the feature file
        """
        super().__init__(feature)


class CirclizerCharts():

    def __init__(self):
        self.groups = ("chb", "siena", "tusz")
        self.base_path = os.path.join(os.getenv("BIOMARKERS_PROJECT_HOME"), "reports", "*")
    
    def aggregate_single_group(self, database: str, band: str=None, feature: str=None) -> tuple:
        """
        Count the number of significant differences per pair-wise
        comparison (Nemenyi's test, intra analysis). It aggregates across features, brain
        regions and bands:
        :param database: name of the database
        :param band: any of [delta, theta, alpha, beta, gamma]
        :param feature: name of the feature file
        """
        report_file = glob.glob(f"{self.base_path}/friedman_{database}*.csv")
        matrix_length = len(settings["intra_windows_categories"])
        aggregated_values = [[0.1 for _ in range(matrix_length)] for _ in range(matrix_length)]

        for file in report_file:
            if (not feature) or (feature and feature in file):
                report_dataframe = pandas.read_csv(file)
            else:
                continue

            if band:
                report_dataframe = report_dataframe[report_dataframe.band == band]

            for idx1, stage1 in enumerate(settings["intra_windows_categories"]):
                for idx2, stage2 in enumerate(settings["intra_windows_categories"]):
                    column_name = f"{stage1[-1]}-{stage2[-1]}"

                    for _, row in report_dataframe.iterrows():
                        p_value = row.get(column_name)
                        if p_value is None or row["feature"] == "Aggregation":
                            continue
                        if float(p_value) < 0.05:
                            aggregated_values[idx2][idx1] += 1
            
        invalid_links = set()
        for idx1, row in enumerate(aggregated_values):
            for idx2, value in enumerate(row):
                if value == 0.1:
                    source = settings["intra_windows_categories"][idx1][-1]
                    target = settings["intra_windows_categories"][idx2][-1]
                    invalid_links.add((source, target))

        columns = [x[-1] for x in settings["intra_windows_categories"]]
        matrix_dataframe = pandas.DataFrame(aggregated_values, index=columns, columns=columns)
        return invalid_links, matrix_dataframe

    def windows_based_charts(self) -> None:
        """
        Aggregate statistical test results and plot chord diagram
        """
        aggregated_matrices = list()
        output_directory = os.getenv("BIOMARKERS_PROJECT_HOME")
        output_file = os.path.join(output_directory, "images", "chord_diagram.png")

        for database in self.groups:
            invalid_link, aggregated_matrix = self.aggregate_single_group(database)
            aggregated_matrices.append((database, invalid_link, aggregated_matrix))

        plot_chord_diagram_windows(*tuple(aggregated_matrices), output_file)

    def windows_based_heat_map(self, band: str=None, feature: str=None) -> None:
        """
        Aggregate statistical test results and plot data as a color-encoded matrix
        :param band: any of [delta, theta, alpha, beta, gamma]
        :param feature: name of the feature
        """
        aggregated_matrices = list()
        output_directory = os.getenv("BIOMARKERS_PROJECT_HOME")
        output_file = os.path.join(output_directory, "images", "heat_map.png")

        for database in self.groups:
            invalid_link, aggregated_matrix = self.aggregate_single_group(database,
                                                                            band=band, feature=feature)
            aggregated_matrices.append((database, invalid_link, aggregated_matrix))

        plot_heatmap_diagram_windows(*tuple(aggregated_matrices), output_file)
