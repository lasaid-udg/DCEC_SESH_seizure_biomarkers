import os
import numpy
import pandas
import logging
from pandasql import sqldf
from . import settings
from .visuals import plot_univariate_intra_bar_chart, plot_univariate_intra_dist_chart, \
                     plot_topoplot_features_time, \
                     plot_univariate_inter_bar_chart, plot_univariate_inter_dist_chart, \
                     plot_univariate_ml_bar_chart, plot_univariate_inter_bar_chart_psd, \
                     plot_univariate_inter_dist_chart_psd, plot_univariate_intra_bar_chart_psd, \
                     plot_univariate_intra_dist_chart_psd, plot_univariate_ml_bar_chart_psd


class IntraUnivariateFeatureAnalyzer():

    def __init__(self, feature_name: str):
        """
        :param feature_name: name of the feature file
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
        :param feature: full path to the feature file
        """
        self._raw_features = pandas.read_csv(filename)

    def univariate_zone_bar_chart(self, zone: str, seizure_type: str) -> None:
        """
        Separate data into groups for bar evaluation
        :param region: name of the brain zone
        :param seizure_type: type of seizure
        """
        features = self.raw_features
        if seizure_type is not "unknown":
            features = features[features.seizure_type == seizure_type]

        channel_groups = self.univariate_channels_groups[zone]
        features = features[features.channel.isin(channel_groups)]

        for stage in settings["intra_windows_categories"]:
                features.loc[((features.time_point == stage[1]) & (features.seizure_stage == stage[0])), "time_point"] = stage[3]

        delta_features = features[features.band == "delta"]
        theta_features = features[features.band == "theta"]
        alpha_features = features[features.band == "alpha"]
        beta_features = features[features.band == "beta"]
        all_features = features[features.band == "all"]

        base_path = os.getenv("BIOMARKERS_PROJECT_HOME")
        output_file = os.path.join(base_path, "images", self.DATASET, f"bar_intra_{self.feature_name}_{zone}_{seizure_type}.png")
        if self.feature_name != "power_spectral_density":
            plot_univariate_intra_bar_chart(delta_features, theta_features, alpha_features, beta_features, all_features, output_file)
        else:
            plot_univariate_intra_bar_chart_psd(delta_features, theta_features, alpha_features, beta_features, output_file)

    def univariate_zone_dist_chart(self, zone: str, seizure_type: str) -> None:
        """
        Separate data into groups for distribution chart
        :param region: name of the brain zone
        :param seizure_type: type of seizure
        """
        features = self.raw_features
        if seizure_type is not "unknown":
            features = features[features.seizure_type == seizure_type]

        channel_groups = self.univariate_channels_groups[zone]
        features = features[features.channel.isin(channel_groups)]
        features["Time point"] = "unknown"
        for stage in settings["intra_windows_categories"]:
                features.loc[((features.time_point == stage[1]) & (features.seizure_stage == stage[0])), "Time point"] = stage[3]

        delta_features = features[features.band == "delta"]
        theta_features = features[features.band == "theta"]
        alpha_features = features[features.band == "alpha"]
        beta_features = features[features.band == "beta"]
        all_features = features[features.band == "all"]

        base_path = os.getenv("BIOMARKERS_PROJECT_HOME")
        output_file = os.path.join(base_path, "images", self.DATASET, f"dist_intra_{self.feature_name}_{zone}_{seizure_type}.png")
        if self.feature_name != "power_spectral_density":
            plot_univariate_intra_dist_chart(delta_features, theta_features, alpha_features, beta_features, all_features, output_file)
        else:
            plot_univariate_intra_dist_chart_psd(delta_features, theta_features, alpha_features, beta_features, output_file)


    def univariate_topo_plot_average(self, seizure_type: int, band: int) -> None:
        """
        Separate data into groups for an averaged topographic chart across ictal events
        :param band: any of [delta, theta, alpha, beta]
        :param seizure_type: type of seizure
        """
        features = self.raw_features

        if seizure_type is not "unknown":
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
        output_file = os.path.join(base_path, "images", self.DATASET, f"topo_time_average_{self.feature_name}_{seizure_type}_{band}.png")

        if self.DATASET == "chb-mit":
            plot_topoplot_features_time(*tuple(groups), output_file, is_monopolar=False)
        else:
            plot_topoplot_features_time(*tuple(groups), output_file)

    def univariate_topo_plot_individual(self, seizure_number: int, band: int) -> None:
        """
        Separate data into groups for topographic chart for an ictal event
        :param seizure_number: index of the seizure
        :param band: any of [delta, theta, alpha, beta]
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
        output_file = os.path.join(base_path, "images", self.DATASET, f"topo_time_{subject}_{self.feature_name}_{band}.png")

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
        if seizure_type is not "unknown":
            features = features[features.seizure_type == seizure_type]

        channel_groups = self.univariate_channels_groups[zone]
        features = features[features.channel.isin(channel_groups)]
        for band in ["delta", "theta", "alpha", "beta", "all"]:
            for stage in settings["intra_windows_categories"]:
                values = features[(features.band == band) & (features.seizure_stage == stage[0]) & (features.time_point == stage[1])]
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
        if seizure_type is not "unknown":
            features = features[features.seizure_type == seizure_type]
        channel_groups = self.univariate_channels_groups[zone]
        features = features[features.channel.isin(channel_groups)]

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

    def processed_data_for_naive_bayes(self, seizure_type: str) -> list:
        """
        Separate data into groups for friedman evaluation
        :param zone: name of the brain zone
        :param seizure_type: type of seizure
        """
        groups = []
        features = self.raw_features
        if seizure_type is not "unknown":
            features = features[features.seizure_type == seizure_type]

        for band in ["delta", "theta", "alpha", "beta", "all"]:
            categories = []
            for stage in settings["intra_windows_categories"]:
                single_group = []
                for channel in self.channels:
                    channel_values = features[(features.band == band) & 
                                              (features.seizure_stage == stage[0]) & 
                                              (features.time_point == stage[1]) &
                                              (features.channel == channel)]
                    channel_values = channel_values.value.tolist()
                    single_group.append(channel_values)
                categories.append([stage[3], numpy.array(single_group)])
            groups.append([band, categories])

        return groups

class IntraUnivariateChbAnalyzer(IntraUnivariateFeatureAnalyzer):

    DATASET = "chb-mit"

    def __init__(self, feature: str):
        """
        :param feature: name of the feature file
        :param feature_type: any of [univariate, bivariate]
        """
        super().__init__(feature)

    def hemisfere_bar_chart(self, hemisfere: str, seizure_type: str) -> None:
        """
        :param hemisfere: any of [left, right]
        """
        self.univariate_zone_bar_chart(hemisfere, seizure_type)

    def hemisfere_dist_chart(self, hemisfere: str, seizure_type: str) -> None:
        """
        :param hemisfere: any of [left, right]
        """
        self.univariate_zone_dist_chart(hemisfere, seizure_type)


class IntraUnivariateSienaAnalyzer(IntraUnivariateFeatureAnalyzer):

    DATASET = "siena"

    def __init__(self, feature: str):
        """
        :param feature: name of the feature file
        :param feature_type: any of [univariate, bivariate]
        """
        super().__init__(feature)

    def region_bar_chart(self, region: str, seizure_type: str) -> None:
        """
        :param region: any of [temporal, parietal, frontal, occipital]
        """
        self.univariate_zone_bar_chart(region, seizure_type)
    
    def region_dist_chart(self, region: str, seizure_type: str) -> None:
        """
        :param region: any of [temporal, parietal, frontal, occipital]
        """
        self.univariate_zone_dist_chart(region, seizure_type)


class IntraUnivariateTuszAnalyzer(IntraUnivariateFeatureAnalyzer):

    DATASET = "tusz"

    def __init__(self, feature: str):
        """
        :param feature: name of the feature file
        :param feature_type: any of [univariate, bivariate]
        """
        super().__init__(feature)

    def region_bar_chart(self, region: str, seizure_type: str) -> None:
        """
        :param region: any of [temporal, parietal, frontal, occipital]
        """
        self.univariate_zone_bar_chart(region, seizure_type)

    def region_dist_chart(self, region: str, seizure_type: str) -> None:
        """
        :param region: any of [temporal, parietal, frontal, occipital]
        """
        self.univariate_zone_dist_chart(region, seizure_type)


class IntraUnivariateTuepAnalyzer(IntraUnivariateFeatureAnalyzer):

    DATASET = "tuep"

    def __init__(self, feature: str):
        """
        :param feature: name of the feature file
        :param feature_type: any of [univariate, bivariate]
        """
        super().__init__(feature)

    def processed_data_for_lilliefors(self, zone: str) -> None:
        """
        Separate data into groups for normality evaluation
        :param zone: name of the brain zone
        :param seizure_type: type of seizure
        """
        groups = []
        features = self.raw_features

        channel_groups = self.univariate_channels_groups[zone]
        features = features[features.channel.isin(channel_groups)]
        for band in ["delta", "theta", "alpha", "beta", "all"]:
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
        for band in ["delta", "theta", "alpha", "beta", "all"]:
            preictal_band_array = preictal_features[preictal_features.band == band][["band", "feature", "value"]]
            preictal_band_array["Group"] = "Preictal"
            ictal_band_array = ictal_features[ictal_features.band == band][["band", "feature", "value"]]
            ictal_band_array["Group"] = "Ictal"
            healthy_band_array = healthy_features[healthy_features.band == band][["band", "feature", "value"]]
            healthy_band_array["Group"] = "Healthy"
            merged_band_array = pandas.concat([preictal_band_array, ictal_band_array, healthy_band_array],ignore_index=True)
            bands.append(merged_band_array)

        base_path = os.getenv("BIOMARKERS_PROJECT_HOME")
        output_file = os.path.join(base_path, "images", self.DATASET, f"bar_inter_{self.feature_name}_{zone}.png")
        if self.feature_name != "power_spectral_density":
            plot_univariate_inter_bar_chart(*bands, output_file)
        else:
            plot_univariate_inter_bar_chart_psd(*bands, output_file)

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
        for band in ["delta", "theta", "alpha", "beta", "all"]:
            preictal_band_array = preictal_features[preictal_features.band == band][["band", "feature", "value"]]
            preictal_band_array["Group"] = "Preictal"
            ictal_band_array = ictal_features[ictal_features.band == band][["band", "feature", "value"]]
            ictal_band_array["Group"] = "Ictal"
            healthy_band_array = healthy_features[healthy_features.band == band][["band", "feature", "value"]]
            healthy_band_array["Group"] = "Healthy"
            merged_band_array = pandas.concat([preictal_band_array, ictal_band_array, healthy_band_array],ignore_index=True)
            bands.append(merged_band_array)

        base_path = os.getenv("BIOMARKERS_PROJECT_HOME")
        output_file = os.path.join(base_path, "images", self.DATASET, f"dist_inter_{self.feature_name}_{zone}.png")
        if self.feature_name != "power_spectral_density":
            plot_univariate_inter_dist_chart(*bands, output_file)
        else:
            plot_univariate_inter_dist_chart_psd(*bands, output_file)

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
        for band in ["delta", "theta", "alpha", "beta", "all"]:
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


class MlUnivariateFeatureAnalyzer():

    def __init__(self, dataset: str, feature_name: str):
        """
        :param dataset: any of [chb-mit, siena, tusz]
        :param feature_name: name of the feature file
        """
        base_path = os.getenv("BIOMARKERS_PROJECT_HOME")
        filename = f"naive_{dataset}_intra_{feature_name}.csv"
        patients_base_path = os.path.join(base_path, "reports", filename)
        self.accuracy_values = pandas.read_csv(patients_base_path)
        self.dataset = dataset
        self.feature_name = feature_name

    def ml_bar_chart(self, seizure_type: str) -> None:
        """
        Plot the accuracy of the Naive Bayes mode
        :param seizure_type: type of seizure
        """
        accuracy_values = self.accuracy_values
        if seizure_type is not "unknown":
            accuracy_values = accuracy_values[accuracy_values.seizure_type == seizure_type]

        groups = []
        for band in ["delta", "theta", "alpha", "beta", "all"]:
            group = accuracy_values[accuracy_values.band == band]
            groups.append(group)

        base_path = os.getenv("BIOMARKERS_PROJECT_HOME")
        if self.dataset == "chb":
            self.dataset = "chb-mit"
        output_file = os.path.join(base_path, "images", self.dataset, f"ml_inter_{seizure_type}_{self.feature_name}.png")
        if self.feature_name != "power_spectral_density":
            plot_univariate_ml_bar_chart(*groups, output_file)
        else:
            plot_univariate_ml_bar_chart_psd(*groups, output_file)