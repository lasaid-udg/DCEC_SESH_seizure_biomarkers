import os
import pandas
from . import settings
from .visuals import plot_univariate_bar_chart, plot_univariate_dist_chart


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
                features.loc[((features.time_point == stage[1]) & (features.seizure_stage == stage[0])), "time_point"] = stage[2]

        delta_features = features[features.band == "delta"]
        theta_features = features[features.band == "theta"]
        alpha_features = features[features.band == "alpha"]
        beta_features = features[features.band == "beta"]

        base_path = os.getenv("BIOMARKERS_PROJECT_HOME")
        output_file = os.path.join(base_path, "images", self.DATASET, f"bar_intra_{self.feature_name}_{zone}_{seizure_type}.png")
        plot_univariate_bar_chart(delta_features, theta_features, alpha_features, beta_features, output_file)

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

        base_path = os.getenv("BIOMARKERS_PROJECT_HOME")
        output_file = os.path.join(base_path, "images", self.DATASET, f"dist_intra_{self.feature_name}_{zone}_{seizure_type}.png")
        plot_univariate_dist_chart(delta_features, theta_features, alpha_features, beta_features, output_file)

    def processed_data_for_lilliefors(self, zone: str, seizure_type: str) -> None:
        """
        Separate data into groups for normality evaluation
        :param region: name of the brain zone
        :param seizure_type: type of seizure
        """
        groups = []
        features = self.raw_features
        if seizure_type is not "unknown":
            features = features[features.seizure_type == seizure_type]

        channel_groups = self.univariate_channels_groups[zone]
        features = features[features.channel.isin(channel_groups)]
        for band in ["delta", "theta", "alpha", "beta"]:
            for stage in settings["intra_windows_categories"]:
                values = features[(features.band == band) & (features.seizure_stage == stage[0]) & (features.time_point == stage[1])]
                groups.append([band, stage[1], values.value])

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
