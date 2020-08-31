import logging
import types
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Tuple, Union

import bokeh.plotting as bk
import numpy as np
import pandas as pd
import sklearn
from pd_extras.dataframe_with_info import (
    DataFrameWithInfo,
    copy_df_info_with_new_df,
)

from .umap_functions import get_umap_embeddings, plot_umap

logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")

TOOLS = (
    "hover,crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,"
    "undo,redo,reset,tap,save,box_select,poly_select,lasso_select,"
)
TOOLTIPS = (
    ("index", "$index"),
    ("(x,y)", "($x, $y)"),
    ("study_UID", "@study_uid"),
)


@dataclass
class FeatureCategory:
    """
    This gathers the info about a set of features of the same kind
    """

    name_f: str
    type_f: Any
    list_f: Tuple[str]
    weight_f: float = None

    def __len__(self):
        return len(self.list_f)


@dataclass
class FeatureCategoriesList:
    """
    This DataClass gathers the dataframe feature names by type.
    It cannot be replaced by a list because I need to get those specific attributes
    """

    numeric_f: FeatureCategory
    categ_f: FeatureCategory
    bool_f: FeatureCategory

    def __iter__(self) -> Iterator[FeatureCategory]:
        """
        This method is to iterate over the FeatureClass attributes of the class, always
        maintaining the same order so every step or list deriving from these is created with the
        same order.

        Returns
        -------
        attr, value: Tuple[str, FeatureCategory]
            Iterator that yields an ordered list of tuples (key, value), where the 'key' is the name
            of the attribute of this class, and 'value' is the corresponding FeatureClass instance
        """
        sorted_tuple_attr = sorted(self.__dict__.items(), key=lambda item: item[0])
        for _, value in sorted_tuple_attr:
            yield value


class UmapExperiment:
    """
    This class gathers all the settings required to build UMAP experiment with optional clustering and plot it.
    """

    def __init__(
        self,
        df_info: DataFrameWithInfo,
        n_neighbors: int,
        min_distance: float,
        not_nan_percentage_threshold: float,
        train_test_split_ratio: float,
        feature_to_color: str,
        multi_marker_feats: Tuple,
        enc_value_to_str_map: Dict[str, Dict],
        file_title_prefix: str,
        exclude_feat_list: Tuple = (),
        numer_feat_list: Tuple = None,
        categ_feat_list: Tuple = None,
        bool_feat_list: Tuple = None,
        random_seed: int = 42,
        metric: Union[str, types.FunctionType] = "euclidean",
        metric_kwds: Dict = None,
        feature_weights: Tuple[float] = (),
        numer_feat_weight: float = 1.0,
        categ_feat_weight: float = 1.0,
        bool_feat_weight: float = 1.0,
        group_values_to_be_shown: Tuple[str, Tuple[Tuple[str]]] = None,
        color_tuple: Tuple = (),
        test_color_tuple: Tuple = (),
        tools: str = TOOLS,
        tooltip_feats: Tuple = None,
        tooltips: List = None,
        marker_fill_alpha: float = 0.2,
        marker_size: int = 6,
        legend_location: str = "bottom_left",
    ):
        """
        This constructor gathers all the settings and methods required to build UMAP experiment
        with optional clustering and plot it.

        Parameters
        ----------
        df_info: DataFrameWithInfo
             Full DataFrameWithInfo which contains all the data and info to be used for UMAP and clustering
        n_neighbors: int
            Number of neighbors to consider in UMAP algorithm. It depends on the number of samples available
        min_distance: float
            Minimum distance to consider in UMAP algorithm. It depends on which scale you want to see
            patterns/structures
        random_seed: int
            Numeric value to fix the initial random state in order to get repeatable results from UMAP algorithm
        exclude_feat_list: Tuple
            Tuple of the feature names (df_info columns) that you want to exclude. Default set to ()
        numer_feat_list: Tuple
            Tuple of the feature names (df_info columns) containing numerical values only. If None,
            'df_info.column_list_by_type.numerical_cols' attribute will be used. Default set to None
        categ_feat_list: Tuple
            Tuple of the feature names (df_info columns) containing categorical values only. If None,
            'df_info.column_list_by_type.categorical_cols' attribute will be used. Default set to None
        bool_feat_list: Tuple
            Tuple of the feature names (df_info columns) containing boolean values only. If None,
            'df_info.column_list_by_type.bool_cols' attribute will be used. Default set to None
        not_nan_percentage_threshold: float
            Every feature, which will contain less than this percentage of not-NaN, will not be considered
            in the returned data used to compute embeddings by UMAP
        train_test_split_ratio: float
            This specifies the ratio between training and test set. It corresponds to the size of test
            set divided by the size of the whole set
        feature_to_color: str
            This will be the df_info column (str) that defines the color of the series
            according to its values (the colors can be set by "color_tuple" argument)
        multi_marker_feats: Tuple
            These column_names/features will be used to define different series of points
            (described and listed in legend). Their unique values combination will define
            different partitions that will be distinguishable by markers
        enc_value_to_str_map: Dict[Dict]
            This is a map to connect the encoded values to the original ones. It will be a Dict of Dicts
            because the first level keys identify the features with encoded values, and the second level
            dict is the actual map
        file_title_prefix: str
            This string will be used as prefix in the name of the exported bokeh plot file and in the
            plot title
        metric: function
            This is a function which calculates the distances between two vectors according to some
            metric_kwds that can be defined and passed. This function should be written according to UMAP
            documentations and as a numba.njit function
        metric_kwds:
            Keywords used by the metric that will be used for clustering or UMAP algorithm computations.
            If not set, the metric_kwds will be set for 'tanimoto_gower' custom metric.
        feature_weights: Tuple[Float]
            This tuple will be used in metric to give different weights for every feature.
            This can be None and it can be set later or, if not provided, weights per category (e.g.
            numer_feat_weight) will be used.
        numer_feat_weight: float
            Weight to be used in metric for every numerical feature. Default set to 1
        categ_feat_weight: float
            Weight to be used in metric for every categorical feature. Default set to 1
        bool_feat_weight: float
            Weight to be used in metric for every boolean feature. Default set to 1
        group_values_to_be_shown: Tuple[str, Tuple]
            This is required if you want to select only a part of DataFrame as a Serie of grey
            points (as background), and the other part will be split in partitions according
            to the other arguments. If you want the DataFrame to be considered all together
            with no grey/background points, set this to None (default value).
            This is a tuple where the first element is the name of the column which you want to set a
            condition on. The second element is a Tuple containing all the values from that column that
            have to be included in the plot.
            The rows that have other values for that column will be plotted as grey points.
            E.g. ('BREED', (('LABRADOR RETRIEVER',), ('MONGREL',))) -> Two plots will be created where these
              two breeds will separately be plotted as colored points, while the other breeds will be
              plotted as grey points
        color_tuple
        test_color_tuple
        tools
        tooltip_feats:
        tooltips: Tuple[Tuple[str, str]]
            If not specified, the tooltips will be defined by using the name of the features in
            'tooltip_feats' argument. If these names are not clear, it can be useful to manually
            specify the tooltip tuples by using this argument
        marker_fill_alpha: float
            Float number in range [0, 1] that describes the transparency of the marker fill_color
            (1 is opaque, 0 transparent)
        marker_size: int
            Size of the plotted markers
        legend_location: str
            Location of the legend in the plot. Default set to 'bottom_left'
        """
        self.df_info = df_info
        self.n_neighbors = n_neighbors
        self.min_distance = min_distance

        self.exclude_feats = FeatureCategory(
            name_f="exclude",
            type_f=None,
            list_f=exclude_feat_list,
        )

        self.cols_by_type = df_info.column_list_by_type
        bool_feat_list = (
            self.cols_by_type.bool_cols if bool_feat_list is None else bool_feat_list
        )
        numer_feat_list = (
            self.cols_by_type.numerical_cols
            if numer_feat_list is None
            else numer_feat_list
        )
        categ_feat_list = (
            self.cols_by_type.num_categorical_cols
            if categ_feat_list is None
            else categ_feat_list
        )

        bool_f = FeatureCategory(
            name_f="bool",
            type_f=np.int,
            list_f=tuple(set(bool_feat_list) - set(exclude_feat_list)),
            weight_f=float(bool_feat_weight),
        )
        numeric_f = FeatureCategory(
            name_f="numeric",
            type_f=np.float,
            list_f=tuple(set(numer_feat_list) - set(exclude_feat_list)),
            weight_f=float(numer_feat_weight),
        )
        categ_f = FeatureCategory(
            name_f="categorical",
            type_f=np.int,
            list_f=tuple(set(categ_feat_list) - set(exclude_feat_list)),
            weight_f=float(categ_feat_weight),
        )
        # Unlike the other attributes, the features list should not be changed once
        # UmapExperiment is instantiated.
        self._feat_list = FeatureCategoriesList(
            numeric_f=numeric_f, categ_f=categ_f, bool_f=bool_f
        )
        self._set_df_feature_types()

        self.not_nan_percentage_threshold = not_nan_percentage_threshold
        self.train_test_split_ratio = train_test_split_ratio

        self.feature_to_color = feature_to_color
        self.multi_marker_feats = multi_marker_feats
        self.enc_value_to_str_map = enc_value_to_str_map
        self.file_title_prefix = file_title_prefix

        # This can be None and it can be set later. The actual feature weights
        # will be set based on these. If not present weights per category (e.g. numer_feat_weight) will be used
        self.feature_weights = feature_weights
        self.random_seed = random_seed
        self.metric = metric
        self.metric_kwds = metric_kwds

        self.group_values_to_be_shown = group_values_to_be_shown
        self.color_tuple = color_tuple
        self.test_color_tuple = test_color_tuple
        self.tools = tools
        self.tooltip_feats = tooltip_feats
        self.tooltips = tooltips
        self.legend_location = legend_location
        self.marker_fill_alpha = marker_fill_alpha
        self.marker_size = marker_size

        # Initialize internal variables

        # 'self._train_umap_data' and 'self._train_notna_full_features_df' will be used even
        # if 'train_test_split_ratio' = 0
        self._train_umap_data, self._train_notna_full_features_df = None, None
        self._test_umap_data, self._test_notna_full_features_df = None, None

        self.reducer, self.embedding, self.test_embedding = None, None, None

    def _set_df_feature_types(self):
        # Set the features to appropriate dtype and check if they are present in df_info
        for feat_class in self._feat_list:
            for col in feat_class.list_f:
                try:
                    self.df_info.df[col] = self.df_info.df[col].astype(
                        feat_class.type_f
                    )
                except KeyError:
                    logger.error(
                        f"Column {col} from {feat_class.name_f} columns is not present in df_info"
                    )

    def _remove_nan(self) -> pd.DataFrame:
        """
        For each feature analyze the number of NaN and keep the ones that have a number
        of not-NaN higher than 'self.not_nan_percentage_threshold'.

        The features with too many NaN will not be considered in UMAP plot and they will be removed from
        self.feat_list attributes (list of features by category)

        Then remove the rows with any NaN left, because UMAP will not handle any NaN value
        (distance would not be defined)
        Returns
        -------
        umap_data: pandas.DataFrame
            This is a slice of 'self.df_info.df' containing only the features to be used for UMAP fitting
            (with no NaN and only numerical/categorical/bool values)
        """

        # Prepare a secondary pandas.DataFrame only with the features that
        # will be used (from 'self.feat_list') and without any NaN
        umap_data = pd.DataFrame()
        many_nan_features = []
        for feat_list in self._feat_list:
            feat_list_notnan = []
            # For each feature analyze the number of NaN and keep the ones that have a number
            # of not-NaN higher than 'self.not_nan_percentage_threshold'
            for f in feat_list.list_f:
                try:
                    not_na_count = sum(self.df_info.df[f].notna())
                    if (
                        not_na_count
                        > self.not_nan_percentage_threshold * self.df_info.df.shape[0]
                    ):
                        feat_list_notnan.append(f)
                    else:
                        many_nan_features.append(f)
                except KeyError:
                    logger.warning(
                        f"The {feat_list.name_f} feature {f} is not present in 'df_info.df' attribute."
                        f"So it will not be considered in further analysis"
                    )
            # Concatenate the features with few NaN (this way the umap_data will have the same order as
            # given by __iter__ method from FeatureList class
            umap_data = pd.concat(
                [umap_data, self.df_info.df[feat_list_notnan]], axis=1
            )
            # Replace the feature list with the ones we selected (less NaN)
            feat_list.list_f = feat_list_notnan
        logging.info(
            f"The features that have too high number of Nan (and will not be considered "
            f"in UMAP) are: {many_nan_features}"
        )

        # Select only the rows based on "feat_list_notnan", dropping the ones containing NaN
        umap_data = umap_data.dropna(axis=0)

        return umap_data

    def _train_test_split(
        self, umap_data: pd.DataFrame, test_over_set_ratio: float
    ) -> Tuple[DataFrameWithInfo, ...]:
        """
        It splits the umap_data argument into train and test.
        It also returns the train and test set with all the features (even the ones that are not used
        in UMAP fit computation. These will be useful when plotting the UMAP

        Parameters
        ----------
        umap_data
        test_over_set_ratio: float
            Percentage between 0 and 1 for test set over full set ratio

        Returns
        -------
            train_umap_data: DataFrameWithInfo
            train_notna_full_features_df: DataFrameWithInfo
                This is a DataFrameWithInfo with the same samples as train_umap_data, but with more features
                because the ones with many NaN have not been dropped
            test_umap_data: DataFrameWithInfo
            test_notna_full_features_df: DataFrameWithInfo

        """
        if self.train_test_split_ratio != 0:
            # SPLIT into training and test set
            train_umap_data, test_umap_data = sklearn.model_selection.train_test_split(
                umap_data, test_size=test_over_set_ratio
            )
            # Use train and test to create DataFrameWithInfo instances
            train_umap_data = copy_df_info_with_new_df(
                df_info=self.df_info, new_pandas_df=train_umap_data
            )
            test_umap_data = copy_df_info_with_new_df(
                df_info=self.df_info, new_pandas_df=test_umap_data
            )
            # Select only the rows in umap data from the df
            train_notna_full_features_df = copy_df_info_with_new_df(
                df_info=self.df_info,
                new_pandas_df=self.df_info.df.iloc[train_umap_data.df.index],
            )
            test_notna_full_features_df = copy_df_info_with_new_df(
                df_info=self.df_info,
                new_pandas_df=self.df_info.df.iloc[test_umap_data.df.index],
            )
            # Reset index (so it is easier for further analysis)
            train_notna_full_features_df.df = (
                train_notna_full_features_df.df.reset_index(drop=True)
            )
            test_notna_full_features_df.df = test_notna_full_features_df.df.reset_index(
                drop=True
            )

            return (
                train_umap_data,
                train_notna_full_features_df,
                test_umap_data,
                test_notna_full_features_df,
            )
        else:
            umap_data_df = copy_df_info_with_new_df(
                df_info=self.df_info, new_pandas_df=umap_data
            )
            notna_full_features_df = copy_df_info_with_new_df(
                df_info=self.df_info,
                new_pandas_df=self.df_info.df.iloc[umap_data.index],
            )
            notna_full_features_df.df = notna_full_features_df.df.reset_index(drop=True)
            return umap_data_df, notna_full_features_df, None, None

    def _validate_features(self):
        """
        The function will validate the features in order to verify that the features marked in the plot
        (i.e. "feature_to_color" or "multi_feat_to_combine_partit_list") are not included in UMAP fitting
        """
        plotted_feats = list(self.multi_marker_feats) + [
            self.feature_to_color,
        ]
        for feat in self._feat_list:
            # Check for every features that will be shown in plot
            for plotted_feat in plotted_feats:
                if plotted_feat in feat.list_f:
                    logger.warning(
                        f"{plotted_feat} is included among the features of UMAP fitting. "
                        f"This leads to data leakage. Be careful"
                    )

    def _prepare_umap_data(self) -> Tuple[DataFrameWithInfo, ...]:
        """
        PREPARING DATA FOR UMAP
        Selecting:
        - only the features in col_list of df (or med_exam_col_list if col_list is not provided)
        - only the features with a number of not_NaN higher than not_nan_percentage_threshold
        - only the rows with no NaN in the remaining features
        Splitting into train-test set if test_over_set_ratio != 0

        Returns
        -------

        """
        umap_data = self._remove_nan()

        return self._train_test_split(umap_data, self.train_test_split_ratio)

    def _get_feat_weights(self) -> np.array:
        """
        If feature weights is not provided before, this creates the feature weights
        (for metric) repeating the weights specified for each feature category.

        Returns
        -------
        self.feature_weights: numpy.array
            It contains one weight for each feature
        """
        if self.feature_weights == ():
            feat_weight_list = []
            for feat_class in self._feat_list:
                class_weights = np.repeat(feat_class.weight_f, len(feat_class))
                feat_weight_list.append(class_weights)

            self.feature_weights = np.array(
                np.concatenate(tuple(feat_weight_list)),
            ).astype(np.float32)

        return self.feature_weights

    def _get_feature_ids_by_category(self) -> Tuple[np.array, ...]:

        bool_end = len(self._feat_list.bool_f)
        cat_end = len(self._feat_list.categ_f)
        num_end = len(self._feat_list.numeric_f)

        bool_feat_metric = (
            np.array(np.arange(bool_end), np.int64) if bool_end > 0 else None
        )
        cat_feat_metric = (
            np.array(np.arange(bool_end, bool_end + cat_end), np.int64)
            if cat_end > 0
            else None
        )
        num_feat_metric = (
            np.array(
                np.arange(bool_end + cat_end, bool_end + cat_end + num_end), np.int64
            )
            if num_end > 0
            else None
        )

        return bool_feat_metric, cat_feat_metric, num_feat_metric

    def _get_tanimoto_gower_metric_kwds(self, df_data: pd.DataFrame) -> Dict:

        feat_weights = self._get_feat_weights()
        (
            bool_feat_metric,
            cat_feat_metric,
            num_feat_metric,
        ) = self._get_feature_ids_by_category()

        minv = np.quantile(df_data.values, q=0.025, axis=0)
        maxv = np.quantile(df_data.values, q=0.975, axis=0)

        metric_kwds = dict(
            min_vals=minv.astype(np.float64),
            max_vals=maxv.astype(np.float64),
            boolean_features=bool_feat_metric,
            categorical_features=cat_feat_metric,
            numerical_features=num_feat_metric,
            feature_weights=feat_weights.astype(np.float32),
        )
        return metric_kwds

    def fit_transform(
        self,
        n_components=2,
        repeat_fitting: bool = False,
        min_distance: float = None,
        n_neighbors: int = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        This will prepare the data (by splitting and removing NaN) and the metric_kwds.
        Then these will be passed to UMAP algorithm for fitting on train set. The train and test set
        embeddings will be computed and returned

        Parameters
        ----------
        n_components: int
            Embedding number of dimensions. E.g. For 2D plot, select 2. Default set to 2.
        repeat_fitting: bool
            This option is to choose to repeat the UMAP fitting using the selected umap_components
        min_distance: float
        n_neighbors: int

        Returns
        -------
        embedding: np.ndarray
            UMAP embeddings of the train samples
        test_embedding: np.ndarray
            UMAP embeddings of the test samples (computed by using the UMAP fit on train set)
        """
        if (
            repeat_fitting
            or self._train_umap_data is None
            or self._train_notna_full_features_df is None
        ):
            (
                self._train_umap_data,
                self._train_notna_full_features_df,
                self._test_umap_data,
                self._test_notna_full_features_df,
            ) = self._prepare_umap_data()

        # If the attribute has not been defined, the metric_kwds are set for 'tanimoto_gower' custom metric
        if self.metric_kwds is None:
            self.metric_kwds = self._get_tanimoto_gower_metric_kwds(
                self._train_umap_data.df
            )

        logging.info(f"UMAP will use the following features: \n{self.data_summary}")

        if min_distance is None:
            min_distance = self.min_distance
        if n_neighbors is None:
            n_neighbors = self.n_neighbors

        self.reducer, self.embedding, self.test_embedding = get_umap_embeddings(
            train_df_info=self._train_umap_data,
            n_neighbors=n_neighbors,
            min_dist=min_distance,
            n_components=n_components,
            random_state=self.random_seed,
            metric=self.metric,
            metric_kwds=self.metric_kwds,
            test_df_info=self._test_umap_data,
        )
        return self.embedding, self.test_embedding

    def plot(self, return_plot: bool = False) -> Union[bk.figure, None]:
        """
        Plot bokeh.figure with UMAP embeddings according to the instance attributes

        The function will first validate the features in order to verify that the features marked in the plot
        (i.e. "feature_to_color" or "multi_feat_to_combine_partit_list") are not included in UMAP fitting

        Parameters
        ----------
        return_plot: bool
            If set to True, no plot will be shown, but a bokeh.figure instance 'bk_plot' will be returned.
            If set to False, the plot will be shown

        Returns
        -------
        bk_plot: bokeh.plotting.figure
            This is None if 'return_plot' argument is set to False. Otherwise an instance of bokeh.plotting.figure
            will be returned. This can be shown by using bokeh.plotting.show(bk_plot),
            or it can be put into a grid.
        """
        if self.embedding is None or self.embedding.shape[1] != 2:
            self.fit_transform(n_components=2)

        self._validate_features()

        bk_plot = plot_umap(
            embedding=self.embedding,
            df_full_feat=self._train_notna_full_features_df,
            umap_params_str=f"{self.n_neighbors}_{self.min_distance}",
            group_values_to_be_shown=self.group_values_to_be_shown,
            feature_to_color=self.feature_to_color,
            multi_marker_feats=self.multi_marker_feats,
            enc_value_to_str_map=self.enc_value_to_str_map,
            color_tuple=self.color_tuple,
            test_embedding=self.test_embedding,
            test_df_full_feat=self._test_notna_full_features_df,
            test_color_tuple=self.test_color_tuple,
            legend_location=self.legend_location,
            filename_title_prefix=self.file_title_prefix,
            tooltips=self.tooltips,
            tooltip_feats=self.tooltip_feats,
            tools=self.tools,
            marker_fill_alpha=self.marker_fill_alpha,
            marker_size=self.marker_size,
            return_plot=return_plot,
        )
        return bk_plot

    @property
    def data_summary(self) -> str:
        """ Returns a list of features by category (numeric, categoric, ..)"""
        data_summary = ""
        for id, feat in enumerate(self._feat_list):
            if len(feat) == 0:
                feat_list_str = "0 feat."
            else:
                feat_list_str = (
                    f"{len(feat)} feat.\n\t[" + " - ".join(feat.list_f) + "]"
                )
            data_summary += f"{id}.\t{str(feat.name_f)}:\t{feat_list_str}\n"
        return data_summary
