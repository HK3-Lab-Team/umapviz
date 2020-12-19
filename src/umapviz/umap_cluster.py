# encoding: utf-8

import logging
import types
from typing import Any, Dict, List, Tuple, Union

import numpy as np

from hdbscan import HDBSCAN
from trousse.dataframe_with_info import DataFrameWithInfo

from .umap_exp import UmapExperiment
from .umap_metrics import tanimoto_gower

logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")

TOOLS = (
    "hover,crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,"
    "undo,redo,reset,tap,save,box_select,poly_select,lasso_select,"
)


def _get_metric_kwds_umap_embeddings(embeddings: np.ndarray) -> Dict:
    feat_weights = np.ones(shape=(embeddings.shape[1],), dtype=np.float32)

    minv = np.quantile(embeddings, q=0.025, axis=0)
    maxv = np.quantile(embeddings, q=0.975, axis=0)

    metric_kwds = dict(
        min_vals=minv.astype(np.float64),
        max_vals=maxv.astype(np.float64),
        boolean_features=(),
        categorical_features=(),
        numerical_features=np.arange(embeddings.shape[1]),
        feature_weights=feat_weights.astype(np.float32),
    )
    return metric_kwds


class UmapClustering(UmapExperiment):
    """
    This class is to reduce the data dimensions in order to compute HDBSCAN.
    Dimensional Reduction is required to make HDBSCAN work better.
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
        super(UmapClustering, self).__init__(
            df_info=df_info,
            n_neighbors=n_neighbors,
            min_distance=min_distance,
            not_nan_percentage_threshold=not_nan_percentage_threshold,
            # this will be set only after clustering because HDBSCAN does not support
            # prediction on test set
            train_test_split_ratio=0.0,
            feature_to_color=feature_to_color,
            multi_marker_feats=multi_marker_feats,
            enc_value_to_str_map=enc_value_to_str_map,
            file_title_prefix=file_title_prefix,
            exclude_feat_list=exclude_feat_list,
            numer_feat_list=numer_feat_list,
            categ_feat_list=categ_feat_list,
            bool_feat_list=bool_feat_list,
            random_seed=random_seed,
            metric=metric,
            metric_kwds=metric_kwds,
            feature_weights=feature_weights,
            numer_feat_weight=numer_feat_weight,
            categ_feat_weight=categ_feat_weight,
            bool_feat_weight=bool_feat_weight,
            group_values_to_be_shown=group_values_to_be_shown,
            color_tuple=color_tuple,
            test_color_tuple=test_color_tuple,
            tools=tools,
            tooltip_feats=tooltip_feats,
            tooltips=tooltips,
            marker_fill_alpha=marker_fill_alpha,
            marker_size=marker_size,
            legend_location=legend_location,
        )
        # This will be set as 'train_test_split_ratio' attribute value only after clustering because
        # HDBSCAN does not support prediction on test set.
        # So the splitting (by method '_prepare_data') will occurr only after clustering
        self.train_test_split_ratio_cluster = train_test_split_ratio

        # Initialize internal variables

        # 'self._train_umap_data' and 'self._train_notna_full_features_df' will be used even
        # if 'train_test_split_ratio' = 0
        self._train_umap_data, self._train_notna_full_features_df = None, None
        self._test_umap_data, self._test_notna_full_features_df = None, None

        # These will store the embeddings generated for clustering, so they contain more dimensions
        self.reducer_cluster, self.cluster_data, self.test_embedding_cluster = (
            None,
            None,
            None,
        )
        self.hdbscan_fit, self.clustering_labels = None, None

    def clustering(
        self,
        min_cluster_size: int,
        use_umap_preprocessing: bool = True,
        umap_components: int = None,
        umap_min_distance: float = None,
        umap_n_neighbors: int = None,
        min_samples=None,
        cluster_selection_epsilon=0.0,
        cluster_metric=None,
        alpha=1.0,
        p=None,
        algorithm="best",
        leaf_size=40,
        cluster_metric_kwds: Dict[str, Any] = None,
    ):
        """
        This method is to identify cluster labels using HDBSCAN after a partial dimensionality
        reduction (UMAP) (in order to have better cluster computation).

        To select HDBSCAN algorithm parameters, see docs:
        https://hdbscan.readthedocs.io/en/latest/parameter_selection.html

        Parameters
        ----------
        min_cluster_size: int
            This is a parameter from hdbscan.HDBSCAN algorithm. From docs: "Set it to the smallest
            size grouping that you wish to consider a cluster"
        use_umap_preprocessing: bool
            Option to choose whether to reduce the data dimensionality by using UMAP (with metric specified
            during instantiation). If True, this may help in finding clusters because it makes samples closer
            (higher density helps HDBSCAN algorithm). On the other hand the sample density (distances)
            is not preserved by UMAP algorithm so if the option is set to True, HDBSCAN is not really
            detecting clusters: only UMAP is.
        umap_components: int
            UMAP embedding number of dimensions. Higher values lead to more dimensional embeddings and lower
            density, but the HDBSCAN algorithm has a more important role in identifying the clusters
            (otherwise UMAP is already separating data into clusters).
            Default set to 10.
        umap_min_distance: float
            Minimum distance used for UMAP algorithm.
        umap_n_neighbors: int
            Number of neighbors used for UMAP algorithm.
        min_samples
        cluster_selection_epsilon
        cluster_metric: function
            This is a function which calculates the distances between two vectors according to some
            metric_kwds that can be defined and passed. This is used by HDBSCAN algorithm to compute distances.
            This function should be written according to UMAP/HDBSCAN documentations (as a numba.njit function)
        alpha
        p
        algorithm
        leaf_size
        cluster_metric_kwds

        Returns
        -------
        self._clustering_labels: numpy.ndarray
            1D-Array with the cluster labels for each data sample
        """
        if use_umap_preprocessing:

            # OPTION 1. Using UMAP to reduce data dimensions for more effective HDBSCAN clustering results

            if umap_components is None:
                logger.error(
                    "No umap_components argument specified. This is required for using UMAP preprocessing"
                )
            if umap_min_distance is None:
                umap_min_distance = self.min_distance
            if umap_n_neighbors is None:
                umap_n_neighbors = self.n_neighbors

            if (
                self.cluster_data is None
                or self.cluster_data.shape[1] > umap_components
            ):
                self.cluster_data, _ = self.fit_transform(
                    n_components=umap_components,
                    n_neighbors=umap_n_neighbors,
                    min_distance=umap_min_distance,
                )

            if cluster_metric is None:
                # We can use 'tanimoto_gower' metric because the UMAP embeddings are all numeric
                # and they need to be normalized
                cluster_metric = tanimoto_gower
                cluster_metric_kwds = _get_metric_kwds_umap_embeddings(
                    self.cluster_data
                )
            elif cluster_metric_kwds is None:
                cluster_metric_kwds = {}

        else:
            # OPTION 2. Using HDBSCAN algorithm only to see differences between this and UMAP algorithm

            self.cluster_data = self._remove_nan()

            # These two are to make possible to specify a different metric and metric_kwds for clustering
            # (instead of using the same as UMAP)
            if cluster_metric is None:
                assert (
                    self.metric is not None
                ), "fit() method has not been called, so 'metric' attribute is None"
                cluster_metric = self.metric
            if cluster_metric_kwds is None:
                # If the attribute has not been defined, the metric_kwds are set for 'tanimoto_gower' custom metric
                if self.metric_kwds is None:
                    self.metric_kwds = self._get_tanimoto_gower_metric_kwds(
                        self.cluster_data
                    )

                cluster_metric_kwds = self.metric_kwds

            logging.info(
                f"HDBSCAN will use the following features: \n{self.data_summary}"
            )

        self.hdbscan_fit = HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric=cluster_metric,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
            alpha=alpha,
            p=p,
            algorithm=algorithm,
            leaf_size=leaf_size,
            **cluster_metric_kwds,
        ).fit(self.cluster_data)
        self.clustering_labels = self.hdbscan_fit.labels_
        logger.info(
            f"Clustering completed. Found: {np.unique(self.clustering_labels)} labels"
        )

        return self.clustering_labels

    def plot_cluster_labels(
        self, multi_marker_feats: Tuple = (), return_plot: bool = False
    ):
        """
        Plot bokeh.figure with UMAP embeddings according to the instance attributes.
        This is different from plot method because it will color the plot samples based on the cluster_labels
        (i.e. self._clustering_labels attribute) (not on feature_to_color).
        Different markers are assigned for each combination of the features in "multi_marker_feats" argument.

        The function will first validate the features in order to verify that the features marked in the plot
        (i.e. "feature_to_color" or "multi_feat_to_combine_partit_list") are not included in UMAP fitting

        Parameters
        ----------
        multi_marker_feats: Tuple

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
        if self.clustering_labels is None or self.cluster_data is None:
            logger.error(
                "The method .clustering() needs to be called before this method so that HDBSCAN "
                "algorithm is performed with appropriate arguments and clustering_labels are "
                "computed."
            )
        else:
            # We use the embedding obtained with previous UMAP (with "umap_component" dimensions)
            # as input for the new dimensionality reduction up to 2-Dimensions
            self._train_umap_data = self.cluster_data

        # Now we set this attribute so that 'self.fit' method will finally split train and test set
        # (after cluster labels computation)
        self.train_test_split_ratio = self.train_test_split_ratio_cluster

        # Fit UMAP in order to get 2D embedding. We need to use the N-dimensional data from
        # df_info because with tanimoto gower we cannot specify which features in
        # self.embedding_cluster are categorical, boolean, numeric,...
        self.embedding, self.test_embedding = self.fit_transform(
            n_components=2, repeat_fitting=True
        )

        self.multi_marker_feats = multi_marker_feats
        self.feature_to_color = "clustering_label"
        self._train_notna_full_features_df.df[
            "clustering_label"
        ] = self.clustering_labels
        self.plot(return_plot=return_plot)
