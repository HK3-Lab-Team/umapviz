import logging
from typing import Dict, Tuple, Union

import bokeh
import bokeh.plotting as bk
import numpy as np
import sklearn.model_selection
import umap
from bokeh import palettes
from pd_extras.dataframe_with_info import DataFrameWithInfo, copy_df_info_with_new_df

from .umap_plot import UmapBokeh


def prepare_umap_data(
    df_info: DataFrameWithInfo,
    col_list: Tuple = None,
    not_nan_percentage_threshold=0.9,
    test_over_set_ratio=0.25,
) -> Tuple[DataFrameWithInfo, DataFrameWithInfo, DataFrameWithInfo, DataFrameWithInfo]:
    """
    PREPARING DATA FOR UMAP

    Selecting:
    - only the features in col_list of df (or med_exam_col_list if col_list is not provided)
    - only the features with a number of not_NaN higher than not_nan_percentage_threshold
    - only the rows with no NaN in the remaining features
    Splitting into train-test set if test_over_set_ratio != 0

    Parameters
    ----------
    df_info: DataFrameWithInfo
        DataFrame containing data with all the feature
    col_list: Tuple
        List of columns that will be used by UMAP to compute embeddings. This specifies which features will
        be contained in the returned train_umap_data and test_umap_data. These features will be selected to choose
        the ones with few NaNs based on "not_nan_percentage_threshold" parameter. The order of these
        columns will remain the same in train_umap_data and test_umap_data
    not_nan_percentage_threshold: float
        Every feature, which will contain less than this precentage of not-NaN, will not be considered in the returned
        data used to compute embeddings by UMAP
    test_over_set_ratio: float
        This specifies the ratio between training and test set. It corresponds to the size of test set divided by the
        size of the whole set

    Returns
    -------
    train_umap_data: DataFrameWithInfo
        the result of above selection, with only the features that will be embedded
        by UMAP (no metadata/demographic data)
    train_notna_full_features_df: DataFrameWithInfo
        the result of above selection (exactly same rows of train_umap_data), with full data,
        and index reset
    test_umap_data: DataFrameWithInfo
        the result of above selection, with only the features that will be embedded
        by UMAP (no metadata/demographic data)
    test_notna_full_features_df: DataFrameWithInfo
        the result of above selection (exactly same rows of test_umap_data), with full data,
        and index reset
    """
    feat_list_notnan = []
    many_nan_features = []
    if col_list is None:
        col_list = df_info.med_exam_col_list

    for f in col_list:
        not_na_count = sum(df_info.df[f].notna())
        if not_na_count > not_nan_percentage_threshold * df_info.df.shape[0]:
            feat_list_notnan.append(f)
        else:
            many_nan_features.append(f)
    logging.info(
        f"The features that have too high number of Nan (and will not be considered "
        f"in UMAP) are: {many_nan_features}"
    )
    # Select only the rows based on "feat_list_notnan", dropping the ones containing NaN
    umap_data = df_info.df[feat_list_notnan].dropna(axis=0)
    if test_over_set_ratio != 0:
        # SPLIT into training and test set
        train_umap_data, test_umap_data = sklearn.model_selection.train_test_split(
            umap_data, test_size=test_over_set_ratio
        )

        # Use train and test to create DataFrameWithInfo instances
        train_umap_data = copy_df_info_with_new_df(
            df_info=df_info, new_pandas_df=train_umap_data
        )
        test_umap_data = copy_df_info_with_new_df(
            df_info=df_info, new_pandas_df=test_umap_data
        )
        # Select only the rows in umap data from the df
        train_notna_full_features_df = copy_df_info_with_new_df(
            df_info=df_info, new_pandas_df=df_info.df.iloc[train_umap_data.df.index]
        )
        test_notna_full_features_df = copy_df_info_with_new_df(
            df_info=df_info, new_pandas_df=df_info.df.iloc[test_umap_data.df.index]
        )
        # Reset index (so it is easier for further analysis)
        train_notna_full_features_df.df = train_notna_full_features_df.df.reset_index(
            drop=True
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
        umap_data = copy_df_info_with_new_df(df_info=df_info, new_pandas_df=umap_data)
        return umap_data, df_info


def get_umap_embeddings(
    train_df_info: DataFrameWithInfo,
    n_neighbors: int,
    min_dist: float,
    n_components=2,
    metric="euclidean",
    metric_kwds=None,
    random_state=42,
    test_df_info: DataFrameWithInfo = None,
) -> Tuple[umap.UMAP, np.ndarray, np.ndarray]:
    """
    The function will fit the UMAP algorithm on 'train_df' and it will then use the model to transform
    train and test set.

    Parameters
    ----------
    train_df_info
    n_neighbors
    min_dist
    n_components: int
        Embedding number of dimensions. E.g. For 2D plot, select 2. Default set to 2.
    random_state
    metric
    metric_kwds
    test_df_info

    Returns
    -------
    reducer
    embedding: np.ndarray
    test_embedding: np.ndarray
    """
    if metric_kwds is None:
        metric_kwds = {}
    # Create and fit UMAP
    reducer = umap.UMAP(
        random_state=random_state,
        metric=metric,
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=min_dist,
        metric_kwds=metric_kwds,
    )
    reducer.fit(train_df_info.df)

    # Transform using UMAP transformation
    embedding = reducer.transform(train_df_info.df)
    if test_df_info is not None:
        test_embedding = reducer.transform(test_df_info.df)
    else:
        test_embedding = None

    return reducer, embedding, test_embedding


def plot_umap(
    embedding: np.ndarray,
    df_full_feat: DataFrameWithInfo,
    umap_params_str: str,
    group_values_to_be_shown: Union[Tuple[str, Tuple[Tuple[str]]], None] = None,
    color_tuple=palettes.Category10[10],
    test_embedding: np.ndarray = None,
    feature_to_color=None,
    multi_marker_feats: Union[Tuple, None] = None,
    enc_value_to_str_map: Dict[str, Dict] = None,
    legend_location="bottom_left",  # random_state=None,
    test_df_full_feat: DataFrameWithInfo = None,
    test_color_tuple: Tuple = palettes.Set3[12],
    tools: str = "box_zoom,box_select,wheel_zoom,reset,tap,save",
    tooltip_feats: Tuple = None,
    tooltips=None,
    plot_height: int = 1200,
    plot_width: int = 1200,
    marker_size: int = 6,
    marker_fill_alpha: float = 0.2,
    return_plot=False,
    filename_title_prefix="",
):
    """
    This method will create a Bokeh plot with UMAP data. The different series of points will be defined
    combining two arguments:
    1. 'multi_feat_to_combine_partit_list' -> List of columns which will be combined together in
        order to define many partitions of the embedding -> These partitions will be
        distinguished by different markers
    2. 'feature_to_color' -> These column values will be used for every partition defined in 1.
        in order to distinguish some sub-partitions which will be colored differently (using
        color_tuple if provided or other categorical preset palettes)
    Optionally, you may provide a test_set with arguments "test_embedding", "test_df_full_feat" that will be
    drawn upon the (training) dataset to test the UMAP algorithm

    @param df_full_feat: DataFrameWithInfo instance containing additional values used to partition the point
        series, to color them, and to show data with HoverTool
    @param embedding: np.ndarray containing the (x, y) values of the points to be plotted
    @param umap_params_str: String containing infos about umap parameters applied
    @param feature_to_color: This will be the df_info column (str) that define the color of the series
        according to its values (the colors can be listed by color_list argument)
    @param multi_marker_feats: These column_names/features will be used to define different
        series of points (described and listed in legend). Their unique values combination will define
        different partitions that will be distinguishable by markers or colors.
    @param enc_value_to_str_map: Dict[Dict]
        This is a map to connect the encoded values to the original ones. It will be a Dict of Dicts
        because the first level keys identify the features with encoded values, and the second level
        dict is the actual map
    @param test_df_full_feat: DataFrameWithInfo -> Test_dataset containing values related to 'test_embedding'
        samples used to partition the point series, to color them, and to show data with HoverTool
    @param test_embedding: np.ndarray -> Test_dataset containing the (x, y) values of the points to be plotted
    @param group_values_to_be_shown: Tuple[str, Tuple] -> This is required if you want to plot a part
        of DataFrame as a Serie of grey points (as background), and the other part will be split in
        partitions according to the other arguments. If you want the DataFrame to considered all together
        with no grey/background points, do not use this argument.
        This is a tuple where the first element is the name of the column that will be used.
        The second element is a Tuple containing all the values that will be included in the plot.
        The rows that have other values for that column will be plotted as grey points.
         E.g. ('BREED', ('LABRADOR RETRIEVER', 'MONGREL'))
    @param legend_location: str -> Location of the legend in the plot. Default set to 'bottom_left'
    @param color_tuple: This is the list of colors to be used to distinguish 'feature_to_color' values
    @param test_color_tuple
    @param tools
    @param tooltip_feats
    @param tooltips
    @param plot_height
    @param plot_width
    @param marker_size
    @param marker_fill_alpha
    @param return_plot
    @param filename_title_prefix

    @return: No return values
    """
    if group_values_to_be_shown is None:
        group_values_to_be_shown = (None, [""])

    # Create a plot for each breed
    umap_plot_per_breed = {}

    # Loop over every subgroup of values you chose from 'feature_column_with_subgroups' column
    for subgroup in group_values_to_be_shown[1]:
        # Create an instance of umap_bokeh_plot
        umap_plot = UmapBokeh(
            tools=tools, plot_width=plot_width, plot_height=plot_height
        )
        # Cast string to tuple if subgroup is only one element
        if subgroup != "":
            subgroup = tuple([subgroup]) if isinstance(subgroup, str) else subgroup
        if multi_marker_feats is None:
            multi_marker_feats = ()

        # Calculate the sample count for the training set to be inserted in the bokeh figure title
        if group_values_to_be_shown[0] is not None:
            sample_count_in_subgroup = df_full_feat.df[
                df_full_feat.df[group_values_to_be_shown[0]].isin(subgroup)
            ].shape[0]
            group_values_to_be_shown = (group_values_to_be_shown[0], tuple(subgroup))
            subgroup_str = "-".join(subgroup)
        else:
            sample_count_in_subgroup = df_full_feat.df.shape[0]
            subgroup_str = ""

        # LOOP over training and test set
        dataset_settings = (
            (embedding, df_full_feat, color_tuple, "tr"),
            (test_embedding, test_df_full_feat, test_color_tuple, "test"),
        )
        for embed, df_full, colors, legend_label_prefix in dataset_settings:
            # Check if test_set has been provided
            if embed is not None:
                # Add multiple series to the figure according to the partitions
                umap_plot_per_breed[
                    subgroup_str
                ] = umap_plot.add_series_combine_multi_feat(
                    df_info=df_full,
                    embedding=embed,
                    feature_to_color=feature_to_color,
                    multi_marker_feats=multi_marker_feats,
                    enc_value_to_str_map_multi_feat=enc_value_to_str_map,
                    title=f"{filename_title_prefix}_{subgroup_str} ({sample_count_in_subgroup})_{umap_params_str}",
                    group_values_to_be_shown=group_values_to_be_shown,
                    color_list=colors,
                    legend_location=legend_location,
                    legend_label_prefix=legend_label_prefix,
                    marker_size=marker_size,
                    marker_fill_alpha=marker_fill_alpha,
                    tooltip_feats=tooltip_feats,
                    tooltips=tooltips,
                )

        # If you choose to show_plot, then "plot_umap_bokeh" cannot and will not return
        # any plot because it cannot be reused
        if not return_plot:
            if filename_title_prefix is None:
                filename_title_prefix = f"UMAP_{subgroup_str}_{umap_params_str}.html"

            bk.output_file(
                filename=f"{filename_title_prefix.replace('.html', '')}_{subgroup_str}_{umap_params_str}.html"
            )
            bk.show(umap_plot_per_breed[subgroup_str])

    if return_plot:
        return umap_plot_per_breed
    else:
        return None


def calculate_plot_umap(
    df_exams_only_umap: DataFrameWithInfo,
    df_full_feat: DataFrameWithInfo,
    n_neighbors: int,
    min_dist: float,
    group_values_to_be_shown: Union[Tuple[str, Tuple[Tuple[str]]], None] = None,
    color_tuple=palettes.Category10[10],
    test_df_exams_only_umap: DataFrameWithInfo = None,
    random_state=42,
    metric="euclidean",
    feature_to_color=None,
    multi_feat_to_combine_partit_list: Union[Tuple, None] = None,
    test_df_full_feat: DataFrameWithInfo = None,
    test_color_tuple: Tuple = palettes.Set3[12],
    return_plot=True,
    filename_prefix=None,
):
    """
    This function is meant to use these arguments to calculate and make only a single plot.
    It will show the plot and, if filename argument is provided, it will save
    the plot in a html file.
    This method will create a Bokeh plot with UMAP data. The different series of points will be defined
    combining two arguments:
    1. 'multi_feat_to_combine_partit_list' -> List of columns which will be combined together in
        order to define many partitions of the embedding -> These partitions will be
        distinguished by different markers
    2. 'feature_to_color' -> These column values will be used for every partition defined in 1.
        in order to distinguish some sub-partitions which will be colored differently (using
        color_tuple if provided or other categorical preset palettes)
    Optionally, you may provide a test_set with arguments "test_embedding", "test_df_full_feat" that will be
    drawn upon the (training) dataset to test the UMAP algorithm

    @param df_exams_only_umap: DataFrameWithInfo instance containing only the features that will be used to create UMAP
    @param df_full_feat: DataFrameWithInfo instance containing values related to 'test_df_exams_only_umap'
        samples. It is used to partition the point series, to color them, and to show data with HoverTool
    @param min_dist: Minimum distance -> important parameter for UMAP algorithm
    @param n_neighbors: Number of nearest neighbors -> important parameter for UMAP algorithm
    @param feature_to_color: This will be the df_info column (str) that define the color of the series
        according to its values (the colors can be listed by color_list argument
    @param multi_feat_to_combine_partit_list: These column_names/features will be used to define different
        series of points (described and listed in legend). Their unique values combination will define
        different partitions that will be distinguishable by markers or colors.
    @param test_df_full_feat: DataFrameWithInfo -> Test_dataset containing values related to 'test_df_exams_only_umap'
        samples. It is used to partition the point series, to color them, and to show data with HoverTool
    @param test_df_exams_only_umap: DataFrameWithInfo -> Test_dataset containing only the features that
        will be used to create UMAP
    @param test_color_tuple: Color Tuple to color the test set
    @param metric: Metric used by UMAP in the calculations
    @param random_state: Initial Random State to fix UMAP calculations
    @param group_values_to_be_shown: Tuple[str, Tuple] -> This is required if you want to plot a part
        of DataFrame as a Serie of grey points (as background), and the other part will be split in
        partitions according to the other arguments. If you want the DataFrame to considered all together
        with no grey/background points, do not use this argument.
        This is a tuple where the first element is the name of the column that will be used.
        The second element is a Tuple containing all the values that will be included in the plot.
        The rows that have other values for that column will be plotted as grey points.
         E.g. ('BREED', ('LABRADOR RETRIEVER', 'MONGREL'))
    @param color_tuple: This is the list of colors to be used to distinguish 'feature_to_color' values
    @return: No return values
    """
    # Create UMAP embeddings
    reducer, embedding, test_embedding = get_umap_embeddings(
        train_df_info=df_exams_only_umap,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        metric=metric,
        test_df_info=test_df_exams_only_umap,
    )

    umap_plot_per_breed = plot_umap(
        embedding=embedding,
        df_full_feat=df_full_feat,
        umap_params_str=f"{n_neighbors}_{min_dist}",
        group_values_to_be_shown=group_values_to_be_shown,
        feature_to_color=feature_to_color,
        multi_marker_feats=multi_feat_to_combine_partit_list,
        color_tuple=color_tuple,
        test_embedding=test_embedding,
        test_df_full_feat=test_df_full_feat,
        test_color_tuple=test_color_tuple,
        return_plot=return_plot,
    )
    return reducer, embedding, umap_plot_per_breed


# TODO: FIX THIS!!
def calculate_plot_umap_multi_breed_multi_params(
    df_exams_only_umap,
    df_full_feat,
    n_neighbors_list,
    min_dist_list,
    multiple_breed_list: Tuple[Tuple],
    feature_to_color=None,
    three_feat_color_list: Tuple = (),
    color_list=bokeh.palettes.Category10[10],
    test_df_exams_only_umap=None,
    test_df_full_feat=None,
    show_grid: bool = True,
    ncols=None,
    filename=None,
):
    """
    :param df_exams_only_umap: This contains only the features that need to be considered by umap algorithm
    :param df_full_feat: This has the same rows like 'df_exams_only_umap' argument, but it contains more features
        (like the 'feature_to_color' needed to highlight different partitions)
    :param feature_to_color: This must contain integers that will be used to color the
        umap points differently according to that value
    """
    if ncols is None:
        ncols = len(min_dist_list)

    reducer = {}
    embedding = {}
    umap_plots = {}

    for n_neighbors in n_neighbors_list:
        for min_dist in min_dist_list:
            (
                reducer[(n_neighbors, min_dist)],
                embedding[(n_neighbors, min_dist)],
                umap_plots[(n_neighbors, min_dist)],
            ) = calculate_plot_umap(
                df_exams_only_umap=df_exams_only_umap,
                df_full_feat=df_full_feat,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                group_values_to_be_shown=("BREED", multiple_breed_list),
                feature_to_color=feature_to_color,
                multi_feat_to_combine_partit_list=three_feat_color_list,
                color_tuple=color_list,
                test_df_exams_only_umap=test_df_exams_only_umap,
                test_df_full_feat=test_df_full_feat,
                return_plot=False,
            )

    if show_grid:
        # List of plots
        umap_plot_list = []

        for k in umap_plots.keys():
            # Extend with the figures from every breed
            umap_plot_list.extend(list(umap_plots[k].values()))

        grid = bk.gridplot(umap_plot_list, ncols=ncols)
        # TODO Insert a function to save a list of plots into a grid
        if feature_to_color is None and filename is None:
            filename = (
                f"UMAP_{'-'.join(three_feat_color_list)}_multibreed_multi_params.html"
            )
        else:
            filename = f"UMAP_{feature_to_color}_multibreed_multi_params.html"

        bk.output_file(filename=filename)
        bk.show(grid)

    return reducer, embedding, umap_plots
