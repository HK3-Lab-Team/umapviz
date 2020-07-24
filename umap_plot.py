import copy
from typing import Tuple, Union, Dict
import pandas as pd
import numpy as np

import bokeh
from bokeh.models import CategoricalColorMapper, ColumnDataSource
from bokeh import palettes
import bokeh.plotting as bk
import bokeh.transform

from plotting.utils.plot_settings import MARKERS_DICT
from pd_extras.utils.dataframe_with_info import DataFrameWithInfo
from plotting.utils.bokeh_boxplot import get_hover_tool


def _get_specific_marker_by_keys(partition_id_keys: Tuple[int],
                                 markers_dict: Dict = MARKERS_DICT) -> str:
    """
    This will return a marker based on the combination 'partition_id_keys' of integer which
    represent the partition ids, so these partitions can be distinguished by markers.

    @param partition_id_keys: Tuple of integers which represent the partition ids, so that each
        partitions has a different marker
    @param markers_dict: Dict -> This is a Dict tree containing integer as keys and values for every layer,
        except the lowest layer where we have the names of the markers that can be used by bokeh.
        (e.g. see Enum.MARKERS_DICT). By default this is set to Enums.MARKERS_DICT
    @return: str -> Name of a marker stored in markers_dict compatible with bokeh
    """
    my_marker = copy.copy(markers_dict)
    partitioning_level = 0
    for key in partition_id_keys:
        try:
            # Iteratively we look for a more specific marker from the Dict tree
            my_marker = my_marker[key]
            partitioning_level += 1
        except KeyError as e:
            logging.warning(f"You chose too many partitioning levels (i.e. {partitioning_level} "
                            "levels) and there are not enough level in Enums.MARKERS_DICT dictionary. "
                            "We set 'circle' as marker by default.")
            return 'circle'
    # We keep on looking the first actual marker name (string) declared at the bottom level of
    # MARKERS_DICT tree. This is useful when in markers_dict there are more levels than the
    # length of partition_id_keys (i.e. the number of partitions)
    while not isinstance(my_marker, str):
        my_marker = my_marker[0]
    return my_marker


class UmapBokeh:

    def __init__(self, tools='box_zoom,box_select,wheel_zoom,reset,tap,save', plot_height: int = 1200,
                 plot_width: int = 1200, title: str = "UMAP plot",
                 ):

        self.umap_fig = bk.figure(plot_width=plot_width, plot_height=plot_height, background_fill_color='white',
                                  tools=tools, toolbar_location="above")

        self.umap_fig.title.text = title
        self.umap_fig.title.align = "center"
        self.umap_fig.title.text_color = "black"
        self.umap_fig.title.text_font_size = "25px"

        # Needed to know that we need to use a LinearColorMapper and
        # ColorBar (not just a list of colors)
        self._has_colorbar = False
        self._partition_id_to_string_map = None
        self._enc_to_str_map_multi_feat = {}

        # List of default features that will be included in hover tooltip
        self._tooltip_feats = []

    def _add_tooltip_infos_to_data_dict(self, df_with_default_infos: pd.DataFrame, data_dict: Dict):
        """
        This fills up 'data_dict' argument with default_infos.
        These will not be added if the option 'add_default_infos_as_hover_tooltips' of the instance is False.
        This will be set during instantiation.
        @param df_with_default_infos: pd.DataFrame containing the columns with default
            infos listed in Enum.DEFAULT_HOVER_TOOLTIPS_LIST
        @param data_dict: Dictionary with the column name as key, and the pd.Series of the column as value
            This will be used as data source for the scatter plot serie
        @return: data_dict -> The dictionary from input where it added the column values from default_infos
        """
        output_dict = data_dict

        if self._tooltip_feats:
            # If needed, we add default_infos to hover_tooltips, listed in Enum.DEFAULT_HOVER_TOOLTIPS_LIST
            for feat in self._tooltip_feats:
                # If one of the default_infos is not in df_columns, we will neither consider it nor plot it
                if feat in df_with_default_infos.columns:
                    output_dict[feat] = df_with_default_infos[feat]
                else:
                    logging.warning(f"The default info {feat} that you required for tooltip ('tooltip_feats' "
                                    f"argument) is not among the columns of the df provided. "
                                    f"Its value will not be shown")

        return output_dict

    def _map_enc_value_to_str(self, feat_name, feat_enc_value):
        try:
            return self._enc_to_str_map_multi_feat[feat_name][int(feat_enc_value)]
        except KeyError:
            logging.info(f"No mapping found for the encoded value {int(feat_enc_value)} of the "
                         f"feature {feat_name}. The encoded value will be used")
            return feat_enc_value

    def _complete_enc_to_str_map_multi_feat(self, df_info: DataFrameWithInfo,
                                            enc_value_to_str_map_multi_feat: Dict[str, Dict],
                                            feats_to_map: Tuple[str]):
        for feat in feats_to_map:
            if feat not in enc_value_to_str_map_multi_feat:
                # If no direct map is available, look if there are informations about encoding
                enc_to_str_map_single_feat = df_info.get_encoded_string_values_map(feat)
                if enc_to_str_map_single_feat is None:
                    logging.info("The feature is not categorical and/or not encoded and/or no mapping "
                                 "has been provided. Partitions will have numerical names")
                    self._enc_to_str_map_multi_feat[feat] = {}
                else:
                    self._enc_to_str_map_multi_feat[feat] = enc_to_str_map_single_feat
            else:
                self._enc_to_str_map_multi_feat[feat] = enc_value_to_str_map_multi_feat[feat]

    def _get_color_list_palette(self, df: pd.DataFrame, unique_values_list: Tuple, feature_to_color: str,
                                input_color_list: Union[Tuple, None]) -> Union[Tuple, Dict]:

        # Test if 'color_list' argument is appropriate (enough colors for every category) or not provided
        if input_color_list is None or len(input_color_list) < len(unique_values_list):
            # If 'color_list' argument is not appropriate, create one
            try:
                # First try using a long categorical palette with 20 elements
                color_list = palettes.Category20[len(unique_values_list)]
                self._has_colorbar = False
                return color_list
            except KeyError:
                # If the palette is not enough, use a continuous palette with LinearColorMapper
                # and add a ColorBar to the plot
                logging.warning(f"The number of unique values ({len(unique_values_list)}) in column {feature_to_color}"
                                "exceeds number of colors in the palette (20). Will cast to integers and "
                                "trying with wider Palette.")
                # Set the feature_to_color values to int (in order to be mappable into a ColorBar)
                df.loc[:, feature_to_color] = df[feature_to_color].astype(int)

                max_color_code = df[feature_to_color].max()
                min_color_code = df[feature_to_color].min()
                # Create ColorMapper to map the partition labels to a color
                mapper = bokeh.models.LinearColorMapper(palette=bokeh.palettes.Turbo256, low=min_color_code,
                                                        high=max_color_code)
                # Mapping feat_value (feature_to_color) to colors of the palette
                color_list = bokeh.transform.transform(feature_to_color, mapper)
                # Adding the Color Bar
                color_bar = bokeh.models.ColorBar(color_mapper=mapper, location=(0, 0),
                                                  ticker=bokeh.models.SingleIntervalTicker(
                                                      desired_num_ticks=max_color_code - min_color_code + 1,
                                                      # choose interval with one order of magnitude less than the max
                                                      interval=10 ** max((int(np.log10(max_color_code)) - 2), 1)
                                                  ),
                                                  formatter=bokeh.models.PrintfTickFormatter(format="%d"))

                self.umap_fig.add_layout(color_bar, 'right')
                # Needed to know that we need to use a LinearColorMapper and
                # ColorBar (not just a list of colors)
                self._has_colorbar = True

                return color_list
        else:
            # Select colors only up to length of unique_values_list
            return input_color_list[:len(unique_values_list)]

    def _add_single_scatter_point_serie(self, df_x_y: np.ndarray,
                                        df_full_info_partition: pd.DataFrame, feature_to_color: str,
                                        color: Union[str, Dict], marker: str = 'circle',
                                        legend_label: Union[str, None] = None, size: int = 10,
                                        legend_label_prefix: str = '', fill_alpha: float = 0.2,
                                        fill_color: str = None):
        """

        @param df_x_y: Numpy NdArray containing the (x, y) coordinates of the points to be plotted
        @param df_full_info_partition: A slice of pd.DataFrame containing additional values used to
            partition the point series, to color them, and to show data with HoverTool
        @param feature_to_color:
        @param color: str -> Color of the markers of the serie.
        @param marker: Type of marker
        @param legend_label: Name of the serie that will be used in legend
        @param size: Size of marker to be used
        @param fill_color:
        @return: No return values
        """
        # We select the df_x_y rows based on the index of the elements in df_full_info_slice,
        data_dict = {
            'x': df_x_y[df_full_info_partition.index, 0],
            'y': df_x_y[df_full_info_partition.index, 1],
            feature_to_color: df_full_info_partition[feature_to_color],
        }
        # Add default_infos to data_dict to allow HoverTool to add those data
        data_dict = self._add_tooltip_infos_to_data_dict(
            df_with_default_infos=df_full_info_partition,
            data_dict=data_dict
        )
        source = ColumnDataSource(data_dict)
        if legend_label is None:
            legend_label = feature_to_color
        # Draw points for this partition and breed
        self.umap_fig.scatter(x='x', y='y', marker=marker, size=size, source=source, color=color,
                              legend_label=f'{legend_label_prefix}_{legend_label}',
                              fill_alpha=fill_alpha, line_width=1.5, fill_color=color, )

    def _add_multi_feature_scatter_series_from_multi_partit(self, df_x_y: np.ndarray,
                                                            df_full_info: pd.DataFrame,
                                                            features_to_partitioned_series: Tuple[str],
                                                            feature_to_color: str,
                                                            color_list: Tuple,
                                                            unique_values_list: Tuple,
                                                            series_legend_label: str = '',
                                                            partition_id_keys: Tuple[int] = (),
                                                            legend_label_prefix: str = '',
                                                            marker_size: int = 5,
                                                            fill_alpha: float = 0.2,
                                                            ):
        """
        This method will add some series to the plot. The different series of points will be defined
        according to the values of the columns 'features_to_partitioned_series'. These series will be
        distinguished by different markers and they will also be colored (and split) only
        according to column 'feature_to_color'.
        @param df_x_y: pd.DataFrame containing the (x, y) values of the points to be plotted
        @param df_full_info: pd.DataFrame containing additional values used to partition the point
            series, to color them, and to show data with HoverTool
        @param feature_to_color: This will be the one that define the color according to its values
        @param color_list: This is the list of colors to be used to distinguish 'feature_to_color' values
        @param unique_values_list: Tuple of the unique values of 'feature_to_color' column
        @param features_to_partitioned_series: These column_names/features will define different
            series of points (described and listed in legend)
        @param partition_id_keys: Tuple[int] -> Tuple of integer keys that identify different partitions.
            It will be used to pick a specific marker from Enum.MARKERS_DICT (it will be created by
            the function recursions)
        @param series_legend_label: str -> Title of the serie (it will be created by the function recursions)
        @param legend_label_prefix: Prefix for the series names in the legend (to be
            distinguishable by other series)
        @return: No return values
        """
        if len(features_to_partitioned_series) > 0:
            # While we still have features to consider, we need to keep calling this function
            # Get name of that feature and remove it from the list
            reduced_features_to_partitioned_series = list(features_to_partitioned_series)
            feat_name = reduced_features_to_partitioned_series.pop(0)
            # Find unique values for that column
            feat_unique_values = df_full_info[feat_name].unique()

            # Loop over each unique value of the 'feat_name' feature
            for value_id in range(len(feat_unique_values)):
                feat_enc_value = feat_unique_values[value_id]
                # Add the value_id to the partition_id_keys tuple
                # (it will be used to select the marker for the partition)
                new_partition_id_keys = list(partition_id_keys)
                new_partition_id_keys.append(value_id)
                # Select the slice of df_full_info corresponding to the partition
                df_full_info_slice = df_full_info[df_full_info[feat_name] == feat_enc_value]
                # Get string value corresponding to encoded one
                feat_value = self._map_enc_value_to_str(feat_name, feat_enc_value)
                # Add the value of the partition to the series title (avoid starting with '-' )
                partition_series_title = str(feat_value) if (
                        series_legend_label == '') else f'{series_legend_label}-{str(feat_value)}'
                self._add_multi_feature_scatter_series_from_multi_partit(
                    df_x_y=df_x_y,
                    df_full_info=df_full_info_slice,
                    features_to_partitioned_series=tuple(reduced_features_to_partitioned_series),
                    feature_to_color=feature_to_color,
                    unique_values_list=unique_values_list,
                    series_legend_label=partition_series_title,
                    partition_id_keys=tuple(new_partition_id_keys),
                    fill_alpha=fill_alpha, marker_size=marker_size,
                    color_list=color_list,
                    legend_label_prefix=legend_label_prefix
                )
        else:
            # This is where we combined every feature (except the last one) and we identified a specific
            # sample partition. So we can elaborate the last feature and the specific partition
            self._add_single_feature_scatter_series_from_single_partit(
                df_x_y=df_x_y,
                df_full_info_partition=df_full_info,
                feature_to_color=feature_to_color,
                unique_values_list=unique_values_list,
                series_legend_label=series_legend_label,
                partition_id_keys=partition_id_keys,
                fill_alpha=fill_alpha, size=marker_size,
                color_list=color_list,
                legend_label_prefix=legend_label_prefix
            )
            return

    def _add_single_feature_scatter_series_from_single_partit(self, df_x_y: np.ndarray,
                                                              df_full_info_partition: pd.DataFrame,
                                                              feature_to_color: str,
                                                              unique_values_list: Tuple,
                                                              series_legend_label: str, color_list: Tuple,
                                                              partition_id_keys: Union[Tuple, None] = None,
                                                              marker: Union[str, None] = None,
                                                              legend_label_prefix: str = '',
                                                              size: int = 8, fill_alpha: float = 0.2,
                                                              markers_dict=MARKERS_DICT):
        """
        This method will add few series to the plot. The different series of points will be defined
        according to the values of the only column 'feature_to_color'. The different series will be
        colored (and split) according to it (using 'color_list').
        The 'df_full_info_partition' already represents a partition of the df and this partition will
        be characterized by a specific marker, a 'series_title', and 'partition_id_keys'.
        @param legend_label_prefix:
        @param df_x_y: Numpy NdArray containing the (x, y) coordinates of the points to be plotted
        @param df_full_info_partition: A slice of pd.DataFrame containing additional values used to
            partition the point series, to color them, and to show data with HoverTool
        @param feature_to_color: This will be the one that define the color according to its values
        @param color_list: This is the list of colors to be used to distinguish 'feature_to_color' values
        @param unique_values_list: Tuple of the unique values of 'feature_to_color' column
        @param marker: Type of marker to be used
        @param size: Size of marker to be used
        @param partition_id_keys: Tuple[int] -> Tuple of integer keys that identify different partitions.
            It will be used to pick a specific marker from Enum.MARKERS_DICT. If there is no
            partition, a '0' tuple will be set dy default, so the first marker from
            Enum.MARKERS_DICT will be used
        @param markers_dict: Dict -> This is a Dict tree containing integer as keys and values for every layer,
            except the lowest layer where we have the names of the markers that can be used by bokeh.
            (e.g. see Enum.MARKERS_DICT). By default this is set to Enums.MARKERS_DICT
        @param legend_label_prefix: Prefix for the series names in the legend (to be
            distinguishable by other series)
        @param series_legend_label: str -> Title of the serie
        @return: No return values
        """
        for color, partition_id in zip(color_list, unique_values_list):
            # Select only the rows of this partition
            selected_partit_df = df_full_info_partition[
                df_full_info_partition[feature_to_color] == partition_id
                ]
            # Map partition_id to corresponding name
            partition_str = self._map_enc_value_to_str(feature_to_color, partition_id)

            if marker is None:
                if partition_id_keys is None:
                    logging.error('You must provide a marker or a partition_id_keys which will be used '
                                  'to get a marker from Enum.MARKERS_DICT or markers_dict argument')
                else:
                    # Get a marker based on the partition
                    marker = _get_specific_marker_by_keys(partition_id_keys, markers_dict=markers_dict)

            self._add_single_scatter_point_serie(df_x_y=df_x_y,
                                                 df_full_info_partition=selected_partit_df,
                                                 feature_to_color=feature_to_color,
                                                 color=color, fill_color='white',
                                                 marker=marker, size=size, fill_alpha=fill_alpha,
                                                 legend_label_prefix=legend_label_prefix,
                                                 legend_label=f"{series_legend_label}-{partition_str}", )

    def _add_single_feature_scatter_series_with_colorbar(self, df_x_y: np.ndarray,
                                                         df_full_info_partition: pd.DataFrame,
                                                         feature_to_color: str,
                                                         series_legend_label: str,
                                                         color_transform_mapper_dict: Dict,
                                                         partition_id_keys: Union[Tuple, None] = None,
                                                         marker: Union[str, None] = None,
                                                         fill_alpha: float = 0.2,
                                                         legend_label_prefix: str = '',
                                                         marker_size: int = 5,
                                                         markers_dict=MARKERS_DICT):
        """
        This method will be used when "self._has_colorbar == True" add few series to the plot. The different
        series of points will be defined according to the values of the only column 'feature_to_color'.
         The different series will be
        colored (and split) according to it (using 'color_list').
        The 'df_full_info_partition' already represents a partition of the df and this partition will
        be characterized by a specific marker, a 'series_title', and 'partition_id_keys'.
        @param legend_label_prefix:
        @param df_x_y: Numpy NdArray containing the (x, y) coordinates of the points to be plotted
        @param df_full_info_partition: A slice of pd.DataFrame containing additional values used to
            partition the point series, to color them, and to show data with HoverTool
        @param feature_to_color: This will be the one that define the color according to its values
        @param color_transform_mapper_dict: In the case where
            "self._has_colorbar == True", this will be a dict (returned by the function
             bokeh.models.transform) which contains a mapping between the elements in 'feature_to_color'
             of the source and the colors in the ColorBar.This is the list of colors to be used to distinguish
            'feature_to_color' values
        @param marker: Type of marker to be used
        @param marker_size: Size of marker to be used
        @param partition_id_keys: Tuple[int] -> Tuple of integer keys that identify different partitions.
            It will be used to pick a specific marker from Enum.MARKERS_DICT. If there is no
            partition, a '0' tuple will be set dy default, so the first marker from
            Enum.MARKERS_DICT will be used
        @param markers_dict: Dict -> This is a Dict tree containing integer as keys and values for every layer,
            except the lowest layer where we have the names of the markers that can be used by bokeh.
            (e.g. see Enum.MARKERS_DICT). By default this is set to Enums.MARKERS_DICT
        @param legend_label_prefix: Prefix for the series names in the legend (to be
            distinguishable by other series)
        @param series_legend_label: str -> Title of the serie
        @return: No return values
        """
        assert self._has_colorbar, "Something went wrong because the colorbar has not been " \
                                   "created by '_get_color_list_palette' method, and this " \
                                   "method requires colorbar"
        # Map partition_id to corresponding name
        if marker is None:
            if partition_id_keys is None:
                logging.error('You must provide a marker or a partition_id_keys which will be used '
                              'to get a marker from Enum.MARKERS_DICT or markers_dict argument')
            else:
                # Get a marker based on the partition
                marker = _get_specific_marker_by_keys(partition_id_keys, markers_dict=markers_dict)

        self._add_single_scatter_point_serie(df_x_y=df_x_y,
                                             df_full_info_partition=df_full_info_partition,
                                             feature_to_color=feature_to_color,
                                             color=color_transform_mapper_dict, fill_color=None,
                                             marker=marker, size=marker_size, fill_alpha=fill_alpha,
                                             legend_label_prefix=legend_label_prefix,
                                             legend_label=series_legend_label)

    def add_series_single_feat(self, df_info: DataFrameWithInfo, embedding: np.ndarray,
                               feature_to_color: str,
                               group_values_to_be_shown: Union[Tuple[str, Tuple], None] = None,
                               tooltips=None,
                               title: str = "UMAP plot", color_list=None):
        return self.add_series_combine_multi_feat(df_info=df_info, embedding=embedding,
                                                  feature_to_color=feature_to_color,
                                                  multi_marker_feats=(),
                                                  tooltip_feats=tooltips,
                                                  group_values_to_be_shown=group_values_to_be_shown,
                                                  title=title, color_list=color_list)

    def add_series_combine_multi_feat(self, df_info: DataFrameWithInfo, embedding: np.ndarray,
                                      feature_to_color: str,
                                      multi_marker_feats: Tuple = (),
                                      enc_value_to_str_map_multi_feat: Dict[str, Dict] = None,
                                      legend_location='bottom_left',
                                      group_values_to_be_shown: Union[Tuple[str, Tuple], None] = (None, ['']),
                                      tooltip_feats: Tuple = None, tooltips=None,
                                      marker_fill_alpha: float = 0.2, marker_size: int = 6,
                                      title: str = "UMAP plot", color_list=None, legend_label_prefix: str = ''):
        """
        This method will add some series to the plot. The different series of points will be defined
        according to the values of the columns 'features_to_partitioned_series'. These series will be
        distinguished by different markers and they will also be colored (and split) only
        according to column 'feature_to_color'.

        @param df_info: DataFrameWithInfo instance containing additional values used to partition the point
            series, to color them, and to show data with HoverTool
        @param embedding: pd.DataFrame containing the (x, y) values of the points to be plotted
        @param feature_to_color: This will be the df_info column (str) that define the color of the series
            according to its values (the colors can be listed by color_list argument
        @param multi_marker_feats: These column_names/features will be used to define different
            series of points (described and listed in legend). Their unique values combination will define
            different partitions that will be distinguishable by markers or colors.
        @param enc_value_to_str_map_multi_feat: Dict[Dict]
            This is a map to connect the encoded values to the original ones. It will be a Dict of Dicts
            because the first level keys identify the features with encoded values, and the second level
            dict is the actual map
        @param group_values_to_be_shown: Tuple[str, Tuple] -> This is required if you want to plot a part
            of DataFrame as a Serie of grey points (as background), and the other part will be split in
            partitions according to the other arguments. If you want the DataFrame to considered all together
            with no grey/background points, do not use this argument.
            This is a tuple where the first element is the name of the column that will be used.
            The second element is a Tuple containing all the values that will be included in the plot.
            The rows that have other values for that column will be plotted as grey points.
             E.g. ('BREED', ('LABRADOR RETRIEVER', 'MONGREL'))
        @param color_list: This is the list of colors to be used to distinguish 'feature_to_color' values
        @param title: str -> Title of the plot
        @param legend_location: str -> Location of the legend in the plot. Default set to 'bottom_left'
        @param legend_label_prefix: Prefix for the series names in the legend (to be
            distinguishable by other series)
        @param tooltip_feats:
        @param tooltips:
        @param marker_fill_alpha:
        @param marker_size:

        @return: No return values
        """
        self._set_tooltip_feats(
            tooltip_feats, multi_series_feats=list(multi_marker_feats) + [feature_to_color],
        )
        # Check if the feature is a numerical_column
        notna_col_df = df_info.df[df_info.df[feature_to_color].notna()]
        # Compare the first_row type with every other row of the same column
        col_types = notna_col_df[feature_to_color].apply(lambda r: str(type(r))).values
        has_same_values = all(col_types == col_types[0])
        col_type = col_types[0]
        if not has_same_values or not ('float' in col_type or 'int' in col_type):
            logging.error('The feature is not Numerical, so the UMAP algorithm cannot proceed')
        # ===========================================
        # STEP 1. Split df_info between the group_values that need to be shown or not (if required),
        #         show all otherwise
        # ===========================================
        if group_values_to_be_shown[0] is None:
            selected_group_df = df_info.df
        else:
            # DRAW GREY POINTS for the group_values we are not interested in
            # Split the df_infp
            groups_column_name, shown_values_from_group = group_values_to_be_shown
            selected_group_df = df_info.df[df_info.df[groups_column_name].isin(shown_values_from_group)]
            remaining_group_df = df_info.df[
                np.logical_not(df_info.df[groups_column_name].isin(shown_values_from_group))]

            self._add_single_scatter_point_serie(df_x_y=embedding,
                                                 df_full_info_partition=remaining_group_df,
                                                 feature_to_color=feature_to_color,
                                                 color="#cccccc", marker='circle', size=marker_size,
                                                 legend_label=f'{legend_label_prefix}_'
                                                              f'{groups_column_name}: others')
        # ===========================================
        # STEP 2. Draw the points from the interesting breeds with different colors
        # ===========================================
        # for each partition colors
        unique_values_list = tuple(sorted(selected_group_df[feature_to_color].unique()))
        color_list = self._get_color_list_palette(df=selected_group_df,
                                                  unique_values_list=unique_values_list,
                                                  feature_to_color=feature_to_color,
                                                  input_color_list=color_list)
        # ===========================================
        # STEP 3. For every partition we draw a separate serie of points
        # ===========================================
        # Get the map for the encoded features:
        feats_to_map = list(multi_marker_feats)
        feats_to_map.append(feature_to_color)
        self._complete_enc_to_str_map_multi_feat(df_info, enc_value_to_str_map_multi_feat,
                                                 feats_to_map=feats_to_map)

        if self._has_colorbar:
            # CASE 1: Distinguish partitions (according to feature_to_color) by color in colorbar
            self._add_single_feature_scatter_series_with_colorbar(
                df_x_y=embedding,
                df_full_info_partition=selected_group_df,
                feature_to_color=feature_to_color, fill_alpha=marker_fill_alpha,
                series_legend_label=f'{feature_to_color}_serie',
                color_transform_mapper_dict=color_list, marker='circle',
                legend_label_prefix=legend_label_prefix, marker_size=marker_size,
            )

        else:
            # CASE 2: Every partition is a separate serie described in legend
            self._add_multi_feature_scatter_series_from_multi_partit(
                df_x_y=embedding,
                df_full_info=selected_group_df,
                features_to_partitioned_series=multi_marker_feats,
                feature_to_color=feature_to_color,
                color_list=tuple(color_list),
                fill_alpha=marker_fill_alpha,
                marker_size=marker_size,
                unique_values_list=tuple(unique_values_list),
                legend_label_prefix=legend_label_prefix
            )
        # Add HOVER tool adding some features to the default ones defined in Enum
        # (it needs to be at last after adding the other point series
        hover = self._get_hover_tooltips(tooltips)

        self.umap_fig.add_tools(hover)

        # Set the title of bokeh figure based on this added serie
        self.umap_fig.title.text = title
        self.umap_fig.legend.location = legend_location
        self.umap_fig.legend.click_policy = 'hide'

        return self.umap_fig

    def _set_tooltip_feats(self, tooltip_feats, multi_series_feats, ):
        """
        It sets the class attribute 'self._tooltip_feats' using the features for the partition and
        additional features provided
        Parameters
        ----------
        tooltip_feats
        multi_series_feats

        Returns
        -------

        """
        if tooltip_feats is None:
            self._tooltip_feats = set(multi_series_feats)
        else:
            self._tooltip_feats = set(tooltip_feats).union(multi_series_feats)

    def _get_hover_tooltips(self, tooltips):
        """
        If 'tooltips' is provided, it uses them to create a HoverTool instance and return it.
        Otherwise, it uses 'self._tooltip_feats' and it does the same.
        """
        if tooltips is None:
            hover = get_hover_tool(tooltip_feats=tuple(self._tooltip_feats))
        else:
            hover = get_hover_tool(tooltip_feats=(), tooltips_list=tooltips)
        return hover


if __name__ == '__main__':
    import logging
    from pd_extras.utils.dataframe_with_info import DataFrameWithInfo
    from pd_extras.utils.dataframe_with_info import import_df_with_info_from_file

    logging.basicConfig(format='%(asctime)s \t %(levelname)s \t Module: %(module)s \t %(message)s ',
                        datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)

    # df_correct_dir = '/home/lorenzo-hk3lab/WorkspaceHK3Lab/smvet/data/data_dump'
    # df_correct = import_df_with_info_from_file(df_correct_dir)

    df_correct = import_df_with_info_from_file(filename='/home/lorenzo-hk3lab/WorkspaceHK3Lab/smvet/df_join_2')

    # df_correct.metadata_cols = list(df_correct.metadata_cols)
    # df_correct.metadata_cols.extend(
    #     ['consolidations_encoded', 'ground_glass_encoded', 'crazy_paving_encoded'])
    # df_correct.metadata_cols = set(df_correct.metadata_cols)

    from plotting.utils.umap import umap_functions

    train_umap_data, train_notna_full_features_df, \
    test_umap_data, test_notna_full_features_df = \
        umap_functions.prepare_umap_data(df_correct,
                                         not_nan_percentage_threshold=0.88,
                                         test_over_set_ratio=0.25)

    age_color_list = (
        '#34c616',  # Verde
        '#735911',  # Marrone
        '#12b6c4',  # Azzurro
    )
    test_age_color_list = (
        '#da2e1a',  # Rosso
        '#da971a',  # Arancione
        '#1a43da'  # Blu
    )
    tr_color_list = (
        '#34c616',  # Verde
        '#da2e1a',  # Rosso
        '#1a43da'  # Blu
    )
    test_color_list = (
        '#735911',  # Marrone
        '#da971a',  # Arancione
        '#12b6c4',  # Azzurro
    )
    # for min_dist in [0.01, 0.1, 1]:
    #     for n_neighbors in [5, 15, 50, 100]:
    reducer, embedding, umap_plot_per_group = umap_functions.calculate_plot_umap(
        df_exams_only_umap=train_umap_data,
        df_full_feat=train_notna_full_features_df,
        n_neighbors=7,  # [5, 15, 50, 100],
        min_dist=0.3,  # [0.01, 0.1, 1],
        # ['GERMAN SHEPHERD'], ['GOLDEN RETRIEVER']],
        feature_to_color='consolidations_encoded',
        multi_feat_to_combine_partit_list=('ground_glass_encoded', 'crazy_paving_encoded'),
        color_tuple=tr_color_list,
        test_df_exams_only_umap=test_umap_data,
        test_df_full_feat=test_notna_full_features_df,
        test_color_tuple=test_color_list,
        filename_prefix='umap_exprivia_db',
        return_plot=True
    )

    # reducer, embedding, umap_plot_per_group = umap_functions.calculate_plot_umap(
    #     df_exams_only_umap=train_umap_data,
    #     df_full_feat=train_notna_full_features_df,
    #     n_neighbors=50,  # [5, 15, 50, 100],
    #     min_dist=0.03,  # [0.01, 0.1, 1],
    #     group_values_to_be_shown=('BREED', (tuple(['MONGREL']), tuple(['LABRADOR RETRIEVER']))),
    #     # ['GERMAN SHEPHERD'], ['GOLDEN RETRIEVER']],
    #     feature_to_color='AGE_bin_id',
    #     multi_feat_to_combine_partit_list=('SEX', 'SEXUAL STATUS'),
    #     color_tuple=color_list,
    #     test_df_exams_only_umap=test_umap_data,
    #     test_df_full_feat=test_notna_full_features_df,
    #     test_color_tuple=test_color_list,
    #     filename_prefix='AGE_50_0.3_with_test',
    #     show_plot=True
    # )
