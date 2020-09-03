from pd_extras.dataframe_with_info import import_df_with_info_from_file
from umap_functions import calculate_plot_umap, prepare_umap_data

# df_correct_dir = '/home/lorenzo-hk3lab/WorkspaceHK3Lab/smvet/data/data_dump'
# df_correct = import_df_with_info_from_file(df_correct_dir)

df_correct = import_df_with_info_from_file(
    filename="/home/lorenzo-hk3lab/WorkspaceHK3Lab/smvet/df_join_2"
)

# df_correct.metadata_cols = list(df_correct.metadata_cols)
# df_correct.metadata_cols.extend(
#     ['consolidations_encoded', 'ground_glass_encoded', 'crazy_paving_encoded'])
# df_correct.metadata_cols = set(df_correct.metadata_cols)

(
    train_umap_data,
    train_notna_full_features_df,
    test_umap_data,
    test_notna_full_features_df,
) = prepare_umap_data(
    df_correct, not_nan_percentage_threshold=0.88, test_over_set_ratio=0.25
)

age_color_list = (
    "#34c616",  # Verde
    "#735911",  # Marrone
    "#12b6c4",  # Azzurro
)
test_age_color_list = ("#da2e1a", "#da971a", "#1a43da")  # Rosso  # Arancione  # Blu
tr_color_list = ("#34c616", "#da2e1a", "#1a43da")  # Verde  # Rosso  # Blu
test_color_list = (
    "#735911",  # Marrone
    "#da971a",  # Arancione
    "#12b6c4",  # Azzurro
)
# for min_dist in [0.01, 0.1, 1]:
#     for n_neighbors in [5, 15, 50, 100]:
reducer, embedding, umap_plot_per_group = calculate_plot_umap(
    df_exams_only_umap=train_umap_data,
    df_full_feat=train_notna_full_features_df,
    n_neighbors=7,  # [5, 15, 50, 100],
    min_dist=0.3,  # [0.01, 0.1, 1],
    # ['GERMAN SHEPHERD'], ['GOLDEN RETRIEVER']],
    feature_to_color="consolidations_encoded",
    multi_feat_to_combine_partit_list=(
        "ground_glass_encoded",
        "crazy_paving_encoded",
    ),
    color_tuple=tr_color_list,
    test_df_exams_only_umap=test_umap_data,
    test_df_full_feat=test_notna_full_features_df,
    test_color_tuple=test_color_list,
    filename_prefix="umap_exprivia_db",
    return_plot=True,
)

# reducer, embedding, umap_plot_per_group = calculate_plot_umap(
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
