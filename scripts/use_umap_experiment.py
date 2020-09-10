# %%
import logging
import os
from pathlib import Path

from pd_extras.dataframe_with_info import import_df_with_info_from_file
from umapviz.umap_exp import UmapExperiment
from umapviz.umap_metrics import tanimoto_gower

logging.basicConfig(
    format="%(asctime)s \t %(levelname)s \t Module: %(module)s \t %(message)s ",
    datefmt="%d-%b-%y %H:%M:%S",
    level=logging.INFO,
)

# -----------------------
# UMAP (DEFAULT) SETTINGS
# -----------------------
NEIGHBOURHOOD = 10
N_COMPONENTS = 2
RANDOM_STATE = 23
PLOT_HEIGHT = 600
PLOT_WIDTH = 900


# --------------
# Bokeh Settings
# --------------
TOOLS = (
    "hover,crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,"
    "undo,redo,reset,tap,save,box_select,poly_select,lasso_select,"
)
TOOLTIPS = [
    ("index", "$index"),
    ("(x,y)", "($x, $y)"),
    ("study_UID", "@study_uid"),
]
# INSERT APPROPRIATE DATA DIRECTORY
CWD = Path(os.path.abspath(os.path.dirname("__file__"))).parents[1]
DATA_PATH = CWD / "smvet" / "data" / "output_data" / "ordinal_encoded"
df_info = import_df_with_info_from_file(DATA_PATH / "df_ordinal_pat")

df_info.metadata_as_features = True
cols_by_type = df_info.column_list_by_type

exclude_cols = {
    "BODYWEIGHT",
    "ID_SCHEDA",
    "TIME OF DEATH",
    "YEAR",
    "AGE",
    "MONTH",
    "AGE_bin_id",
    "SEX_enc",
    "SEXUAL STATUS_enc"
    # FIXED COLS
    # 'Osmolal Gap', 'TLI', 'pH (quantitative)', 'Serum Total Bilirubin', 'Lipase/Crea', 'D Dimer',
    # 'RETICULOCYTE COUNT'
}

CAT_FEATURES = {
    "FILARIOSI_enc",
    "ANAMNESI_AMBIENTALE_enc",
    "VACCINAZIONI_enc",
    "ANAMNESI_ALIMENTARE_enc",
    "PROFILO_PAZIENTE_enc",
}

BOOL_FEATURES = {
    "SEX_enc",
    "SEXUAL STATUS_enc",
}
# cat_features_list = set()
# bool_features_list = set()
# num_features_list = df_info.med_exam_col_list - df_info.metadata_cols - exclude_cols -
#       CAT_FEATURES - BOOL_FEATURES
# cols_by_type.numerical_cols - exclude_cols - CAT_FEATURES - BOOL_FEATURES

color_list = (
    "#34c616",  # Verde
    "#735911",  # Marrone
    "#12b6c4",  # Azzurro
)
test_color_list = ("#da2e1a", "#da971a", "#1a43da")  # Rosso  # Arancione  # Blu

group_values_to_be_shown = (
    "BREED",
    (tuple(["MONGREL"]), tuple(["LABRADOR RETRIEVER"])),
)
# ['GERMAN SHEPHERD'], ['GOLDEN RETRIEVER']],
random_state = 42
num_cols = cols_by_type.numerical_cols.union(cols_by_type.num_categorical_cols).union(
    cols_by_type.bool_cols
)

umap_exp = UmapExperiment(
    df_info=df_info,
    n_neighbors=5,
    min_distance=0.01,
    not_nan_percentage_threshold=0.88,
    train_test_split_ratio=0.2,
    feature_to_color="AGE_bin_id",
    multi_marker_feats=("SEX_enc", "SEXUAL STATUS_enc"),
    enc_value_to_str_map={
        "SEX_enc": {0: "F", 1: "M"},
        "SEXUAL STATUS_enc": {0: "I", 1: "NI"},
    },
    file_title_prefix="AGE",
    exclude_feat_list=exclude_cols,
    numer_feat_list=num_cols,
    # TODO: We have a problem to understand which columns are numerical and categorical based on the
    #   count of unique values. At the moment even Serum Albumin is there.
    #   Moreover:
    #   1. Numer / categ /bool feat_list cannot share some columns names!
    categ_feat_list=(),  # cols_by_type.num_categorical_cols,
    bool_feat_list=(),  # BOOL_FEATURES,  # cols_by_type.bool_cols,
    random_seed=42,
    metric=tanimoto_gower,
    numer_feat_weight=1.0,
    categ_feat_weight=0,
    bool_feat_weight=0.0,
    group_values_to_be_shown=("BREED", (("LABRADOR RETRIEVER",), ("MONGREL",))),
    color_tuple=color_list,
    test_color_tuple=test_color_list,
    tooltip_feats=(
        "SEX",
        "SEXUAL STATUS",
        "AGE",
        "BODYWEIGHT",
    ),
    marker_size=8,
    marker_fill_alpha=0.0,
    tools=TOOLS,
)
# TODO:
#   2. Check the clustering labels and how they are split between train and test set
#    3. Check how the colors are passed and managed
# hdbscan_labels = umap_exp.clustering(min_cluster_size=10, umap_components=40,
#                                      min_samples=2, use_umap_preprocessing=True,)
# umap_exp.plot_cluster_labels(multi_marker_feats=('SEX_enc', 'SEXUAL STATUS_enc'))

umap_exp.fit_transform()
umap_exp.plot(return_plot=False)
