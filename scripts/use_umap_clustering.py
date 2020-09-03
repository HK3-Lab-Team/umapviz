import logging
import os
from pathlib import Path

from pd_extras.dataframe_with_info import import_df_with_info_from_file
from src.umapviz.umap_cluster import UmapClustering
from src.umapviz.umap_metrics import tanimoto_gower

logging.basicConfig(
    format="%(asctime)s \t %(levelname)s \t Module: %(module)s \t %(message)s ",
    datefmt="%d-%b-%y %H:%M:%S",
    level=logging.INFO,
)

# -----------------------
# UMAP (DEFAULT) SETTINGS
# -----------------------
TOOLS = (
    "hover,crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,"
    "undo,redo,reset,tap,save,box_select,poly_select,lasso_select,"
)
TOOLTIPS = [
    ("index", "$index"),
    ("(x,y)", "($x, $y)"),
    ("study_UID", "@study_uid"),
]

CWD = Path(os.path.abspath(os.path.dirname("__file__"))).parent
DATA_PATH = CWD / "umap_analysis" / "data"
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
    # 'SEX_enc',
    # 'SEXUAL STATUS_enc',
}

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
feature_to_color = "AGE_bin_id"
multi_feat_to_combine_partit_list = ("SEX_enc", "SEXUAL STATUS_enc")
filename_prefix = "AGE_15_0.1_with_test"
random_state = 42
metric = tanimoto_gower

umap_exp = UmapClustering(
    df_info=df_info,
    n_neighbors=15,
    min_distance=0.1,
    not_nan_percentage_threshold=0.88,
    train_test_split_ratio=0.0,
    feature_to_color="AGE_bin_id",
    multi_marker_feats=("SEX_enc", "SEXUAL STATUS_enc"),
    enc_value_to_str_map={
        "SEX_enc": {0: "F", 1: "M"},  # TODO: Check these!!!!!
        "SEXUAL STATUS_enc": {0: "I", 1: "NI"},
    },
    file_title_prefix="AGE",
    exclude_feat_list=exclude_cols,
    numer_feat_list=cols_by_type.numerical_cols,
    # TODO: We have a problem to understand which columns are numerical and categorical based on the
    #   count of unique values. At the moment even Serum Albumin is there.
    #   Moreover:
    #   1. Numer / categ /bool feat_list cannot share some columns names!
    # TODO:  Understand why self.feature_weights make the metric fail!!!!!!!!!!!!!!
    categ_feat_list=(),  # cols_by_type.num_categorical_cols,
    bool_feat_list=(),  # cols_by_type.bool_cols,
    random_seed=42,
    metric=tanimoto_gower,
    numer_feat_weight=1.0,
    categ_feat_weight=0,
    bool_feat_weight=0,
    group_values_to_be_shown=("BREED", (("LABRADOR RETRIEVER", "MONGREL"),)),
    color_tuple=color_list,
    test_color_tuple=test_color_list,
    tooltip_feats=(
        "SEX",
        "SEXUAL STATUS",
        "AGE",
        "BODYWEIGHT",
    ),
    tools=TOOLS,
)
# TODO: 1. Use the embedding from M-dimension clustering for further UMAP and plotting
#   2. Check the clustering labels and how they are split between train and test set
#    3. Check how the colors are passed and managed
hdbscan_labels = umap_exp.clustering(
    min_cluster_size=15,
    umap_components=5,
    min_samples=2,
    use_umap_preprocessing=True,
)
umap_exp.plot_cluster_labels(multi_marker_feats=("SEX_enc", "SEXUAL STATUS_enc"))
