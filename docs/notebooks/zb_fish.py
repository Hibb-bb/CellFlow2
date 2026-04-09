import warnings
from pandas.errors import SettingWithCopyWarning

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", SettingWithCopyWarning)

import numpy as np
import pandas as pd
import seaborn as sns
import jax
import functools

# CellFlow uses JAX: GPU if jaxlib+cuda is installed, otherwise CPU.
print(f"JAX backend: {jax.default_backend()}, devices: {jax.devices()}")
import matplotlib.pyplot as plt
import anndata as ad
import scanpy as sc
import rapids_singlecell as rsc
import flax.linen as nn
import optax
import cellflow
from cellflow.model import CellFlow
import cellflow.preprocessing as cfpp
from cellflow.utils import match_linear
from cellflow.plotting import plot_condition_embedding
from cellflow.preprocessing import transfer_labels, compute_wknn, centered_pca, project_pca, reconstruct_pca
from cellflow.metrics import compute_r_squared, compute_e_distance


# ---- Marson dataset (minimal changes from original notebook) ----
MARSON_TRAIN_PATH = "/projects/b1094/ywl7940/CellFlow2/marson/train/train.h5ad"
adata = ad.read_h5ad(MARSON_TRAIN_PATH)

# Representation: prefer X_hvg in obsm (fallback to X)
sample_rep = "X_hvg" if "X_hvg" in adata.obsm_keys() else "X"

# Control definition: guide_target_gene_symbol == "NTC"
CONTROL_LABEL = "NTC"
CONTROL_COL = "is_control"
PERT_COL = "guide_target_gene_symbol"
# Full vocabulary in uns so test-only targets still resolve (avoids train-only sklearn OHE).
PERT_EMB_KEY = "guide_target_gene_symbol_embeddings"
DONOR_COL = "donor_id"
TIME_COL = "timepoint"
CONDITION_COL = "condition"
CELL_TYPE_COL = "cell_type_broad"

missing = [c for c in (PERT_COL, DONOR_COL, TIME_COL) if c not in adata.obs.columns]
if missing:
    raise KeyError(
        f"Missing required obs columns: {missing}. "
        f"Available obs columns (first 50): {list(map(str, adata.obs.columns[:50]))}"
    )

adata.obs[CONTROL_COL] = (adata.obs[PERT_COL].astype(str) == CONTROL_LABEL)
adata.obs[CONDITION_COL] = (
    adata.obs[PERT_COL].astype(str)
    + "||"
    + adata.obs[DONOR_COL].astype(str)
    + "||"
    + adata.obs[TIME_COL].astype(str)
)

# Simple one-hot embeddings for donor_id and timepoint
donor_cats = adata.obs[DONOR_COL].astype("category").cat.categories
time_cats = adata.obs[TIME_COL].astype("category").cat.categories

adata.uns["donor_id_embeddings"] = {
    d: np.eye(len(donor_cats), dtype=float)[i] for i, d in enumerate(donor_cats)
}
adata.uns["timepoint_embeddings"] = {
    t: np.eye(len(time_cats), dtype=float)[i] for i, t in enumerate(time_cats)
}

pert_genes = sorted(adata.obs[PERT_COL].astype(str).unique())
adata.uns[PERT_EMB_KEY] = {
    g: np.eye(len(pert_genes), dtype=float)[i] for i, g in enumerate(pert_genes)
}

# Simple train/test split (random) while keeping code structure similar
rs = np.random.RandomState(0)
mask_test = rs.rand(adata.n_obs) < 0.2
adata_train = adata[~mask_test].copy()
adata_test = adata[mask_test].copy()

cf = CellFlow(adata_train, solver="otfm")

cf.prepare_data(
    sample_rep=sample_rep,
    control_key=CONTROL_COL,
    perturbation_covariates={"genetic_perturbation": (PERT_COL,)},
    perturbation_covariate_reps={"genetic_perturbation": PERT_EMB_KEY},
    sample_covariates=(DONOR_COL, TIME_COL),
    sample_covariate_reps={
        DONOR_COL: "donor_id_embeddings",
        TIME_COL: "timepoint_embeddings",
    },
    split_covariates = None,
    max_combination_length=1,
    null_value = 0.0,
)

cf.prepare_validation_data(
    adata_train,
    name="train",
    n_conditions_on_log_iteration=10,
    n_conditions_on_train_end=10,
)

cf.prepare_validation_data(
    adata_test,
    name="test",
    n_conditions_on_log_iteration=None,
    n_conditions_on_train_end=None,
)

# Narrower than tutorial defaults to lower Slurm/JAX memory use (see batch_size below).
layers_before_pool = {
    "genetic_perturbation": {"layer_type": "mlp", "dims": [256, 256], "dropout_rate": 0.0},
    DONOR_COL: {"layer_type": "mlp", "dims": [256, 256], "dropout_rate": 0.0},
    TIME_COL: {"layer_type": "mlp", "dims": [256, 256], "dropout_rate": 0.0},
}

layers_after_pool = {
    "layer_type": "mlp", "dims": [512, 512], "dropout_rate": 0.0,
}

match_fn = functools.partial(match_linear, epsilon=0.5, tau_a=1.0, tau_b=1.0)

cf.prepare_model(
    condition_mode="deterministic",
    regularization=0.0,
    pooling="attention_token",
    layers_before_pool=layers_before_pool,
    layers_after_pool=layers_after_pool,
    condition_embedding_dim=128,
    cond_output_dropout=0.9,
    hidden_dims=[1024, 1024, 1024],
    decoder_dims=[2048, 2048, 2048],
    probability_path={"constant_noise": 0.5},
    match_fn=match_fn,
    )

metrics_callback = cellflow.training.Metrics(metrics=["mmd", "e_distance"])
callbacks = [metrics_callback]

cf.train(
        num_iterations=150_000,
        batch_size=512,
        callbacks=callbacks,
        valid_freq=30_000,
    )

e_distances_train = cf.trainer.training_logs["train_e_distance_mean"]
e_distances_test = cf.trainer.training_logs["test_e_distance_mean"]

iterations_train = np.arange(len(e_distances_train))
iterations_test  = np.arange(len(e_distances_test))

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

axes[0].plot(iterations_train, e_distances_train, linestyle='-', color='blue', label='Energy distance training data')
axes[0].set_xlabel('Iteration')
axes[0].set_ylabel('Energy distance')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(iterations_test, e_distances_test, linestyle='-', color='red', label='Energy distance test data')
axes[1].set_xlabel('Validation iteration')
axes[1].set_ylabel('Energy distance')
axes[1].legend()
axes[1].grid(True)


plt.tight_layout()
plt.savefig("stuff.png")

# --- Post-training analysis (201_zebrafish_continuous-style; Marson obs / categorical time) ---
covariate_data_train = (
    adata_train[~adata_train.obs[CONTROL_COL].to_numpy()]
    .obs.drop_duplicates(subset=[CONDITION_COL])
    .copy()
)
covariate_data_test = (
    adata_test[~adata_test.obs[CONTROL_COL].to_numpy()]
    .obs.drop_duplicates(subset=[CONDITION_COL])
    .copy()
)

df_embedding_train = cf.get_condition_embedding(
    covariate_data=covariate_data_train,
    condition_id_key=CONDITION_COL,
    rep_dict=adata_train.uns,
)[0]
df_embedding_test = cf.get_condition_embedding(
    covariate_data=covariate_data_test,
    condition_id_key=CONDITION_COL,
    rep_dict=adata_train.uns,
)[0]

df_embedding_train["seen_during_training"] = True
df_embedding_test["seen_during_training"] = False
df_condition_embedding = pd.concat((df_embedding_train, df_embedding_test))

fig = plot_condition_embedding(
    df_condition_embedding, embedding="PCA", hue="seen_during_training", circle_size=50
)

cov_meta = pd.concat([covariate_data_train, covariate_data_test]).drop_duplicates(CONDITION_COL)
cov_meta = cov_meta.set_index(CONDITION_COL)[[PERT_COL, DONOR_COL, TIME_COL]]
df_condition_embedding = df_condition_embedding.join(cov_meta, how="left")
df_condition_embedding[CONDITION_COL] = df_condition_embedding.index

fig = plot_condition_embedding(
    df_condition_embedding, embedding="PCA", hue=PERT_COL, circle_size=50
)
fig = plot_condition_embedding(
    df_condition_embedding, embedding="PCA", hue=TIME_COL, circle_size=50, legend=False
)


def duplicate_across_observed_timepoints(df, time_values):
    rows = []
    for _, row in df.iterrows():
        for t in time_values:
            r = row.copy()
            r[TIME_COL] = t
            r[CONDITION_COL] = f"{r[PERT_COL]}||{r[DONOR_COL]}||{t}"
            rows.append(r)
    return pd.DataFrame(rows)


time_values_sorted = sorted(adata.obs[TIME_COL].unique(), key=lambda x: (str(type(x)), str(x)))
covariate_data_interpolated = duplicate_across_observed_timepoints(
    covariate_data_test.iloc[[0]],
    time_values_sorted,
)

baseline_time = adata.obs[TIME_COL].iloc[0]
adata_ctrl = adata[
    adata.obs[CONTROL_COL].to_numpy() & (adata.obs[TIME_COL] == baseline_time)
].copy()

preds = cf.predict(
    adata=adata_ctrl,
    sample_rep=sample_rep,
    condition_id_key=CONDITION_COL,
    covariate_data=covariate_data_interpolated,
)

adata_preds = []
for cond, array in preds.items():
    obs_data = pd.DataFrame({CONDITION_COL: [cond] * array.shape[0]})
    adata_pred = ad.AnnData(X=np.empty((len(array), adata_train.n_vars)), obs=obs_data)
    adata_pred.obsm[sample_rep] = np.squeeze(array)
    adata_preds.append(adata_pred)

adata_preds = ad.concat(adata_preds)
adata_preds.var_names = adata_train.var_names

if CELL_TYPE_COL in adata.obs.columns:
    compute_wknn(
        ref_adata=adata,
        query_adata=adata_preds,
        n_neighbors=1,
        ref_rep_key=sample_rep,
        query_rep_key=sample_rep,
    )
    transfer_labels(
        query_adata=adata_preds, ref_adata=adata, label_key=CELL_TYPE_COL
    )
    transfer_col = f"{CELL_TYPE_COL}_transfer"
    df_pert_timepoint = (
        adata_preds.obs.groupby([CONDITION_COL])[transfer_col]
        .value_counts(normalize=True)
        .to_frame(name="fraction")
        .reset_index()
    )
    df_pert_timepoint[TIME_COL] = df_pert_timepoint[CONDITION_COL].map(
        covariate_data_interpolated.set_index(CONDITION_COL)[TIME_COL]
    )

    top_types = df_pert_timepoint[transfer_col].value_counts().head(2).index.tolist()
    n_plots = max(len(top_types), 1)
    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 3), sharey=True)
    if n_plots == 1:
        axes = [axes]
    for ax, cell_type in zip(axes, top_types):
        data = df_pert_timepoint[df_pert_timepoint[transfer_col] == cell_type]
        sns.lineplot(
            data=data,
            x=TIME_COL,
            y="fraction",
            marker="D",
            linestyle="--",
            linewidth=2,
            markersize=8,
            alpha=1.0,
            ax=ax,
        )
        ax.set_title(f"{cell_type} fraction over time", fontsize=10)
        ax.set_xlabel(str(TIME_COL), fontsize=10)
        ax.set_ylabel("Fraction" if ax == axes[0] else "", fontsize=10)
        ax.tick_params(labelsize=10)
    plt.tight_layout()
    plt.show()
else:
    print(
        f"Skipping WKNN / line plots: obs column {CELL_TYPE_COL!r} not found. "
        "Set CELL_TYPE_COL to an existing annotation column."
    )
