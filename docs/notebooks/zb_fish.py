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
DONOR_COL = "donor_id"
TIME_COL = "timepoint"

missing = [c for c in (PERT_COL, DONOR_COL, TIME_COL) if c not in adata.obs.columns]
if missing:
    raise KeyError(
        f"Missing required obs columns: {missing}. "
        f"Available obs columns (first 50): {list(map(str, adata.obs.columns[:50]))}"
    )

adata.obs[CONTROL_COL] = (adata.obs[PERT_COL].astype(str) == CONTROL_LABEL)

# Simple one-hot embeddings for donor_id and timepoint
donor_cats = adata.obs[DONOR_COL].astype("category").cat.categories
time_cats = adata.obs[TIME_COL].astype("category").cat.categories

adata.uns["donor_id_embeddings"] = {
    d: np.eye(len(donor_cats), dtype=float)[i] for i, d in enumerate(donor_cats)
}
adata.uns["timepoint_embeddings"] = {
    t: np.eye(len(time_cats), dtype=float)[i] for i, t in enumerate(time_cats)
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
    perturbation_covariate_reps=None,
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

layers_before_pool = {
    "genetic_perturbation": {"layer_type": "mlp", "dims": [512, 512], "dropout_rate": 0.0},
    DONOR_COL: {"layer_type": "mlp", "dims": [512, 512], "dropout_rate": 0.0},
    TIME_COL: {"layer_type": "mlp", "dims": [512, 512], "dropout_rate": 0.0},
}

layers_after_pool = {
    "layer_type": "mlp", "dims": [1024, 1024], "dropout_rate": 0.0,
}

match_fn = functools.partial(match_linear, epsilon=0.5, tau_a=1.0, tau_b=1.0)

cf.prepare_model(
    condition_mode="deterministic",
    regularization=0.0,
    pooling="attention_token",
    layers_before_pool=layers_before_pool,
    layers_after_pool=layers_after_pool,
    condition_embedding_dim=256,
    cond_output_dropout=0.9,
    hidden_dims=[2048, 2048, 2048],
    decoder_dims=[4096, 4096, 4096],
    probability_path={"constant_noise": 0.5},
    match_fn=match_fn,
    )

metrics_callback = cellflow.training.Metrics(metrics=["mmd", "e_distance"])
callbacks = [metrics_callback]

cf.train(
        num_iterations=150_000,
        batch_size=2048,
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
plt.savefig("loss.png")

