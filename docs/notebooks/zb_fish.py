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


adata = cellflow.datasets.zesta(force_download=False)
adata = adata[adata.obs["tissue"] == "Central Nervous System"]


adata_train = adata[adata.obs["condition"]!="cdx4_cdx1a_24"].copy()
adata_test = adata[(adata.obs["condition"]=="cdx4_cdx1a_24") | (adata.obs["condition"]=="control_control_18")].copy()
# adata_train.n_obs, adata_test.n_obs


cf = CellFlow(adata_train, solver="otfm")



cf.prepare_data(
    sample_rep = "X_aligned",
    control_key = "first_t_control",
    perturbation_covariates = {"genetic_perturbation": ("gene_target_1" , "gene_target_2")},
    perturbation_covariate_reps = {"genetic_perturbation": "gene_embeddings"},
    sample_covariates = ("logtime",),
    split_covariates = None,
    max_combination_length = 2,
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
    "logtime": {"layer_type": "mlp", "dims": [512, 512], "dropout_rate": 0.0},
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

