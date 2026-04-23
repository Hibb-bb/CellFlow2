import warnings
from pandas.errors import SettingWithCopyWarning

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", SettingWithCopyWarning)

import argparse
import json
import sys
from pathlib import Path
from typing import Any

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

import yaml


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


def _load_run_config() -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="zb_fish CellFlow pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "zb_fish.yaml",
        help="YAML with paths.paths.*, paths.metadata_json, checkpoint.*",
    )
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"zb_fish.py: ignoring unrecognized CLI args: {unknown}", file=sys.stderr)
    config_path = args.config.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open() as f:
        return yaml.safe_load(f)


RUN_CONFIG = _load_run_config()
_ckpt = RUN_CONFIG.get("checkpoint") or {}
_default_ckpt_dir = Path(__file__).resolve().parent / "zb_fish_checkpoints"
CHECKPOINT_DIR = Path(_ckpt.get("dir", _default_ckpt_dir))
CHECKPOINT_FILE_PREFIX = str(_ckpt.get("file_prefix", "zb_fish"))
CHECKPOINT_PATH = CHECKPOINT_DIR / f"{CHECKPOINT_FILE_PREFIX}_CellFlow.pkl"


def concat_h5ad_dir(directory: Path) -> ad.AnnData:
    """Load every ``*.h5ad`` in ``directory`` (sorted) and concatenate along obs."""
    directory = directory.expanduser().resolve()
    h5ad_paths = sorted(directory.glob("*.h5ad"))
    if not h5ad_paths:
        raise FileNotFoundError(f"No .h5ad files found in {directory}")
    return ad.concat([ad.read_h5ad(p) for p in h5ad_paths], axis=0, merge="same")


def _assert_obs_columns(adata: ad.AnnData, label: str) -> None:
    missing = [c for c in (PERT_COL, DONOR_COL, TIME_COL) if c not in adata.obs.columns]
    if missing:
        raise KeyError(
            f"{label}: missing required obs columns: {missing}. "
            f"Available obs columns (first 50): {list(map(str, adata.obs.columns[:50]))}"
        )


def _apply_condition_columns(adata: ad.AnnData) -> None:
    adata.obs[CONTROL_COL] = (adata.obs[PERT_COL].astype(str) == CONTROL_LABEL)
    adata.obs[CONDITION_COL] = (
        adata.obs[PERT_COL].astype(str)
        + "||"
        + adata.obs[DONOR_COL].astype(str)
        + "||"
        + adata.obs[TIME_COL].astype(str)
    )


def _copy_embedding_uns(src: ad.AnnData, dst: ad.AnnData) -> None:
    for key in ("donor_id_embeddings", "timepoint_embeddings", PERT_EMB_KEY):
        dst.uns[key] = src.uns[key]


def create_metadata(config: dict[str, Any] | None = None) -> tuple[ad.AnnData, ad.AnnData, ad.AnnData, str, ad.AnnData]:
    """Load train / test / validation from separate folders; return ref AnnData (concat) for analysis."""
    cfg = config if config is not None else RUN_CONFIG
    paths = cfg["paths"]

    train_dir = Path(paths["marson_train_dir"])
    test_dir = Path(paths["marson_test_dir"])
    validation_dir = Path(paths["marson_validation_dir"])
    de_stats_csv = Path(paths["de_stats_csv"])

    meta_path = paths.get("metadata_json")
    if meta_path is None or str(meta_path).lower() in ("null", "none", ""):
        marson_metadata_path = train_dir / "metadata.json"
    else:
        marson_metadata_path = Path(meta_path)

    adata_train = concat_h5ad_dir(train_dir)
    adata_test = concat_h5ad_dir(test_dir)
    adata_val = concat_h5ad_dir(validation_dir)

    for obj, name in (
        (adata_train, "train"),
        (adata_test, "test"),
        (adata_val, "validation"),
    ):
        _assert_obs_columns(obj, name)

    with marson_metadata_path.open() as f:
        marson_meta = json.load(f)

    sample_rep = "X_hvg" if "X_hvg" in adata_train.obsm_keys() else "X"

    donor_values: set[str] = set()
    for obj in (adata_train, adata_test, adata_val):
        donor_values.update(obj.obs[DONOR_COL].astype(str).unique())
    donor_cats = pd.Index(sorted(donor_values))

    time_cats = list(marson_meta["timepoints_included"])
    meta_time_vals = set(map(str, time_cats))
    meta_time_vals = set(["rest", "8hr", "48hr"])
    for obj, name in (
        (adata_train, "train"),
        (adata_test, "test"),
        (adata_val, "validation"),
    ):
        adata_time_vals = set(obj.obs[TIME_COL].astype(str).unique())
        if adata_time_vals - meta_time_vals:
            
            raise ValueError(
                f"{name} {TIME_COL} has values not listed in metadata timepoint: "
                f"{sorted(adata_time_vals - meta_time_vals)}"
            )

    adata_train.uns["donor_id_embeddings"] = {
        str(d): np.eye(len(donor_cats), dtype=float)[i] for i, d in enumerate(donor_cats)
    }
    adata_train.uns["timepoint_embeddings"] = {
        str(t): np.eye(len(time_cats), dtype=float)[i] for i, t in enumerate(time_cats)
    }

    _de_stats = pd.read_csv(de_stats_csv, usecols=["target_contrast_gene_name"])
    _from_csv = _de_stats["target_contrast_gene_name"].dropna().astype(str).unique().tolist()
    pert_values: set[str] = set()
    for obj in (adata_train, adata_test, adata_val):
        pert_values.update(obj.obs[PERT_COL].astype(str).unique())
    pert_genes = sorted(set(_from_csv) | pert_values | {CONTROL_LABEL})
    adata_train.uns[PERT_EMB_KEY] = {
        g: np.eye(len(pert_genes), dtype=float)[i] for i, g in enumerate(pert_genes)
    }

    _copy_embedding_uns(adata_train, adata_test)
    _copy_embedding_uns(adata_train, adata_val)

    _apply_condition_columns(adata_train)
    _apply_condition_columns(adata_test)
    _apply_condition_columns(adata_val)

    # Inject a subsample of training NTCs into val/test so _verify_control_data passes.
    # split_covariates=None means any NTC will do — no (donor, timepoint) matching needed.
    train_ctrls = adata_train[adata_train.obs[CONTROL_COL].to_numpy()].copy()
    rng = np.random.default_rng(0)
    n_ctrl = min(train_ctrls.n_obs, 5000)
    ctrl_sample = train_ctrls[rng.choice(train_ctrls.n_obs, n_ctrl, replace=False)].copy()

    adata_test = ad.concat([ctrl_sample, adata_test], axis=0, merge="same")
    adata_val  = ad.concat([ctrl_sample, adata_val],  axis=0, merge="same")

    adata_ref = ad.concat([adata_train, adata_test, adata_val], axis=0, merge="same")

    return adata_train, adata_test, adata_val, sample_rep, adata_ref


adata_train, adata_test, adata_val, sample_rep, adata = create_metadata()

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
    adata_test,
    name="test",
    n_conditions_on_log_iteration=None,
    n_conditions_on_train_end=None,
)

cf.prepare_validation_data(
    adata_val,
    name="val",
    n_conditions_on_log_iteration=10,
    n_conditions_on_train_end=10,
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
        num_iterations=150,
        batch_size=16,
        callbacks=callbacks,
        valid_freq=30_000,
    )

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
cf.save(str(CHECKPOINT_DIR), file_prefix=CHECKPOINT_FILE_PREFIX, overwrite=True)

_ed_keys = [
    k
    for k in ("train_e_distance_mean", "test_e_distance_mean", "val_e_distance_mean")
    if k in cf.trainer.training_logs
]
_colors = {"train_e_distance_mean": "blue", "test_e_distance_mean": "red", "val_e_distance_mean": "green"}
_labels = {
    "train_e_distance_mean": "Energy distance (train)",
    "test_e_distance_mean": "Energy distance (test)",
    "val_e_distance_mean": "Energy distance (val)",
}
n_ed = max(len(_ed_keys), 1)
fig, axes = plt.subplots(1, n_ed, figsize=(6 * n_ed, 6))
if n_ed == 1:
    axes = [axes]
for ax, key in zip(axes, _ed_keys):
    series = cf.trainer.training_logs[key]
    ax.plot(np.arange(len(series)), series, linestyle="-", color=_colors.get(key, "black"), label=_labels.get(key, key))
    ax.set_xlabel("Validation iteration" if key != "train_e_distance_mean" else "Iteration")
    ax.set_ylabel("Energy distance")
    ax.legend()
    ax.grid(True)


plt.tight_layout()
plt.savefig("stuff.png")

cf = CellFlow.load(str(CHECKPOINT_PATH))

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
    plt.savefig("ok2.png")
else:
    print(
        f"Skipping WKNN / line plots: obs column {CELL_TYPE_COL!r} not found. "
        "Set CELL_TYPE_COL to an existing annotation column."
    )
