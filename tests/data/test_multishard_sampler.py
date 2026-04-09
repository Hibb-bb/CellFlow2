import json
from pathlib import Path

import anndata as ad
import numpy as np
import pytest

from cellflow.data._dataloader import MultiShardTrainSampler


def _write_h5ad(path: Path, X: np.ndarray) -> None:
    adata = ad.AnnData(X=X)
    adata.write_h5ad(path)


@pytest.mark.parametrize("batch_size", [1, 8])
def test_multishard_train_sampler_pairs_and_shapes(tmp_path: Path, batch_size: int) -> None:
    # Two shards, 5 rows each => global rows [0..9]
    shard0 = tmp_path / "adata_0.h5ad"
    shard1 = tmp_path / "adata_1.h5ad"

    X0 = np.arange(5 * 3, dtype=np.float32).reshape(5, 3)
    X1 = (100 + np.arange(5 * 3, dtype=np.float32)).reshape(5, 3)
    _write_h5ad(shard0, X0)
    _write_h5ad(shard1, X1)

    # Controls: global ids [0, 1, 2, 5, 6, 7] assigned to source_dist 0
    # Perturbed: global ids [3, 4, 8, 9] assigned to target_dist 0
    n_cells = 10
    split_covariates_mask = np.full((n_cells,), -1, dtype=np.int32)
    perturbation_covariates_mask = np.full((n_cells,), -1, dtype=np.int32)
    control_ids = np.array([0, 1, 2, 5, 6, 7], dtype=np.int32)
    pert_ids = np.array([3, 4, 8, 9], dtype=np.int32)
    split_covariates_mask[control_ids] = 0
    perturbation_covariates_mask[pert_ids] = 0

    meta = {
        "rows_per_shard": [5, 5],
        "split_covariates_mask": split_covariates_mask.tolist(),
        "perturbation_covariates_mask": perturbation_covariates_mask.tolist(),
        "control_to_perturbation": {"0": [0]},
        "condition_data": {"cond": [[1.0, 2.0]]},  # (n_targets=1, cond_dim=2)
    }
    meta_path = tmp_path / "metadata.json"
    meta_path.write_text(json.dumps(meta))

    sampler = MultiShardTrainSampler(
        shard_paths=[str(shard0), str(shard1)],
        metadata_path=str(meta_path),
        batch_size=batch_size,
        sample_rep="X",
        dtype="float32",
    )

    rng = np.random.default_rng(0)
    batch = sampler.sample(rng)

    assert set(batch.keys()) == {"src_cell_data", "tgt_cell_data", "condition"}
    assert batch["src_cell_data"].shape == (batch_size, 3)
    assert batch["tgt_cell_data"].shape == (batch_size, 3)
    assert batch["condition"]["cond"].shape == (1, 2)

    # Verify that sampled src rows come from the control pool, and tgt rows from the perturbed pool.
    control_rows = {tuple(r) for r in np.vstack([X0, X1])[control_ids]}
    pert_rows = {tuple(r) for r in np.vstack([X0, X1])[pert_ids]}

    for r in batch["src_cell_data"]:
        assert tuple(r) in control_rows
    for r in batch["tgt_cell_data"]:
        assert tuple(r) in pert_rows

