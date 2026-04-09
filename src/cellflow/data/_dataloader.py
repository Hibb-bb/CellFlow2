import abc
import json
import queue
import threading
from collections.abc import Generator
from typing import Any, Literal

import jax
import numpy as np

from cellflow.data._data import PredictionData, TrainingData, ValidationData

__all__ = [
    "TrainSampler",
    "MultiShardTrainSampler",
    "ValidationSampler",
    "PredictionSampler",
    "OOCTrainSampler",
]


class TrainSampler:
    """Data sampler for :class:`~cellflow.data.TrainingData`.

    Parameters
    ----------
    data
        The training data.
    batch_size
        The batch size.

    """

    def __init__(self, data: TrainingData, batch_size: int = 1024):
        self._data = data
        self._data_idcs = np.arange(data.cell_data.shape[0])
        self.batch_size = batch_size
        self.n_source_dists = data.n_controls
        self.n_target_dists = data.n_perturbations

        self._control_to_perturbation_keys = sorted(data.control_to_perturbation.keys())
        self._has_condition_data = data.condition_data is not None

    def _sample_target_dist_idx(self, source_dist_idx, rng):
        """Sample a target distribution index given the source distribution index."""
        return rng.choice(self._data.control_to_perturbation[source_dist_idx])

    def _get_embeddings(self, idx, condition_data) -> dict[str, np.ndarray]:
        """Get embeddings for a given index."""
        result = {}
        for key, arr in condition_data.items():
            result[key] = np.expand_dims(arr[idx], 0)
        return result

    def _sample_from_mask(self, rng, mask) -> np.ndarray:
        """Sample indices according to a mask."""
        # Convert mask to probability distribution
        valid_indices = np.where(mask)[0]

        # Handle case with no valid indices (should not happen in practice)
        if len(valid_indices) == 0:
            raise ValueError("No valid indices found in the mask")

        # Sample from valid indices with equal probability
        batch_idcs = rng.choice(valid_indices, self.batch_size, replace=True)
        return batch_idcs

    def sample(self, rng) -> dict[str, Any]:
        """Sample a batch of data.

        Parameters
        ----------
        seed : int, optional
            Random seed

        Returns
        -------
        Dictionary with source and target data
        """
        # Sample source distribution index
        source_dist_idx = rng.integers(0, self.n_source_dists)

        # Get source cells
        source_cells_mask = self._data.split_covariates_mask == source_dist_idx
        source_batch_idcs = self._sample_from_mask(rng, source_cells_mask)
        source_batch = self._data.cell_data[source_batch_idcs]

        target_dist_idx = self._sample_target_dist_idx(source_dist_idx, rng)
        target_cells_mask = self._data.perturbation_covariates_mask == target_dist_idx
        target_batch_idcs = self._sample_from_mask(rng, target_cells_mask)
        target_batch = self._data.cell_data[target_batch_idcs]

        if not self._has_condition_data:
            return {"src_cell_data": source_batch, "tgt_cell_data": target_batch}
        else:
            condition_batch = self._get_embeddings(target_dist_idx, self._data.condition_data)
            return {
                "src_cell_data": source_batch,
                "tgt_cell_data": target_batch,
                "condition": condition_batch,
            }

    @property
    def data(self):
        """The training data."""
        return self._data


class MultiShardTrainSampler:
    """Streaming multi-shard sampler producing the same batch dict as :class:`TrainSampler`.

    This sampler keeps only per-cell integer masks/indices in memory and fetches the
    actual cell representation rows on demand from `.h5ad` shards.

    Notes
    -----
    The provided metadata JSON must contain per-cell arrays needed for pairing.
    Expected keys (minimum):

    - ``rows_per_shard``: list[int]
    - ``split_covariates_mask``: list[int] (len = n_cells)
    - ``perturbation_covariates_mask``: list[int] (len = n_cells)
    - ``control_to_perturbation``: dict[str|int, list[int]]
    - ``condition_data``: dict[str, array-like] (optional; if omitted, no condition returned)
    """

    def __init__(
        self,
        *,
        shard_paths: list[str],
        metadata_path: str,
        batch_size: int = 1024,
        sample_rep: str = "X",
        dtype: str | None = "float32",
    ) -> None:
        if len(shard_paths) == 0:
            raise ValueError("`shard_paths` must be non-empty.")
        self.shard_paths = list(shard_paths)
        self.batch_size = int(batch_size)
        self.sample_rep = sample_rep
        self._dtype = np.dtype(dtype) if dtype is not None else None

        with open(metadata_path) as f:
            meta = json.load(f)

        rows_per_shard = meta.get("rows_per_shard", None)
        if rows_per_shard is None:
            raise KeyError("metadata is missing required key `rows_per_shard`.")
        self.rows_per_shard = np.asarray(rows_per_shard, dtype=np.int64)
        if len(self.rows_per_shard) != len(self.shard_paths):
            raise ValueError("`rows_per_shard` length must match `shard_paths` length.")

        self.shard_offsets = np.concatenate([[0], np.cumsum(self.rows_per_shard, dtype=np.int64)])
        self.n_cells = int(self.shard_offsets[-1])

        split_mask = meta.get("split_covariates_mask", None)
        pert_mask = meta.get("perturbation_covariates_mask", None)
        if split_mask is None or pert_mask is None:
            raise KeyError(
                "metadata must include `split_covariates_mask` and `perturbation_covariates_mask` (per-cell arrays)."
            )
        self.split_covariates_mask = np.asarray(split_mask, dtype=np.int32)
        self.perturbation_covariates_mask = np.asarray(pert_mask, dtype=np.int32)
        if len(self.split_covariates_mask) != self.n_cells or len(self.perturbation_covariates_mask) != self.n_cells:
            raise ValueError("per-cell mask lengths must match sum(rows_per_shard).")

        c2p = meta.get("control_to_perturbation", None)
        if c2p is None:
            raise KeyError("metadata is missing required key `control_to_perturbation`.")
        self.control_to_perturbation: dict[int, np.ndarray] = {
            int(k): np.asarray(v, dtype=np.int32) for k, v in c2p.items()
        }
        self._control_to_perturbation_keys = sorted(self.control_to_perturbation.keys())

        condition_data = meta.get("condition_data", None)
        self.condition_data: dict[str, np.ndarray] | None = None
        if condition_data is not None:
            self.condition_data = {k: np.asarray(v) for k, v in condition_data.items()}

        self._has_condition_data = self.condition_data is not None

        self.n_source_dists = int(np.max(self.split_covariates_mask)) + 1 if self.n_cells > 0 else 0
        self.n_target_dists = int(np.max(self.perturbation_covariates_mask)) + 1 if self.n_cells > 0 else 0

        # Build per-group indices once (int arrays only, no expression matrix).
        self._control_indices_by_source: list[np.ndarray] = [
            np.where(self.split_covariates_mask == s)[0].astype(np.int32) for s in range(self.n_source_dists)
        ]
        self._pert_indices_by_target: list[np.ndarray] = [
            np.where(self.perturbation_covariates_mask == t)[0].astype(np.int32) for t in range(self.n_target_dists)
        ]

        self._shards = None  # lazily opened backed AnnData objects

    def _open_shards(self) -> None:
        if self._shards is not None:
            return
        import anndata as ad

        self._shards = [ad.read_h5ad(p, backed="r") for p in self.shard_paths]

    def _sample_target_dist_idx(self, source_dist_idx: int, rng: np.random.Generator) -> int:
        return int(rng.choice(self.control_to_perturbation[source_dist_idx]))

    def _get_embeddings(self, idx: int) -> dict[str, np.ndarray]:
        if self.condition_data is None:
            raise ValueError("No condition data available.")
        result: dict[str, np.ndarray] = {}
        for key, arr in self.condition_data.items():
            result[key] = np.expand_dims(arr[idx], 0)
        return result

    def _sample_from_index(self, rng: np.random.Generator, idcs: np.ndarray) -> np.ndarray:
        if len(idcs) == 0:
            raise ValueError("No valid indices found for requested distribution.")
        return rng.choice(idcs, size=self.batch_size, replace=True).astype(np.int32)

    def _global_to_shard_local(self, global_idcs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # shard_id = max i such that offsets[i] <= gid
        shard_ids = np.searchsorted(self.shard_offsets[1:], global_idcs, side="right").astype(np.int32)
        local_ids = (global_idcs - self.shard_offsets[shard_ids]).astype(np.int64)
        return shard_ids, local_ids

    def _fetch_rows(self, global_idcs: np.ndarray) -> np.ndarray:
        self._open_shards()
        assert self._shards is not None

        shard_ids, local_ids = self._global_to_shard_local(global_idcs.astype(np.int64))
        out_chunks: list[np.ndarray] = [None] * len(global_idcs)  # type: ignore[list-item]

        # group by shard to minimize random IO overhead
        for shard_id in np.unique(shard_ids):
            shard_id_int = int(shard_id)
            positions = np.where(shard_ids == shard_id_int)[0]
            loc = local_ids[positions]

            shard = self._shards[shard_id_int]
            if self.sample_rep == "X":
                X = shard.X[loc, :]
            else:
                # only support obsm keys (common for PCA representations)
                X = shard.obsm[self.sample_rep][loc, :]

            X = np.asarray(X)
            if self._dtype is not None:
                X = X.astype(self._dtype, copy=False)

            for pos_i, row in zip(positions, X, strict=False):
                out_chunks[int(pos_i)] = row

        return np.stack(out_chunks, axis=0)

    def sample(self, rng: np.random.Generator) -> dict[str, Any]:
        source_dist_idx = int(rng.integers(0, self.n_source_dists))
        src_ids = self._sample_from_index(rng, self._control_indices_by_source[source_dist_idx])
        src_batch = self._fetch_rows(src_ids)

        target_dist_idx = self._sample_target_dist_idx(source_dist_idx, rng)
        tgt_ids = self._sample_from_index(rng, self._pert_indices_by_target[target_dist_idx])
        tgt_batch = self._fetch_rows(tgt_ids)

        if not self._has_condition_data:
            return {"src_cell_data": src_batch, "tgt_cell_data": tgt_batch}

        condition_batch = self._get_embeddings(target_dist_idx)
        return {"src_cell_data": src_batch, "tgt_cell_data": tgt_batch, "condition": condition_batch}


class BaseValidSampler(abc.ABC):
    @abc.abstractmethod
    def sample(*args, **kwargs):
        pass

    def _get_key(self, cond_idx: int) -> tuple[str, ...]:
        if len(self._data.perturbation_idx_to_id):  # type: ignore[attr-defined]
            return self._data.perturbation_idx_to_id[cond_idx]  # type: ignore[attr-defined]
        cov_combination = self._data.perturbation_idx_to_covariates[cond_idx]  # type: ignore[attr-defined]
        return tuple(cov_combination[i] for i in range(len(cov_combination)))

    def _get_perturbation_to_control(self, data: ValidationData | PredictionData) -> dict[int, np.ndarray]:
        d = {}
        for k, v in data.control_to_perturbation.items():
            for el in v:
                d[el] = k
        return d

    def _get_condition_data(self, cond_idx: int) -> dict[str, np.ndarray]:
        return {k: v[[cond_idx], ...] for k, v in self._data.condition_data.items()}  # type: ignore[attr-defined]


class ValidationSampler(BaseValidSampler):
    """Data sampler for :class:`~cellflow.data.ValidationData`.

    Parameters
    ----------
    val_data
        The validation data.
    seed
        Random seed.
    """

    def __init__(self, val_data: ValidationData, seed: int = 0) -> None:
        self._data = val_data
        self.perturbation_to_control = self._get_perturbation_to_control(val_data)
        self.n_conditions_on_log_iteration = (
            val_data.n_conditions_on_log_iteration
            if val_data.n_conditions_on_log_iteration is not None
            else val_data.n_perturbations
        )
        self.n_conditions_on_train_end = (
            val_data.n_conditions_on_train_end
            if val_data.n_conditions_on_train_end is not None
            else val_data.n_perturbations
        )
        self.rng = np.random.default_rng(seed)
        if self._data.condition_data is None:
            raise NotImplementedError("Validation data must have condition data.")

    def sample(self, mode: Literal["on_log_iteration", "on_train_end"]) -> Any:
        """Sample data for validation.

        Parameters
        ----------
        mode
            Sampling mode. Either ``"on_log_iteration"`` or ``"on_train_end"``.

        Returns
        -------
        Dictionary with source, condition, and target data from the validation data.
        """
        size = self.n_conditions_on_log_iteration if mode == "on_log_iteration" else self.n_conditions_on_train_end
        condition_idcs = self.rng.choice(self._data.n_perturbations, size=(size,), replace=False)

        source_idcs = [self.perturbation_to_control[cond_idx] for cond_idx in condition_idcs]
        source_cells_mask = [self._data.split_covariates_mask == source_idx for source_idx in source_idcs]
        source_cells = [self._data.cell_data[mask] for mask in source_cells_mask]
        target_cells_mask = [cond_idx == self._data.perturbation_covariates_mask for cond_idx in condition_idcs]
        target_cells = [self._data.cell_data[mask] for mask in target_cells_mask]
        conditions = [self._get_condition_data(cond_idx) for cond_idx in condition_idcs]
        cell_rep_dict = {}
        cond_dict = {}
        true_dict = {}
        for i in range(len(condition_idcs)):
            k = self._get_key(condition_idcs[i])
            cell_rep_dict[k] = source_cells[i]
            cond_dict[k] = conditions[i]
            true_dict[k] = target_cells[i]

        return {"source": cell_rep_dict, "condition": cond_dict, "target": true_dict}

    @property
    def data(self) -> ValidationData:
        """The validation data."""
        return self._data


class PredictionSampler(BaseValidSampler):
    """Data sampler for :class:`~cellflow.data.PredictionData`.

    Parameters
    ----------
    pred_data
        The prediction data.

    """

    def __init__(self, pred_data: PredictionData) -> None:
        self._data = pred_data
        self.perturbation_to_control = self._get_perturbation_to_control(pred_data)
        if self._data.condition_data is None:
            raise NotImplementedError("Validation data must have condition data.")

    def sample(self) -> Any:
        """Sample data for prediction.

        Returns
        -------
        Dictionary with source and condition data from the prediction data.
        """
        condition_idcs = range(self._data.n_perturbations)

        source_idcs = [self.perturbation_to_control[cond_idx] for cond_idx in condition_idcs]
        source_cells_mask = [self._data.split_covariates_mask == source_idx for source_idx in source_idcs]
        source_cells = [self._data.cell_data[mask] for mask in source_cells_mask]
        conditions = [self._get_condition_data(cond_idx) for cond_idx in condition_idcs]
        cell_rep_dict = {}
        cond_dict = {}
        for i in range(len(condition_idcs)):
            k = self._get_key(condition_idcs[i])
            cell_rep_dict[k] = source_cells[i]
            cond_dict[k] = conditions[i]

        return {
            "source": cell_rep_dict,
            "condition": cond_dict,
        }

    @property
    def data(self) -> PredictionData:
        """The training data."""
        return self._data


def prefetch_to_device(
    sampler: TrainSampler, seed: int, num_iterations: int, prefetch_factor: int = 2, num_workers: int = 4
) -> Generator[dict[str, Any], None, None]:
    seq = np.random.SeedSequence(seed)
    random_generators = [np.random.default_rng(s) for s in seq.spawn(num_workers)]

    q: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=prefetch_factor * num_workers)
    sem = threading.Semaphore(num_iterations)
    stop_event = threading.Event()

    def worker(rng: np.random.Generator):
        while not stop_event.is_set() and sem.acquire(blocking=False):
            batch = sampler.sample(rng)
            batch = jax.device_put(batch, jax.devices()[0], donate=True)
            jax.block_until_ready(batch)
            while not stop_event.is_set():
                try:
                    q.put(batch, timeout=1.0)
                    break  # Batch successfully put into the queue; break out of retry loop
                except queue.Full:
                    continue

        return

    # Start multiple worker threads
    ts = []
    for i in range(num_workers):
        t = threading.Thread(target=worker, daemon=True, name=f"worker-{i}", args=(random_generators[i],))
        t.start()
        ts.append(t)

    try:
        for _ in range(num_iterations):
            # Yield batches from the queue; will block waiting for available batch
            yield q.get()
    finally:
        # When the generator is closed or garbage collected, clean up the worker threads
        stop_event.set()  # Signal all workers to exit
        for t in ts:
            t.join()  # Wait for all worker threads to finish


class OOCTrainSampler:
    def __init__(
        self, data: TrainingData, seed: int, batch_size: int = 1024, num_workers: int = 4, prefetch_factor: int = 2
    ):
        self.inner = TrainSampler(data=data, batch_size=batch_size)
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.seed = seed
        self._iterator = None

    def set_sampler(self, num_iterations: int) -> None:
        self._iterator = prefetch_to_device(
            sampler=self.inner, seed=self.seed, num_iterations=num_iterations, prefetch_factor=self.prefetch_factor
        )

    def sample(self, rng=None) -> dict[str, Any]:
        if self._iterator is None:
            raise ValueError(
                "Sampler not set. Use `set_sampler` to set the sampler with"
                "the number of iterations. Without the number of iterations,"
                " the sampler will not be able to sample the data."
            )
        if rng is not None:
            del rng
        return next(self._iterator)
