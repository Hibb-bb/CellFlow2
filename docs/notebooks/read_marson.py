#!/usr/bin/env python3

import argparse
import json
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd


def _safe_head(seq: list[str], n: int) -> list[str]:
    return seq[:n]


def _maybe_len(x: Any) -> int | None:
    try:
        return len(x)  # type: ignore[arg-type]
    except Exception:
        return None


def _series_summary(s: pd.Series, *, top_n: int, max_categories: int) -> dict[str, Any]:
    out: dict[str, Any] = {"dtype": str(s.dtype)}

    n_missing = int(s.isna().sum())
    if n_missing:
        out["n_missing"] = n_missing

    # Categoricals (including pandas "category")
    is_cat = pd.api.types.is_categorical_dtype(s.dtype)
    if is_cat:
        n_cat = int(s.cat.categories.size)  # type: ignore[union-attr]
        out["n_categories"] = n_cat
        if n_cat <= max_categories:
            vc = s.value_counts(dropna=False).head(top_n)
            out["top_values"] = {str(k): int(v) for k, v in vc.items()}
        return out

    # Booleans
    if pd.api.types.is_bool_dtype(s.dtype):
        vc = s.value_counts(dropna=False).head(top_n)
        out["top_values"] = {str(k): int(v) for k, v in vc.items()}
        return out

    # Numeric
    if pd.api.types.is_numeric_dtype(s.dtype):
        desc = s.describe(percentiles=[0.01, 0.5, 0.99])
        out["summary"] = {k: (None if pd.isna(v) else float(v)) for k, v in desc.to_dict().items()}
        return out

    # Strings / objects
    nunique = int(s.nunique(dropna=True))
    out["n_unique"] = nunique
    if nunique <= max_categories:
        vc = s.value_counts(dropna=False).head(top_n)
        out["top_values"] = {str(k): int(v) for k, v in vc.items()}
    return out


def summarize_h5ad(
    path: str,
    *,
    backed: str | None,
    head_cols: int,
    top_n: int,
    max_categories: int,
    show_uns: bool,
) -> dict[str, Any]:
    mode = None if backed is None else backed
    adata = ad.read_h5ad(path, backed=mode)

    out: dict[str, Any] = {
        "path": path,
        "shape": [int(adata.n_obs), int(adata.n_vars)],
        "X": {
            "type": type(adata.X).__name__,
            "dtype": (str(getattr(adata.X, "dtype", None)) if getattr(adata, "X", None) is not None else None),
        },
        "obs": {"n_cols": int(adata.obs.shape[1])},
        "var": {"n_cols": int(adata.var.shape[1])},
        "layers": sorted(list(adata.layers.keys())),
        "obsm": sorted(list(adata.obsm.keys())),
        "varm": sorted(list(adata.varm.keys())),
        "obsp": sorted(list(adata.obsp.keys())),
        "varp": sorted(list(adata.varp.keys())),
        "uns_keys": sorted(list(adata.uns.keys())),
    }

    # Basic shapes for matrices (avoid materializing backed arrays)
    def _mat_info(x: Any) -> dict[str, Any]:
        info: dict[str, Any] = {"type": type(x).__name__}
        shp = getattr(x, "shape", None)
        if shp is not None:
            info["shape"] = [int(shp[0]), int(shp[1])] if len(shp) == 2 else [int(v) for v in shp]
        dt = getattr(x, "dtype", None)
        if dt is not None:
            info["dtype"] = str(dt)
        return info

    out["layers_info"] = {k: _mat_info(adata.layers[k]) for k in _safe_head(out["layers"], head_cols)}
    out["obsm_info"] = {k: _mat_info(adata.obsm[k]) for k in _safe_head(out["obsm"], head_cols)}

    # Columns + light summaries
    obs_cols = list(map(str, adata.obs.columns))
    var_cols = list(map(str, adata.var.columns))
    out["obs"]["columns"] = _safe_head(obs_cols, head_cols)
    out["var"]["columns"] = _safe_head(var_cols, head_cols)

    out["obs"]["column_summaries"] = {
        c: _series_summary(adata.obs[c], top_n=top_n, max_categories=max_categories)
        for c in _safe_head(obs_cols, head_cols)
    }
    out["var"]["column_summaries"] = {
        c: _series_summary(adata.var[c], top_n=top_n, max_categories=max_categories)
        for c in _safe_head(var_cols, head_cols)
    }

    if show_uns:
        # Only show types / lengths to avoid dumping huge objects.
        uns_meta: dict[str, Any] = {}
        for k in out["uns_keys"]:
            v = adata.uns.get(k)
            meta: dict[str, Any] = {"type": type(v).__name__}
            l = _maybe_len(v)
            if l is not None:
                meta["len"] = int(l)
            if isinstance(v, (np.ndarray,)):
                meta["shape"] = [int(x) for x in v.shape]
                meta["dtype"] = str(v.dtype)
            uns_meta[k] = meta
        out["uns_meta"] = uns_meta

    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="Read an .h5ad file and print a concise summary of its contents.",
    )
    p.add_argument("h5ad_path", help="Path to the .h5ad file")
    p.add_argument(
        "--backed",
        choices=["r", "r+"],
        default=None,
        help="Open in backed mode to avoid loading X into memory (default: fully load).",
    )
    p.add_argument(
        "--head-cols",
        type=int,
        default=30,
        help="How many obs/var column names (and summaries) to show.",
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Top-N values to show for categorical/object columns.",
    )
    p.add_argument(
        "--max-categories",
        type=int,
        default=50,
        help="Only show value counts if number of unique categories is <= this.",
    )
    p.add_argument(
        "--show-uns",
        action="store_true",
        help="Include uns key metadata (types/lengths; avoids dumping full objects).",
    )
    p.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output.",
    )

    args = p.parse_args()

    summary = summarize_h5ad(
        args.h5ad_path,
        backed=args.backed,
        head_cols=args.head_cols,
        top_n=args.top_n,
        max_categories=args.max_categories,
        show_uns=args.show_uns,
    )

    if args.pretty:
        print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    else:
        print(json.dumps(summary, separators=(",", ":"), sort_keys=True, default=str))


if __name__ == "__main__":
    main()


# python3 read_marson.py  /projects/b1094/ywl7940/CellFlow2/marson/train/train.h5ad --backed r --head-cols 40 --show-uns --pretty


