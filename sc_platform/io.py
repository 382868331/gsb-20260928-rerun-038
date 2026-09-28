from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .models import CountsDataset


def build_dataset(
    dataset_id: str,
    matrix: Sequence[Sequence[int]] | np.ndarray | pd.DataFrame,
    gene_ids: Sequence[str],
    cell_ids: Sequence[str],
    batches: Mapping[str, Sequence[str]],
    integer: bool = True,
) -> CountsDataset:
    """Build raw data from a small manual matrix.

    ``batches`` is an explicit sample manifest, so empty samples can be
    represented as an empty list instead of disappearing.
    """
    array = pd.DataFrame(matrix, index=list(gene_ids), columns=list(cell_ids))
    if integer and not all(pd.api.types.is_integer_dtype(s) for s in [array[col] for col in array.columns]):
        if not np.all(array.to_numpy() == array.to_numpy().astype(int)):
            raise ValueError("count matrix must contain integer values")
        array = array.astype(int)
    batch_manifest = {batch: tuple(cells) for batch, cells in batches.items()}
    cell_batches = {cell: batch for batch, cells in batch_manifest.items() for cell in cells}
    return CountsDataset(
        dataset_id=dataset_id,
        counts=array,
        cell_batches=cell_batches,
        batch_cell_ids=batch_manifest,
    )
