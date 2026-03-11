from typing import Any, TypeAlias

from torch import Tensor


Sample: TypeAlias = tuple[Tensor, Tensor]
PredictSample: TypeAlias = tuple[Tensor, dict[str, Any]]

# Batches - after collate
Batch: TypeAlias = tuple[Tensor, Tensor]
PredictBatch: TypeAlias = tuple[Tensor, list[dict[str, Any]]]

Outputs: TypeAlias = Tensor
