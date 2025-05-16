from typing import Any, TypeAlias

from torch import Tensor


PredictSample: TypeAlias = tuple[Tensor, dict[str, Any]]
Sample: TypeAlias = tuple[Tensor, Tensor]

Input: TypeAlias = Sample

Outputs: TypeAlias = Tensor


