from typing import TypeAlias, TypedDict,  Dict, Any

from torch import Tensor

PredictSample: TypeAlias = tuple[Tensor, Dict[str, Any]]
Sample: TypeAlias = tuple[Tensor, Tensor]

Input: TypeAlias = Sample

Outputs: TypeAlias = Tensor


