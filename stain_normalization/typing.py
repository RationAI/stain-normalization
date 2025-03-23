from typing import TypeAlias, TypedDict,  Dict, Any

from torch import Tensor

# class Metadata(TypedDict):
#     slide: str
#     x: int
#     y: int
#     transormation: str
# PredictSample: TypeAlias = tuple[Tensor, Metadata]
# Sample: TypeAlias = tuple[Tensor, Tensor, Metadata]

PredictSample: TypeAlias = tuple[Tensor, Dict[str, Any]]
Sample: TypeAlias = tuple[Tensor, Tensor]

Input: TypeAlias = Sample

Outputs: TypeAlias = Tensor


