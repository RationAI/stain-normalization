import numpy as np

from rationai.staining.typing import StainTupleMatrix


def inv_mat(matrix: StainTupleMatrix) -> StainTupleMatrix:
    mat = np.linalg.inv(np.stack(matrix))
    return tuple(tuple(row) for row in mat)  # type: ignore [return-value]
