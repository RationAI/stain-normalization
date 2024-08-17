from numpy import float64
from numpy.linalg import norm
from numpy.typing import NDArray


def normalize(x: NDArray[float64]) -> NDArray[float64]:
    """Normalizes a given vector to unit length."""
    n = norm(x)
    return x if n == 0 else x / n
