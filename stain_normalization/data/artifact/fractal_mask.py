import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import label
from skimage.transform import resize


class FractalMask:
    """Generate a binary artifact mask from value noise (fractal Brownian
     motion) as it is faster than Perlin noise.

    Attributes:
        base: Periods of the coarsest noise octave (region size and count).
        octaves: Number of noise octaves summed (edge detail).
        persistence: Amplitude decay per octave (edge roughness).
        coverage_range: Range for the fraction of the tile kept before filtering.
        keep_range: Range for the number of largest regions to keep.
        splat_range: Range for the number of small splatter regions to keep.
        splat_frac: Max size, as a fraction of the tile, that counts as splatter.
    """

    def __init__(
        self,
        base: int = 3,
        octaves: int = 6,
        persistence: float = 0.7,
        coverage_range: tuple[float, float] = (0.21, 0.23),
        keep_range: tuple[int, int] = (1, 3),
        splat_range: tuple[int, int] = (0, 6),
        splat_frac: float = 0.006,
    ):
        self.base = base
        self.octaves = octaves
        self.persistence = persistence
        self.coverage_range = coverage_range
        self.keep_range = keep_range
        self.splat_range = splat_range
        self.splat_frac = splat_frac

    def generate(self, height: int, width: int) -> NDArray[np.bool_]:
        """Generate a mask of the given size.

        Args:
            height: Mask height in pixels.
            width: Mask width in pixels.

        Returns:
            Boolean array where True marks the artifact region(s).
        """
        field = self.noise(height, width)
        coverage = np.random.uniform(*self.coverage_range)
        mask = field > np.quantile(field, 1 - coverage)

        n_main = np.random.randint(self.keep_range[0], self.keep_range[1] + 1)
        n_splat = np.random.randint(self.splat_range[0], self.splat_range[1] + 1)
        return self.keep_regions(mask, n_main, n_splat, height * width)

    def noise(self, height: int, width: int) -> NDArray[np.float64]:
        """Sum of bicubic-upsampled random grids of rising frequency, in [0, 1]."""
        noise = np.zeros((height, width))
        amplitude, total = 1.0, 0.0
        for octave in range(self.octaves):
            frequency = self.base * (2**octave)
            grid = np.random.rand(frequency, frequency)
            noise += amplitude * resize(
                grid, (height, width), order=3, mode="reflect", anti_aliasing=False
            )
            total += amplitude
            amplitude *= self.persistence

        noise /= total
        return (noise - noise.min()) / (np.ptp(noise) + 1e-9)

    def keep_regions(
        self, mask: NDArray[np.bool_], n_main: int, n_splat: int, area: int
    ) -> NDArray[np.bool_]:
        """Keep the ``n_main`` largest regions plus ``n_splat`` small splatters."""
        labels, num = label(mask)
        if num == 0:
            return mask

        sizes = sorted(
            ((i, int((labels == i).sum())) for i in range(1, num + 1)),
            key=lambda pair: -pair[1],
        )
        keep = [i for i, _ in sizes[:n_main]]
        small = [i for i, size in sizes[n_main:] if size <= self.splat_frac * area]
        np.random.shuffle(small)
        keep += small[:n_splat]
        return np.isin(labels, keep)
