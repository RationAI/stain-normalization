import numpy as np
from numpy.typing import NDArray
from perlin_noise import PerlinNoise
from PIL import Image

from rationai.staining import ColorConversion, convert_color
from rationai.staining.typing import RGBArray


ZEROS = np.zeros(shape=(512, 512), dtype=np.float64)
H_RGB = np.array([88, 76, 180], dtype=np.uint8)  # from QuPath
E_RGB = np.array([199, 50, 112], dtype=np.uint8)  # from QuPath
DAB_RGB = np.array([186, 110, 56], dtype=np.uint8)  # from QuPath


def _stretch(x: NDArray[np.float64], down: float, up: float) -> NDArray[np.float64]:
    low, high = np.min(x), np.max(x)

    return (x - low) * (up - down) / (high - low) + down


def _noise_array(
    size: int, params: list[tuple[int | None, float]]
) -> NDArray[np.float64]:
    result = np.zeros(shape=(size, size), dtype=np.float64)

    for seed, octaves in params:
        noise = PerlinNoise(octaves=octaves, seed=seed)

        arr = np.array(
            [[noise([i / size, j / size]) for j in range(size)] for i in range(size)],
            dtype=np.float64,
        )
        arr = _stretch(arr, -1, 1)
        result += arr

    result[result < 0] = 1
    result[(result >= 0) & (result < 0.5)] = 2
    result[(result >= 0.5) & (result < 0.85)] = 3
    result[(result >= 0.85) & (result < 1)] = 4

    return result


def artificial_image(c0: RGBArray, c1: RGBArray, conv: ColorConversion) -> RGBArray:
    noise = _noise_array(size=512, params=[(None, 10), (None, 4)])

    c0_pure = convert_color(c0, conv)[0]
    c1_pure = convert_color(c1, conv)[1]

    c0_noise = np.where((noise == 3) | (noise == 4), c0_pure, 0)
    c1_noise = np.where((noise >= 2) & (noise < 4), c1_pure, 0)

    img = np.stack([c0_noise, c1_noise, ZEROS], axis=-1)

    return np.asarray(convert_color(img, conv.inverse), dtype=np.uint8)


def main() -> None:
    h_e = artificial_image(H_RGB, E_RGB, ColorConversion.RGB2HER)
    Image.fromarray(h_e).save("tests/data/generated/h_e.jpg")

    h_dab = artificial_image(H_RGB, DAB_RGB, ColorConversion.RGB2HDR)
    Image.fromarray(h_dab).save("tests/data/generated/h_dab.jpg")


if __name__ == "__main__":
    main()
