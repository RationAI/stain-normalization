from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from rationai.staining.modify_and_reconstruct import modify_and_reconstruct


Channel: TypeAlias = NDArray[np.float64]


DIRECTORY_PATH = "/mnt/data/Projects/MOU/Bad_scans/stain separation/"
FILES = Path(DIRECTORY_PATH).glob("*.JPG")

# STAIN DEFINITIONS
HEMATOXYLIN = (0.849, 0.514, 0.124)
DAB = (0.262, 0.631, 0.730)
RED = (0.128, 0.875, 0.466)


def modify(c0: Channel, c1: Channel, c2: Channel) -> tuple[Channel, Channel, Channel]:
    c2 = np.zeros_like(c2)

    return c0, c1, c2


def main() -> None:
    for file in FILES:
        img = np.asarray(Image.open(file).convert("RGB"))
        modified = modify_and_reconstruct(img, modify, HEMATOXYLIN, DAB, RED)

        Image.fromarray(modified).save(f"{file.stem}.png")


if __name__ == "__main__":
    main()
