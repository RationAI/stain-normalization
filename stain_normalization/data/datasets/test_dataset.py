from collections.abc import Iterable

import numpy as np
import pandas as pd
from albumentations import Transform3D
from albumentations.pytorch import ToTensorV2
from rationai.mlkit.data.datasets import MetaTiledSlides, OpenSlideTilesDataset
from torch.utils.data import Dataset

from stain_normalization.data.artifact import Artifact
from stain_normalization.type_aliases import PredictSample


class TestDataset(MetaTiledSlides[PredictSample]):
    """Dataset for testing and analysis.

    Same as TrainDataset but also returns a metadata dict with slide_name, xy
    coordinates, the mask, and raw copies of the target and modified images —
    needed for callbacks that export tiles or run analysis.

    Differences from TrainDataset: returns metadata dict alongside tensors;
    keeps raw image copies for export.
    Differences from PredictDataset: applies modify transform; real slides are
    not pre-modified.
    """

    def __init__(
        self,
        uris: Iterable[str],
        modify: Transform3D,
        normalize: Transform3D | None = None,
        artifact: Artifact | None = None,
    ) -> None:
        self.modify = modify
        self.normalize = normalize
        self.artifact = artifact
        super().__init__(uris=uris)

    def generate_datasets(self) -> Iterable[Dataset[PredictSample]]:
        return (
            _TestSlideTiles(
                slide,
                tiles=self.filter_tiles_by_slide(slide["id"]),
                modify=self.modify,
                normalize=self.normalize,
                artifact=self.artifact,
            )
            for _, slide in self.slides.iterrows()
        )


class _TestSlideTiles(Dataset[PredictSample]):
    def __init__(
        self,
        slide_metadata: pd.Series,
        tiles: pd.DataFrame,
        modify: Transform3D,
        normalize: Transform3D | None = None,
        artifact: Artifact | None = None,
    ) -> None:
        super().__init__()
        self.slide_tiles = OpenSlideTilesDataset(
            slide_path=slide_metadata.path,
            level=slide_metadata.level,
            tile_extent_x=slide_metadata.tile_extent_x,
            tile_extent_y=slide_metadata.tile_extent_y,
            tiles=tiles,
        )
        self.modify = modify
        self.normalize = normalize
        self.artifact = artifact
        self.to_tensor = ToTensorV2()

    def __len__(self) -> int:
        return len(self.slide_tiles)

    def __getitem__(self, idx: int) -> PredictSample:
        original_image_255 = self.slide_tiles[idx]
        slide_name = self.slide_tiles.slide_path.stem
        x = self.slide_tiles.tiles.iloc[idx]["x"]
        y = self.slide_tiles.tiles.iloc[idx]["y"]

        # Create "wrong" image to use as input. Outputs image in float 0-1
        modified_image = self.modify(image=original_image_255)["image"]
        if self.artifact is not None:
            modified_image, target, mask = self.artifact.apply(
                original_image_255, modified_image
            )
        else:
            target = original_image_255 / 255.0
            mask = np.zeros(original_image_255.shape[:2], dtype=bool)

        # copies for export, before normalize
        target_image = target
        modified_image_raw = modified_image
        # real data in 255, if artifact wasnt applied it's the same as targget_imageso we skip it
        original_image_raw = original_image_255 if mask.any() else None

        if self.normalize:
            target = self.normalize(image=target)["image"]
            modified_image = self.normalize(image=modified_image)["image"]

        target = self.to_tensor(image=target)["image"]
        modified_image = self.to_tensor(image=modified_image)["image"]

        return (
            modified_image,
            {
                "target_tensor": target,
                "target_image": target_image,
                "modified_image": modified_image_raw,
                "original_image": original_image_raw,
                "mask": mask,
                "slide_name": slide_name,
                "xy": f"{x}_{y}",
            },
        )
