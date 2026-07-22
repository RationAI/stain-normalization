from collections.abc import Iterable

import pandas as pd
from albumentations import Transform3D
from albumentations.pytorch import ToTensorV2
from rationai.mlkit.data.datasets import MetaTiledSlides, OpenSlideTilesDataset
from torch.utils.data import Dataset

from stain_normalization.data.artifact import Artifact
from stain_normalization.type_aliases import Sample


class TrainDataset(MetaTiledSlides[Sample]):
    """Dataset for training and validation.

    Kept as separate classes for distinct return types and clean Hydra config
    targets.
    Main difference is in __getitem__;

    Returns (modified_tensor, target_tensor) pairs — no metadata, no raw copies.

    Differences from TestDataset: no metadata returned, no raw image copies kept.
    Differences from PredictDataset: applies modify transform; no metadata.
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

    def generate_datasets(self) -> Iterable[Dataset[Sample]]:
        return (
            _TrainSlideTiles(
                slide,
                tiles=self.filter_tiles_by_slide(slide.id),
                modify=self.modify,
                normalize=self.normalize,
                artifact=self.artifact,
            )
            for _, slide in self.slides.iterrows()
        )


class _TrainSlideTiles(Dataset[Sample]):
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

    def __getitem__(self, idx: int) -> Sample:
        original_image_255 = self.slide_tiles[idx]

        # Create "wrong" image to use as input. Outputs image in float 0-1
        modified_image = self.modify(image=original_image_255)["image"]
        if self.artifact is not None:
            modified_image, target, _ = self.artifact.apply(original_image_255, modified_image)
        else:
            target = original_image_255 / 255.0

        if self.normalize:
            target = self.normalize(image=target)["image"]
            modified_image = self.normalize(image=modified_image)["image"]

        target = self.to_tensor(image=target)["image"]
        modified_image = self.to_tensor(image=modified_image)["image"]

        return modified_image, target  # type: ignore[return-value]  # untyped import
