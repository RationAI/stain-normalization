from collections.abc import Iterable

import numpy as np
import pandas as pd
from albumentations import Transform3D
from albumentations.pytorch import ToTensorV2
from rationai.mlkit.data.datasets import MetaTiledSlides, OpenSlideTilesDataset
from torch.utils.data import Dataset

from stain_normalization.typing import PredictSample, Sample


class StainNormalization(MetaTiledSlides[Sample]):
    def __init__(
        self,
        uris: Iterable[str],
        modify: Transform3D,
        normalize: Transform3D | None = None,  
        transforms: Transform3D | None = None,
    ) -> None:
        self.modify = modify
        self.transforms = transforms
        self.normalize = normalize
        super().__init__(uris=uris)

    def generate_datasets(self) -> Iterable[Dataset[Sample]]:
        return (
            _StainNormalizationTrainSlideTiles(
                slide,
                tiles=self.filter_tiles_by_slide(slide["id"]),
                modify=self.modify,
                normalize=self.normalize,
                transforms=self.transforms,
            )
            for _, slide in self.slides.iterrows()
        )


class StainNormalizationPredict(MetaTiledSlides[PredictSample]):
    def __init__(
        self,
        uris: Iterable[str],
        modify: Transform3D,
        normalize: Transform3D | None = None, 
        transforms: Transform3D | None = None,
    ) -> None:
        self.modify = modify
        self.normalize = normalize
        self.transforms = transforms
        super().__init__(uris=uris)

    def generate_datasets(self) -> Iterable[Dataset[PredictSample]]:
        return (
            _StainNormalizationPredictSlideTiles(
                slide,
                tiles=self.filter_tiles_by_slide(slide["id"]),
                modify=self.modify,
                normalize=self.normalize,
                transforms=self.transforms,
            )
            for _, slide in self.slides.iterrows()
        )


class _StainNormalizationTrainSlideTiles(Dataset[Sample]):
    def __init__(
        self,
        slide_metadata: pd.Series,
        tiles: pd.DataFrame,
        modify: Transform3D,
        normalize: Transform3D | None = None,  
        transforms: Transform3D | None = None,
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
        self.transforms = transforms
        self.to_tensor = ToTensorV2()

    def __len__(self) -> int:
        return len(self.slide_tiles)

    def __getitem__(self, idx: int) -> Sample:
        original_image = self.slide_tiles[idx]
        # Apply non-disruptive transformations such as rotation, flip
        if self.transforms is not None:
            original_image = self.transforms(image=original_image)["image"]

        # Create "wrong" image to use as input. Outputs image in float 0-1
        modified_image = self.modify(image=original_image)["image"]
        original_image = original_image / 255.0 

        if self.normalize:
            original_image = self.normalize(image=original_image)["image"]
            modified_image = self.normalize(image=modified_image)["image"]

        original_image = self.to_tensor(image=original_image)["image"]
        modified_image = self.to_tensor(image=modified_image)["image"]
        
        return modified_image, original_image
        


class _StainNormalizationPredictSlideTiles(Dataset[PredictSample]):
    def __init__(
        self,
        slide_metadata: pd.Series,
        tiles: pd.DataFrame,
        modify: Transform3D,
        normalize: Transform3D | None = None,  
        transforms: Transform3D | None = None,
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
        self.transforms = transforms
        self.to_tensor = ToTensorV2()

    def __len__(self) -> int:
        return len(self.slide_tiles)

    def __getitem__(self, idx: int) -> PredictSample:
        original_image_255 = self.slide_tiles[idx]
        # Apply non-disruptive transformations such as rotation, flip
        if self.transforms is not None:
            original_image_255 = self.transforms(image=original_image_255)["image"]

        # Create "wrong" image to use as input. Outputs image in float 0-1
        modified_image_raw = self.modify(image=original_image_255)["image"]
        modified_image = modified_image_raw
        original_image = original_image_255 / 255.0 

        if self.normalize:
            original_image = self.normalize(image=original_image)["image"]
            modified_image = self.normalize(image=modified_image)["image"]

        original_image = self.to_tensor(image=original_image)["image"]
        modified_image = self.to_tensor(image=modified_image)["image"]
        
        
        return modified_image, {"original_image": original_image_255, "modified_image": modified_image_raw, "index": idx}

