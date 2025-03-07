from collections.abc import Iterable

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
        transforms: Transform3D | None = None,
        
    ) -> None:
        self.modify = modify
        self.transforms = transforms
        super().__init__(uris=uris)

    def generate_datasets(self) -> Iterable[Dataset[Sample]]:
        return (
            _StainNormalizationSlideTiles(
                slide,
                tiles=self.filter_tiles_by_slide(slide["id"]),
                include_target=True,
                modify=self.modify,
                transforms=self.transforms,
            )
            for _, slide in self.slides.iterrows()
        )


class StainNormalizationPredict(MetaTiledSlides[PredictSample]):
    def __init__(
        self,
        uris: Iterable[str],
        modify: Transform3D,
        transforms: Transform3D | None = None,

    ) -> None:        
        self.modify = modify
        self.transforms = transforms
        super().__init__(uris=uris)

    def generate_datasets(self) -> Iterable[Dataset[PredictSample]]:
        return (
            _StainNormalizationSlideTiles(
                slide,
                tiles=self.filter_tiles_by_slide(slide["id"]),
                include_target=False,
                modify=self.modify,
                transforms=self.transforms,
            )
            for _, slide in self.slides.iterrows()
        )


class _StainNormalizationSlideTiles(Dataset[Sample | PredictSample]):
    def __init__(
        self,
        slide_metadata: pd.Series,
        tiles: pd.DataFrame,
        include_target: bool,
        modify: Transform3D,
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
        self.transforms = transforms
        self.include_target = include_target
        self.to_tensor = ToTensorV2()
        self.modify = modify

    def __len__(self) -> int:
        return len(self.slide_tiles)

    def __getitem__(self, idx: int) -> Sample | PredictSample:
        original_image = self.slide_tiles[idx]
        

        if self.transforms is not None:
            original_image = self.transforms(image=original_image)["image"]

        modified_image = self.modify(image=original_image)["image"]

        # modification_name = "Original"
        # if self.modify:
        #     out = self.modify(original_image)
        #     modified_image = out["image"]
        #     modification_name = out["modification_name"]  
        # metadata = Metadata(
        #         slide=self.slide_tiles.slide_path.stem,
        #         x=self.slide_tiles.tiles.iloc[idx]["x"],
        #         y=self.slide_tiles.tiles.iloc[idx]["y"],
        #         modification=modification_name
        #     )

        original_image = self.to_tensor(image=original_image)["image"].float()
        modified_image = self.to_tensor(image=modified_image)["image"].float()

        if self.include_target:
            return modified_image, original_image
        
        return modified_image
