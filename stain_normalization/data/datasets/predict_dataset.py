from collections.abc import Iterable
import numpy as np
import pandas as pd
from albumentations import Transform3D
from albumentations.pytorch import ToTensorV2
from rationai.mlkit.data.datasets import MetaTiledSlides, OpenSlideTilesDataset
from torch.utils.data import Dataset
from stain_normalization.typing import PredictSample

class PredictDataset(MetaTiledSlides[PredictSample]):
    def __init__(
        self,
        uris: Iterable[str],
        normalize: Transform3D | None = None, 
    ) -> None:
        self.normalize = normalize
        super().__init__(uris=uris)

    def generate_datasets(self) -> Iterable[Dataset[PredictSample]]:
        return (
            _PredictSlideTiles(
                slide,
                tiles=self.filter_tiles_by_slide(slide["id"]),
                normalize=self.normalize,
            )
            for _, slide in self.slides.iterrows()
        )


class _PredictSlideTiles(Dataset[PredictSample]):
    def __init__(
        self,
        slide_metadata: pd.Series,
        tiles: pd.DataFrame,
        normalize: Transform3D | None = None,  
    ) -> None:
        super().__init__()
        self.slide_tiles = OpenSlideTilesDataset(
            slide_path=slide_metadata.path,
            level=slide_metadata.level,
            tile_extent_x=slide_metadata.tile_extent_x,
            tile_extent_y=slide_metadata.tile_extent_y,
            tiles=tiles,
        )
        
        self.normalize = normalize
        self.to_tensor = ToTensorV2()

    def __len__(self) -> int:
        return len(self.slide_tiles)

    def __getitem__(self, idx: int) -> PredictSample:
        input_image_255 = self.slide_tiles[idx]
        slide_name = self.slide_tiles.slide_path.stem
        level = self.slide_tiles.level
        x = self.slide_tiles.tiles.iloc[idx]["x"]
        y = self.slide_tiles.tiles.iloc[idx]["y"]

        input_image = input_image_255 / 255.0 

        if self.normalize:
            input_image = self.normalize(image=input_image)["image"]
        
        input_image = self.to_tensor(image=input_image)["image"]

        return input_image, {"original_image": input_image_255, 
        "modified_image": None, 
        "slide_name": slide_name, 
        "level":level ,
        "index": idx, 
        "xy":f"{x}_{y}",
        "name":f"{slide_name}-{x}-{y}"}


