from pathlib import Path

import mlflow
import ray
from rationai.tiling import tiling
from rationai.tiling.modules.masks import PyvipsMask
from rationai.tiling.modules.tile_sources import OpenSlideTileSource
from rationai.tiling.typing import TiledSlideMetadata, TileMetadata
from rationai.tiling.writers import save_mlflow_dataset
from sklearn.model_selection import train_test_split


SLIDES_PATH = "/mnt/data/scans/AI scans/Prostata/"
TISSUE_MASKS_PATH = "./data/tissue_masks"


# level avg_mpp_x   avg_mpp_y
# 0     0.233876    0.234331
# 1     0.467751    0.468661
# 2     0.935503    0.937323
# 3     1.871006    1.874646
# 4     3.742012    3.749291
# 5     7.484024    7.498583
# 6     14.968047   14.997165
# 7     29.936095   29.994331
# 8     59.872189   59.988661
# 9     119.744379  119.97732
SlideMPP = 0.46

source = OpenSlideTileSource(mpp=SlideMPP, tile_extent=512, stride=256)


TISSUE_PERCENTAGE = 0.5


class TissueMask(PyvipsMask[TileMetadata]):
    def forward_tile(
        self, tile_labels: TileMetadata, class_overlaps: dict[int, float]
    ) -> TileMetadata | None:
        if class_overlaps.get(0, 0) > TISSUE_PERCENTAGE:
            return None
        return tile_labels


tissue_mask = TissueMask(
    tile_extent=source.tile_extent, absolute_roi_extent=256, relative_roi_offset=0
)


@ray.remote
def handler(slide_path: Path) -> TiledSlideMetadata:
    slide, tiles = source(slide_path)

    tissue_mask_path = Path(TISSUE_MASKS_PATH, slide_path.name[:-5] + ".tiff")
    tiles = tissue_mask(tissue_mask_path, slide.extent, tiles)

    return slide, tiles

BROKEN_SLIDES = {
    'P-2016_3829-04-1.mrxs', 'P-2016_3732-06-1.mrxs', 'P-2016_3732-03-1.mrxs', 'P-2016_3760-06-1.mrxs', 'P-2016_3629-13-1.mrxs', 'P-2016_3926-03-1.mrxs',
    'P-2016_3852-02-1.mrxs', 'P-2016_3988-07-1.mrxs', 'P-2016_3852-01-1.mrxs',
    'P-2016_3851-02-1.mrxs', 'P-2016_3629-12-1.mrxs', 'P-2016_3667-10-0.mrxs', 'P-2016_3606-04-1.mrxs', 'P-2016_3597-10-1.mrxs', 'P-2016_3988-10-1.mrxs', 'P-2016_3829-01-1.mrxs', 'P-2016_3926-02-1.mrxs', 'P-2019_3025-03-1.mrxs',
    'P-2016_3760-04-1.mrxs', 'P-2016_3732-04-1.mrxs', 'P-2016_3667-09-1.mrxs', 'P-2019_3292-06-1.mrxs', 'P-2016_3988-02-1.mrxs', 'P-2016_3667-11-0.mrxs', 'P-2016_3606-05-1.mrxs', 'P-2016_3851-09-1.mrxs', 'P-2016_3627-09-1.mrxs', 'P-2016_3606-03-1.mrxs', 'P-2016_3627-10-1.mrxs', 'P-2016_3852-03-1.mrxs', 'P-2016_3851-07-1.mrxs', 'P-2016_3829-03-1.mrxs', 'P-2016_3627-06-1.mrxs', 'P-2016_3597-09-1.mrxs', 'P-2016_3926-01-1.mrxs', 'P-2016_3629-11-1.mrxs', 'P-2016_3760-03-1.mrxs'}

def main() -> None:
    folders = [
        "archive tumor cases",
        "archive negative cases", 
        "Prospective negative cases", 
        "Prospective test cases", 
        "Prospective tumor cases" 
        ]

    slides = []
    for folder in folders:
        for slide in Path(SLIDES_PATH, folder).rglob("*.mrxs"):
            if slide.name in BROKEN_SLIDES:
                continue
            slides.append(slide)

    slides, test_slides = train_test_split(
        slides, test_size=0.2
    )
    train_slides, val_slides = train_test_split(slides, test_size=0.1)

    train_slides_df, train_tiles_df = tiling(
        slides=train_slides, handler=handler)
    val_slides_df, val_tiles_df = tiling(
        slides=list(val_slides), handler=handler)
    test_slides_df, test_tiles_df = tiling(
        slides=list(test_slides), handler=handler)
    
    train_slides_df.to_csv("./data/datasets/train_slides.csv", index=False)
    train_tiles_df.to_csv("./data/datasets/train_tiles.csv", index=False)

    val_slides_df.to_csv("./data/datasets/val_slides.csv", index=False)
    val_tiles_df.to_csv("./data/datasets/val_tiles.csv", index=False)

    test_slides_df.to_csv("./data/datasets/test_slides.csv", index=False)
    test_tiles_df.to_csv("./data/datasets/test_tiles.csv", index=False)

    mlflow.set_experiment(experiment_name="Stain-Normalization")
    with mlflow.start_run(run_name="Stain Normalization Dataset") as _:
        save_mlflow_dataset(
            slides=train_slides_df,
            tiles=train_tiles_df,
            dataset_name="Stain Normalization - train",
        )
        save_mlflow_dataset(
            slides=val_slides_df, tiles=val_tiles_df, dataset_name="Stain Normalization - val"
        )
        save_mlflow_dataset(
            slides=test_slides_df,
            tiles=test_tiles_df,
            dataset_name="Stain Normalization - test",
        )


if __name__ == "__main__":
    # pass
    main()
