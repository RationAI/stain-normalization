from pathlib import Path

import pyvips
import ray
from openslide import PROPERTY_NAME_MPP_X, PROPERTY_NAME_MPP_Y, OpenSlide
from rationai.masks import process_items, tissue_mask, write_big_tiff


# folders:
# archive tumor cases
# chive negative cases
# Prospective negative cases
# Prospective test cases
# Prospective tumor cases

SLIDES_PATH = "/mnt/data/MOU/prostate/tile_level_annotations/"
MASK_DEST = "./mask/tissue_masks"
LEVEL = 3


@ray.remote
def process_slide(slide_path: Path) -> None:
    # pyvips.Image xres and yres variables don't respect level therefore we
    # extract proper spatial resulution based on desired level using OpenSlide
    with OpenSlide(slide_path) as slide:
        downsample = slide.level_downsamples[LEVEL]
        xres = 1000 / (float(slide.properties[PROPERTY_NAME_MPP_X]) * downsample)
        yres = 1000 / (float(slide.properties[PROPERTY_NAME_MPP_Y]) * downsample)

    slide = pyvips.Image.new_from_file(slide_path, level=LEVEL)
    mask = tissue_mask(slide, xres)
    mask_path = Path(MASK_DEST, f"{Path(slide_path).stem}.tiff")
    mask_path.parent.mkdir(exist_ok=True, parents=True)
    write_big_tiff(mask, path=mask_path, mpp_x=xres, mpp_y=yres)



def main() -> None:
    slides = list(Path(SLIDES_PATH).rglob("*.mrxs"))
    # process_slide(slides[0])
    process_items(slides, process_item=process_slide)

if __name__ == "__main__":
    main()
