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

SLIDES_PATH = "/mnt/data/scans/AI scans/Prostata/"
MASK_DEST = "./data/tissue_masks"
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
    mask = tissue_mask(slide)
    mask_path = Path(MASK_DEST, f"{Path(slide_path).stem}.tiff")
    mask_path.parent.mkdir(exist_ok=True, parents=True)
    write_big_tiff(mask, path=mask_path, xres=xres, yres=yres)


BROKEN_SLIDES = {
    "P-2016_3829-04-1.mrxs",
    "P-2016_3732-06-1.mrxs",
    "P-2016_3732-03-1.mrxs",
    "P-2016_3760-06-1.mrxs",
    "P-2016_3629-13-1.mrxs",
    "P-2016_3926-03-1.mrxs",
    "P-2016_3852-02-1.mrxs",
    "P-2016_3988-07-1.mrxs",
    "P-2016_3852-01-1.mrxs",
    "P-2016_3851-02-1.mrxs",
    "P-2016_3629-12-1.mrxs",
    "P-2016_3667-10-0.mrxs",
    "P-2016_3606-04-1.mrxs",
    "P-2016_3597-10-1.mrxs",
    "P-2016_3988-10-1.mrxs",
    "P-2016_3829-01-1.mrxs",
    "P-2016_3926-02-1.mrxs",
    "P-2019_3025-03-1.mrxs",
    "P-2016_3760-04-1.mrxs",
    "P-2016_3732-04-1.mrxs",
    "P-2016_3667-09-1.mrxs",
    "P-2019_3292-06-1.mrxs",
    "P-2016_3988-02-1.mrxs",
    "P-2016_3667-11-0.mrxs",
    "P-2016_3606-05-1.mrxs",
    "P-2016_3851-09-1.mrxs",
    "P-2016_3627-09-1.mrxs",
    "P-2016_3606-03-1.mrxs",
    "P-2016_3627-10-1.mrxs",
    "P-2016_3852-03-1.mrxs",
    "P-2016_3851-07-1.mrxs",
    "P-2016_3829-03-1.mrxs",
    "P-2016_3627-06-1.mrxs",
    "P-2016_3597-09-1.mrxs",
    "P-2016_3926-01-1.mrxs",
    "P-2016_3629-11-1.mrxs",
    "P-2016_3760-03-1.mrxs",
}


def main() -> None:
    folders = [
        # "archive tumor cases", #DONE
        # "archive negative cases", # DONE
        # "Prospective negative cases", #DONE
        # "Prospective test cases", # DONE
        # "Prospective tumor cases" # DONE
    ]

    for folder in folders:
        slides = []
        for slide in Path(SLIDES_PATH, folder).rglob("*.mrxs"):
            if slide.name in BROKEN_SLIDES:
                continue
            slides.append(slide)
        # slides.extend(list(Path(SLIDES_PATH, folder).rglob("*.mrxs")))

        # slides = list(Path(SLIDES_PATH).rglob("*.mrxs"))
        # process_slide(slides[0])
        process_items(slides, process_item=process_slide)
        print(f"Processed {len(slides)} slides from {folder} folder")


if __name__ == "__main__":
    main()
