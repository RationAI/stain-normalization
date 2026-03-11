"""Compare a dataset against a reference image or between two datasets.

Usage:
  # Compare dataset against a reference image
  python analyze_dataset.py --reference ref.png --uri "mlflow-artifacts:/79/..."

  # Compare two datasets (e.g. original vs normalized)
  # Assume that both datasets have the same slides and tiles in the same order for paired comparison
  python analyze_dataset.py --original "mlflow-artifacts:/79/...original" \
                            --compared "mlflow-artifacts:/79/...normalized"

  # Subsample for faster run
  python analyze_dataset.py --reference ref.png --uri "mlflow-artifacts:/79/..." --max-tiles 5000
"""

import argparse
from collections.abc import Generator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from rationai.mlkit.data.datasets import MetaTiledSlides, OpenSlideTilesDataset
from tqdm import tqdm

from stain_normalization.analysis import StainAnalyzer
from stain_normalization.analysis.report import REPORT_METRICS


def load_image(path: str | Path) -> np.ndarray[Any, Any]:
    return np.array(Image.open(path).convert("RGB"))


def iterate_tiles(
    slides: pd.DataFrame, tiles: pd.DataFrame
) -> Generator[tuple[str, Any, str], None, None]:
    """Yield (slide_name, tile_uint8, image_id) for each tile."""
    for _, slide in slides.iterrows():
        slide_name = Path(slide.path).stem
        slide_tiles = tiles[tiles["slide_id"] == slide["id"]]

        if slide_tiles.empty:
            continue

        dataset = OpenSlideTilesDataset(
            slide_path=slide.path,
            level=slide.level,
            tile_extent_x=slide.tile_extent_x,
            tile_extent_y=slide.tile_extent_y,
            tiles=slide_tiles,
        )

        for i in range(len(dataset)):
            image_id = (
                f"{slide_name}_{slide_tiles.iloc[i]['x']}_{slide_tiles.iloc[i]['y']}"
            )
            yield slide_name, dataset[i], image_id


def run_reference_mode(args: argparse.Namespace) -> tuple[StainAnalyzer, int]:
    """Compare all tiles in a dataset against a single reference image."""
    ref_img = load_image(args.reference)
    slides, tiles = MetaTiledSlides.load_slides_and_tiles(paths=[], uris=args.uri)
    print(f"Dataset: {len(slides)} slides, {len(tiles)} tiles")
    print(f"Reference: {args.reference}")

    if args.max_tiles and len(tiles) > args.max_tiles:
        tiles = tiles.sample(n=args.max_tiles, random_state=42)
        print(f"Subsampled to {args.max_tiles} tiles")

    analyzer = StainAnalyzer(reference=ref_img)
    for _, tile, image_id in tqdm(iterate_tiles(slides, tiles), total=len(tiles)):
        analyzer.compare(tile, image_id=image_id)

    return analyzer, len(analyzer.results)


def run_paired_mode(args: argparse.Namespace) -> tuple[StainAnalyzer, int]:
    """Compare matching tiles between two datasets (original vs compared)."""
    orig_slides, orig_tiles = MetaTiledSlides.load_slides_and_tiles(
        paths=[], uris=args.original
    )
    comp_slides, comp_tiles = MetaTiledSlides.load_slides_and_tiles(
        paths=[], uris=args.compared
    )
    print(f"Original: {len(orig_slides)} slides, {len(orig_tiles)} tiles")
    print(f"Compared: {len(comp_slides)} slides, {len(comp_tiles)} tiles")

    if args.max_tiles and len(orig_tiles) > args.max_tiles:
        orig_tiles = orig_tiles.sample(n=args.max_tiles, random_state=42)
        comp_tiles = comp_tiles.loc[orig_tiles.index]
        print(f"Subsampled to {args.max_tiles} tile pairs")

    analyzer = StainAnalyzer()
    orig_iter = iterate_tiles(orig_slides, orig_tiles)
    comp_iter = iterate_tiles(comp_slides, comp_tiles)

    for (_, orig_tile, image_id), (_, comp_tile, _) in tqdm(
        zip(orig_iter, comp_iter, strict=False), total=len(orig_tiles)
    ):
        analyzer.compare(comp_tile, image_id=image_id, reference=orig_tile)

    return analyzer, len(analyzer.results)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dataset stain analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output",
        default="./analysis_output",
        help="Output directory (default: ./analysis_output)",
    )
    parser.add_argument(
        "--max-tiles", type=int, default=None, help="Limit number of tiles to analyze"
    )

    # Mode 1: reference image
    parser.add_argument("--reference", help="Path to reference image")
    parser.add_argument("--uri", nargs="+", help="MLflow dataset URI(s) to analyze")

    # Mode 2: two datasets
    parser.add_argument(
        "--original", nargs="+", help="MLflow URI(s) for original dataset"
    )
    parser.add_argument(
        "--compared", nargs="+", help="MLflow URI(s) for compared dataset"
    )

    args = parser.parse_args()

    if args.reference and args.uri:
        analyzer, count = run_reference_mode(args)
    elif args.original and args.compared:
        analyzer, count = run_paired_mode(args)
    else:
        parser.error("Use either (--reference + --uri) or (--original + --compared)")

    if analyzer is None:
        return

    print(f"\nAnalyzed {count} tiles")

    analyzer.save_csv(args.output)
    print(f"Results saved to: {args.output}/")

    stats = analyzer.get_statistics()
    print("\nStatistics:")
    for m in REPORT_METRICS:
        if m in stats.columns:
            print(f"  {m:25s}: mean={stats[m]['mean']:.4f}  std={stats[m]['std']:.4f}")


if __name__ == "__main__":
    main()
