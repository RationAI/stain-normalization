import tempfile
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lightning import LightningModule, Trainer
from omegaconf import DictConfig
from rationai.mlkit.lightning.callbacks import MultiloaderLifecycle

from stain_normalization.callbacks._base import DenormalizationCallback
from stain_normalization.type_aliases import Outputs


@dataclass
class _SlideMeta:
    path: str
    level: int
    extent_x: int
    extent_y: int
    tile_extent_x: int
    tile_extent_y: int
    mpp_x: float
    mpp_y: float


@dataclass
class _SlideBuffers:
    meta: _SlideMeta
    temp_dir: tempfile.TemporaryDirectory[str]
    result_buffer: np.memmap[Any, Any]
    count_buffer: np.memmap[Any, Any]


class WSIAssembler(DenormalizationCallback, MultiloaderLifecycle):
    """Assembles predicted tiles back into whole-slide pyramid TIFFs.

    Uses one dataloader per slide (via MultiloaderLifecycle) — buffers are
    opened on dataloader start and saved/freed on dataloader end.
    """

    def __init__(
        self,
        output_dir: str | Path,
        normalization_config: DictConfig,
        temp_dir: str | Path | None = None,
    ) -> None:
        DenormalizationCallback.__init__(self, normalization_config)
        MultiloaderLifecycle.__init__(self)
        self.output_dir = Path(output_dir)
        self.temp_dir = str(temp_dir) if temp_dir else None
        self._active: _SlideBuffers | None = None
        self._active_name: str | None = None
        self._failed_slides: list[str] = []

    def on_predict_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def on_predict_dataloader_start(
        self, trainer: Trainer, pl_module: LightningModule, dataloader_idx: int
    ) -> None:
        slide = trainer.datamodule.predict.slides.iloc[dataloader_idx]  # type: ignore[attr-defined]
        meta = _SlideMeta(
            path=slide.path,
            level=int(slide.level),
            extent_x=int(slide.extent_x),
            extent_y=int(slide.extent_y),
            tile_extent_x=int(slide.tile_extent_x),
            tile_extent_y=int(slide.tile_extent_y),
            mpp_x=float(slide.mpp_x),
            mpp_y=float(slide.mpp_y),
        )
        slide_name = Path(slide.path).stem
        self._open_slide(slide_name, meta)

    def on_predict_dataloader_end(
        self, trainer: Trainer, pl_module: LightningModule, dataloader_idx: int
    ) -> None:
        self._close_slide()

    def _open_slide(self, slide_name: str, meta: _SlideMeta) -> None:
        """Allocate memmap buffers for one slide."""
        h, w = meta.extent_y, meta.extent_x

        tmp = tempfile.TemporaryDirectory(
            prefix=f"wsi_{slide_name}_", dir=self.temp_dir
        )
        result_buf = np.memmap(
            Path(tmp.name) / "result.raw",
            dtype=np.uint8,
            mode="w+",
            shape=(h, w, 3),
        )
        count_buf = np.memmap(
            Path(tmp.name) / "count.raw",
            dtype=np.uint8,
            mode="w+",
            shape=(h, w),
        )

        self._active = _SlideBuffers(
            meta=meta,
            temp_dir=tmp,
            result_buffer=result_buf,
            count_buffer=count_buf,
        )
        self._active_name = slide_name

    def _close_slide(self) -> None:
        """Save and free the currently active slide."""
        if self._active is None:
            return
        assert self._active_name is not None
        slide_name = self._active_name
        try:
            self._save_slide(slide_name, self._active)
        except Exception:
            print(f"ERROR: Failed to save slide '{slide_name}'")
            traceback.print_exc()
            self._failed_slides.append(slide_name)
        finally:
            del self._active.result_buffer
            del self._active.count_buffer
            self._active.temp_dir.cleanup()
            self._active = None
            self._active_name = None

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Outputs,
        batch: tuple[torch.Tensor, list[dict[str, Any]]],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        for b in range(len(outputs)):
            tile = self.tensor_to_image(outputs[b])
            metadata = batch[1][b]
            x, y = (int(v) for v in metadata["xy"].split("_"))
            self._place_tile(tile, x, y)

    def _place_tile(self, tile: np.ndarray[Any, Any], x: int, y: int) -> None:
        """Place a predicted tile into the active slide buffer with overlap averaging."""
        assert self._active is not None
        sb = self._active
        ex, ey = sb.meta.extent_x, sb.meta.extent_y

        h = max(0, min(tile.shape[0], ey - y))
        w = max(0, min(tile.shape[1], ex - x))
        if h == 0 or w == 0:
            return
        tile = tile[:h, :w]

        region = sb.result_buffer[y : y + h, x : x + w]
        count = sb.count_buffer[y : y + h, x : x + w]

        # Running average: avg = (old * n + new) / (n + 1)
        overlap = count > 0
        if overlap.any():
            n = count[:, :, np.newaxis].astype(np.float32)
            blended = np.where(
                overlap[:, :, np.newaxis],
                (region.astype(np.float32) * n + tile) / (n + 1),
                tile,
            )
            sb.result_buffer[y : y + h, x : x + w] = np.clip(blended, 0, 255).astype(
                np.uint8
            )
        else:
            sb.result_buffer[y : y + h, x : x + w] = tile

        sb.count_buffer[y : y + h, x : x + w] = count + 1

    def on_predict_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self._failed_slides:
            print(
                f"WARNING: Failed to save {len(self._failed_slides)} slide(s): "
                f"{self._failed_slides}"
            )

    def _save_slide(self, slide_name: str, sb: _SlideBuffers) -> None:
        # Imported here — module-level import causes OpenSlide segfault (libtiff conflict).
        import pyvips

        meta = sb.meta
        sb.result_buffer.flush()
        sb.count_buffer.flush()

        result_path = Path(sb.temp_dir.name) / "result.raw"
        count_path = Path(sb.temp_dir.name) / "count.raw"

        result_img = pyvips.Image.rawload(
            str(result_path), meta.extent_x, meta.extent_y, 3
        )
        result_img = result_img.copy(interpretation=pyvips.Interpretation.SRGB)

        count_img = pyvips.Image.rawload(
            str(count_path), meta.extent_x, meta.extent_y, 1
        )
        mask = count_img > 0
        # add white background for untouched areas (count=0)
        white = (pyvips.Image.black(meta.extent_x, meta.extent_y, bands=3) + 255).cast(
            pyvips.BandFormat.UCHAR
        )
        final_img = mask.ifthenelse(result_img, white)

        output_path = self.output_dir / f"{slide_name}_norm.tiff"
        final_img.tiffsave(
            str(output_path),
            bigtiff=True,
            compression=pyvips.enums.ForeignTiffCompression.DEFLATE,
            tile=True,
            tile_width=512,
            tile_height=512,
            pyramid=True,
            xres=1000.0 / meta.mpp_x,
            yres=1000.0 / meta.mpp_y,
        )
