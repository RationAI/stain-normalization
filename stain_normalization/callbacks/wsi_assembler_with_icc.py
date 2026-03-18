"""Callback for assembling predicted tiles into whole-slide pyramid TIFFs."""

import tempfile
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lightning import LightningModule, Trainer
from omegaconf import DictConfig
from PIL.ImageCms import createProfile

from stain_normalization.callbacks._base import NormalizationCallback


def _srgb_icc_bytes() -> bytes:
    """Generate sRGB ICC profile bytes for embedding in TIFFs."""
    raw = createProfile("sRGB")
    # Pillow < 10: .tobuffer(), Pillow >= 10: .tobytes()
    try:
        return raw.tobuffer()  # type: ignore[attr-defined]  # deprecated in Pillow >=10
    except AttributeError:
        from PIL.ImageCms import ImageCmsProfile

        return ImageCmsProfile(raw).tobytes()


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


class WSIAssembler(NormalizationCallback):
    """Assembles predicted tiles back into whole-slide pyramid TIFFs."""

    def __init__(
        self,
        output_dir: str | Path,
        normalization_config: DictConfig,
        temp_dir: str | Path | None = None,
    ) -> None:
        super().__init__(normalization_config)
        self.output_dir = Path(output_dir)
        self.temp_dir = str(temp_dir) if temp_dir else None
        self._slide_meta: dict[str, _SlideMeta] = {}
        self._active: _SlideBuffers | None = None
        self._active_name: str | None = None

    def on_predict_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        slides_df = trainer.datamodule.predict.slides  # type: ignore[attr-defined]  # Lightning stub gap

        # Cache metadata only — buffers are opened lazily per slide
        for _, row in slides_df.iterrows():
            name = Path(row.path).stem
            self._slide_meta[name] = _SlideMeta(
                path=row.path,
                level=int(row.level),
                extent_x=int(row.extent_x),
                extent_y=int(row.extent_y),
                tile_extent_x=int(row.tile_extent_x),
                tile_extent_y=int(row.tile_extent_y),
                mpp_x=float(row.mpp_x),
                mpp_y=float(row.mpp_y),
            )

    def _open_slide(self, slide_name: str) -> None:
        """Allocate memmap buffers for one slide."""
        meta = self._slide_meta[slide_name]
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
        try:
            self._save_slide(self._active_name, self._active)
        except Exception:
            traceback.print_exc()
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
        outputs: list[torch.Tensor],
        batch: tuple[torch.Tensor, list[dict[str, Any]]],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        for b in range(len(outputs)):
            metadata = batch[1][b]
            slide_name = metadata["slide_name"]

            if slide_name not in self._slide_meta:
                print(f"Unknown slide '{slide_name}', skipping tile.")
                continue

            if slide_name != self._active_name:
                if self._active_name is not None:
                    print(f"Slide transition: {self._active_name} → {slide_name}")
                    self._close_slide()
                self._open_slide(slide_name)

            tile = self.tensor_to_image(outputs[b])
            x, y = (int(v) for v in metadata["xy"].split("_"))
            self._place_tile(tile, x, y)

    def _place_tile(self, tile: np.ndarray[Any, Any], x: int, y: int) -> None:
        """Place a predicted tile into the active slide buffer with overlap averaging."""
        assert self._active is not None
        sb = self._active
        ex, ey = sb.meta.extent_x, sb.meta.extent_y

        h, w = min(tile.shape[0], ey - y), min(tile.shape[1], ex - x)
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
        self._close_slide()
        self._slide_meta.clear()

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

        final_img.set_type(
            pyvips.GValue.blob_type, "icc-profile-data", _srgb_icc_bytes()
        )

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
