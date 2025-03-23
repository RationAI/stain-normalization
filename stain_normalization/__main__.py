from random import randint

import hydra
from lightning import seed_everything
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf
from rationai.mlkit import Trainer, autolog

from stain_normalization.callbacks.tiles_export import TilesExport
from stain_normalization.data import DataModule
from stain_normalization.stain_normalization_model import StainNormalizationModel


OmegaConf.register_new_resolver(
    "random_seed", lambda: randint(0, 2**31), use_cache=True
)


@hydra.main(config_path="../configs", config_name="default", version_base=None)
@autolog
def main(config: DictConfig, logger: Logger | None) -> None:
    seed_everything(config.seed, workers=True)

    data = hydra.utils.instantiate(
        config.data,
        _recursive_=False,  # to avoid instantiating all the datasets
        _target_=DataModule,
    )
    model = hydra.utils.instantiate(config.model, _target_=StainNormalizationModel)

    tile_export_callbeck = TilesExport(config.output_dir, config.data.predict.normalize)

    trainer = hydra.utils.instantiate(config.trainer, _target_=Trainer, logger=logger, callbacks=[tile_export_callbeck])
    getattr(trainer, config.mode)(model, datamodule=data, ckpt_path=config.checkpoint)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
