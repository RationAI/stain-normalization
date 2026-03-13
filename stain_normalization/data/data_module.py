from collections.abc import Iterable
from typing import Any

import torch
from hydra.utils import instantiate
from lightning import LightningDataModule
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import DataLoader

from stain_normalization.type_aliases import Batch, PredictBatch


def collate_fn(batch: list[tuple[Tensor, Any]]) -> tuple[Tensor, list[Any]]:
    return torch.stack([x[0] for x in batch]), [x[1] for x in batch]


class DataModule(LightningDataModule):
    def __init__(
        self, batch_size: int, num_workers: int = 0, **datasets: DictConfig
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.datasets = datasets

    def setup(self, stage: str) -> None:
        match stage:
            case "fit":
                self.train = instantiate(self.datasets["train"])
                self.val = instantiate(self.datasets["val"])
            case "validate":
                self.val = instantiate(self.datasets["val"])
            case "test":
                self.test = instantiate(self.datasets["test"])
            case "predict":
                self.predict = instantiate(self.datasets["predict"])

    def train_dataloader(self) -> Iterable[Batch]:
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> Iterable[Batch]:
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> Iterable[PredictBatch]:
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def predict_dataloader(self) -> Iterable[PredictBatch]:
        return DataLoader(
            self.predict,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )
