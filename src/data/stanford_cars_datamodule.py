import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from typing import Any, Dict, Optional
import os

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split

from src.data.components.stanford_cars_dataset import StanfordCarsCustomDataset
from src.data.components.transforms import img_classification_transform


class StanfordCarsDataModule(LightningDataModule):
    """Example of LightningDataModule for MNIST dataset.

    A DataModule implements 6 key methods:
        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 64,
        train_val_test_split: list = [0.5, 0.5],
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        self.transforms = img_classification_transform

        self.data_train: Optional[StanfordCarsCustomDataset] = None
        self.data_val: Optional[StanfordCarsCustomDataset] = None
        self.data_test: Optional[StanfordCarsCustomDataset] = None

    @property
    def num_classes(self):
        return 196

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        print("hparams:", self.hparams)

        StanfordCarsCustomDataset(
            self.hparams.data_dir, train=True, transforms=self.transforms
        )

        subdirectories = ["cars_train", "cars_test"]
        for subdir in subdirectories:
            current_dir = (
                f"{self.hparams.data_dir}/stanford-cars-dataset/{subdir}/{subdir}/"
            )
            for item in os.listdir(current_dir):
                if not item.endswith(".jpg"):
                    os.remove(f"{current_dir}{item}")

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = StanfordCarsCustomDataset(
                self.hparams.data_dir, train=True, transforms=self.transforms
            )
            testset = StanfordCarsCustomDataset(
                self.hparams.data_dir, train=False, transforms=self.transforms
            )

            self.data_val, self.data_test = random_split(
                dataset=testset,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    _ = StanfordCarsDataModule()
    dm = StanfordCarsDataModule(data_dir="data/", batch_size=64)
    dm.prepare_data()
    dm.setup()
    dl = dm.train_dataloader()
    x, y = next(iter(dl))
    print(y)
