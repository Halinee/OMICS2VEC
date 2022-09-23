from typing import Optional

import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from dataset.omics import OmicsDataset


class O2VDataModule(pl.LightningDataModule):
    def __init__(
        self,
        processed_data_path: str,
        processed_data_name: str,
        processed_label_name: str,
        batch_size: int = 32,
        test_size: float = 0.2,
        num_workers: int = 4,
        pin_memory: bool = True,
        drop_last: bool = True,
        seed: int = 42,
        **kwargs,
    ):
        super(O2VDataModule, self).__init__()
        self.batch_size = batch_size
        self.test_size = test_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.seed = seed
        self.dataset = OmicsDataset(
            processed_data_path,
            processed_data_name,
            processed_label_name,
        )
        self.train_idx = None
        self.valid_idx = None

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_idx, self.valid_idx = train_test_split(
            range(len(self.dataset)), test_size=self.test_size, random_state=self.seed
        )

    def generate_dataloader(self, split: str, shuffle: bool) -> DataLoader:
        splits = {"train": self.train_idx, "valid": self.valid_idx}
        dataset = Subset(self.dataset, splits[split])

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def train_dataloader(self) -> DataLoader:
        return self.generate_dataloader(split="train", shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self.generate_dataloader(split="valid", shuffle=False)
