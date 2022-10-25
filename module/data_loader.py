import os.path as osp
import pickle as pkl
from typing import List, Optional

import numpy as np
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from dataset.omics import OmicsDataset


class O2VDataModule(pl.LightningDataModule):
    def __init__(
        self,
        raw_data_path: str,
        processed_data_path: str,
        processed_data_name: List[str],
        processed_label_name: str,
        processed_label_list: List[str],
        test_index_data_name: str,
        data_split: str = "random",
        batch_size: int = 32,
        test_size: float = 0.2,
        num_workers: int = 4,
        pin_memory: bool = True,
        drop_last: bool = True,
        seed: int = 42,
        **kwargs,
    ):
        super(O2VDataModule, self).__init__()
        self.data_split = data_split
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
            processed_label_list,
        )
        self.train_idx = None
        self.test_idx = None
        with open(osp.join(raw_data_path, test_index_data_name), "rb") as f:
            self.test_sample = pkl.load(f)

    def setup(self, stage: Optional[str] = None) -> None:
        match self.data_split:
            case "random":
                self.train_idx, self.test_idx = train_test_split(
                    range(len(self.dataset)),
                    test_size=self.test_size,
                    random_state=self.seed,
                )
            case "completed":
                self.train_idx = []
                self.test_idx = []
                for i in range(len(self.dataset)):
                    if 0 in [data[i] for data in self.dataset.masking.values()]:
                        self.test_idx.append(i)
                    else:
                        self.train_idx.append(i)
            case "fix":
                data_idx = np.array(self.dataset.label.index)
                self.test_idx = [
                    np.where(data_idx == i)[0][0] for i in self.test_sample
                ]
                self.train_idx = list(set(range(len(data_idx))) - set(self.test_idx))
            case _:
                raise ValueError(
                    "Only random, completed, fix can be the value of data_split."
                )

    def generate_dataloader(self, split: str, shuffle: bool) -> DataLoader:
        splits = {"train": self.train_idx, "test": self.test_idx}
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
        return self.generate_dataloader(split="test", shuffle=False)
