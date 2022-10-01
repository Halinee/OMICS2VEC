import os
import os.path as osp
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
import torch as th
from torch.utils.data import Dataset


class OmicsDataset(Dataset):
    def __init__(
        self,
        processed_data_path: str,
        processed_data_list: List[str],
        processed_label_name: str,
        processed_label_list: List[str],
    ):
        super(OmicsDataset, self).__init__()
        # Make directory
        if not osp.exists(processed_data_path):
            os.mkdir(processed_data_path)
        # Load omics data
        self.x = {}
        for data_type in processed_data_list:
            self.x[data_type] = th.as_tensor(
                self.load_data(processed_data_path, data_type + ".ftr").values,
                dtype=th.float32,
            )
        self.label = self.load_data(processed_data_path, processed_label_name)
        # Load label data
        self.y = {}
        for label_type in processed_label_list:
            self.y[label_type] = self.label[label_type].values
        # Load masking data
        processed_masking_list = ["masking_" + data for data in processed_data_list]
        self.masking = {}
        for data_type in processed_masking_list:
            self.masking[data_type] = self.label[data_type].values

    @staticmethod
    def load_data(path: str, name: str) -> pd.DataFrame:
        data = pd.read_feather(osp.join(path, name))
        data.set_index(data.columns[0], inplace=True)
        return data

    def __getitem__(self, idx: int) -> Tuple[Tuple[Any], np.ndarray, np.ndarray]:
        """
        data_type_index: Reference data_type in yaml file
        label_type_index: Reference label_type in yaml file
        """
        x = tuple(data[idx] for data in self.x.values())
        y = np.array([data[idx] for data in self.y.values()])
        masking = np.array([data[idx] for data in self.masking.values()])
        return (
            x,
            y,
            masking,
        )

    def __len__(self) -> int:
        return len(self.label)
