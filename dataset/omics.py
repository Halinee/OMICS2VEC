import os
import os.path as osp
from typing import Tuple

import numpy as np
import pandas as pd
import torch as th
from torch.utils.data import Dataset


class OmicsDataset(Dataset):
    def __init__(
        self,
        processed_data_path: str,
        processed_data_name: str,
        processed_label_name: str,
    ):
        super(OmicsDataset, self).__init__()
        # Make directory
        if not osp.exists(processed_data_path):
            os.mkdir(processed_data_path)
        # Load data
        x_path = osp.join(processed_data_path, processed_data_name)
        x = pd.read_feather(x_path)
        x.set_index(x.columns[0], inplace=True)
        self.data = np.array(x)
        y_path = osp.join(processed_data_path, processed_label_name)
        y = pd.read_feather(y_path)
        y.set_index(y.columns[0], inplace=True)
        y = y.loc[x.index]
        self.cancer_type = y["cancer_type"].values
        self.primary_site = y["primary_site"].values
        self.sample_type = y["sample_type"].values
        self.stage = y["stage"].values
        self.tumor_event = y["tumor_event"].values

    def __getitem__(
        self, idx: int
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """
        cancer_type: 0
        primary_site: 1
        sample_type: 2
        stage: 3
        tumor_event: 4
        """
        return (
            th.as_tensor(self.data[idx], dtype=th.float32),
            self.cancer_type[idx],
            self.primary_site[idx],
            self.sample_type[idx],
            self.stage[idx],
            self.tumor_event[idx],
        )

    def __len__(self) -> int:
        return len(self.data)
