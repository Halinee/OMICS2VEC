import os.path as osp
from typing import Tuple, Union

import numpy as np
import pandas as pd
import torch as th
from pytorch_lightning import LightningModule

from module.model import O2VModule
from module.util import check_file
from .task import Task
from .train import Train


class Embed(Task):
    def __init__(self, processor: Train, **kwargs):
        super(Embed, self).__init__(**kwargs)
        self.project_params = processor.project_params
        self.model_params = processor.model_params
        self.data_params = processor.data_params
        self.model_checkpoint_path = processor.save_file_path

    def load_ckpt_and_data(
        self, data_type: str
    ) -> Tuple[LightningModule, pd.DataFrame]:
        print("Loading pre-trained model...")
        self.model_params["origin_dim"] = self.model_params[data_type + "_dim"]
        model = O2VModule(**self.model_params)
        model = model.load_from_checkpoint(
            osp.join(self.model_checkpoint_path, data_type + ".ckpt")
        )
        print("Loading input data...")
        data = pd.read_feather(
            osp.join(self.config["raw_data_path"], data_type + ".ftr")
        )
        data = data.set_index(data.columns[0])

        return model, data

    def generate_embed(
        self, index: pd.Index, embed: Union[float, th.Tensor], data_type: str
    ) -> None:
        print("Saving data...")
        result = pd.DataFrame(
            embed.detach().numpy(),
            index=index,
            columns=list(map(str, range(embed.shape[1]))),
        )
        result = result.reset_index()
        result.to_feather(osp.join(self.save_file_path, data_type + ".ftr"))

    def run_task(self) -> None:
        for t in self.config["data_type"]:
            if not check_file(osp.join(self.save_file_path, t + ".ftr")):
                model, data = self.load_ckpt_and_data(t)
                embed = model.g_model.encode_from_omics(np.array(data))
                self.generate_embed(data.index, embed, t)
