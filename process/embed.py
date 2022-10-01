import os.path as osp
import pickle as pkl
from typing import Dict, Union

import pandas as pd
import torch as th

from module.model import O2VModule
from .task import Task
from .train import Train


class Embed(Task):
    def __init__(self, processor: Train, **kwargs):
        super(Embed, self).__init__(**kwargs)
        self.project_params = processor.project_params
        self.model_params = processor.model_params
        self.data_params = processor.data_params
        self.model_checkpoint_path = processor.save_file_path
        with open(
            osp.join(
                self.data_params["processed_data_path"], self.config["index_data_name"]
            ),
            "rb",
        ) as f:
            self.data_idx = pkl.load(f)

    def load_model(self) -> O2VModule:
        model = O2VModule(**self.model_params)
        model = model.load_from_checkpoint(
            osp.join(self.model_checkpoint_path, self.config["experiment"] + ".ckpt")
        )

        return model

    def load_data_dict(self) -> Dict[str, pd.DataFrame]:
        print("Loading input data...")
        data = {
            i: self.load_feather(self.data_params["processed_data_path"], i + ".ftr")
            for i in self.config["data_type"]
        }

        return data

    def generate_embed(self, index: pd.Index, embed: Union[float, th.Tensor]) -> None:
        print("Saving data...")
        result = pd.DataFrame(
            embed.detach().numpy(),
            index=index,
            columns=list(map(str, range(embed.shape[1]))),
        )
        result = result.reset_index()
        result.to_feather(osp.join(self.save_file_path, "embed.ftr"))

    def run_task(self) -> None:
        pre_trained_model = self.load_model()
        processed_data = self.load_data_dict()
        tensor_data = {
            data: th.as_tensor(processed_data[data].to_numpy(), dtype=th.float32)
            for data in self.config["data_type"]
        }
        # Generate embed
        embed_data = pre_trained_model.g_model.encode_from_omics(tensor_data)
        self.generate_embed(index=self.data_idx, embed=embed_data)
