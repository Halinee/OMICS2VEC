import os.path as osp
import pickle as pkl
import random
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch as th
from tqdm import tqdm

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

    def generate_hide_data(
        self,
        data_dict: Dict[str, th.Tensor],
    ) -> Tuple[Dict[str, th.Tensor], List[int]]:
        hidden_data_dict = {}
        hide_list = random.sample(
            range(len(data_dict[self.params["hidden_data_type"]])),
            int(
                len(data_dict[self.params["hidden_data_type"]])
                * self.params["hidden_rate"]
            ),
        )
        print(f"Number of data to be hidden: {len(hide_list)}")
        virtual_data = th.zeros(len(data_dict[self.params["hidden_data_type"]][0]))
        # If a sample has virtual data, the sample is not hidden.
        check_data_type = list(
            set(self.config["data_type"]) - set(self.params["hidden_data_type"])
        )
        hidden_idx = []
        hidden_data = data_dict[self.params["hidden_data_type"]].clone()
        for idx in tqdm(hide_list):
            check_idx = True
            for data in check_data_type:
                if data_dict[data][idx].sum() == 0:
                    check_idx = False
                    break
            if check_idx:
                hidden_idx.append(idx)
                hidden_data[idx] = virtual_data

        print(f"Number of hidden data: {len(hidden_idx)}")
        hidden_data_dict[self.params["hidden_data_type"]] = hidden_data
        for data in self.config["data_type"]:
            if not data == self.params["hidden_data_type"]:
                hidden_data_dict[data] = data_dict[data]

        return hidden_data_dict, hidden_idx

    def generate_reconstructed_data(
        self, data_dict: Dict[str, pd.DataFrame], embed: th.Tensor, model: O2VModule
    ) -> None:
        reconstructed_data = data_dict.copy()
        for data in self.config["data_type"]:
            print(f"\nReconstructing {data} data...")
            for i, idx in tqdm(enumerate(self.data_idx)):
                if reconstructed_data[data].loc[idx].sum() == 0:
                    reconstructed_data[data].loc[idx] = (
                        model.g_model.decoder[data](embed[i]).detach().numpy()
                    )
            self.save_feather(
                reconstructed_data[data],
                self.data_params["processed_data_path"],
                data + "_recon.ftr",
            )

    def generate_embed(
        self, index: List[str], embed: Union[float, th.Tensor], name: str = "embed"
    ) -> None:
        print(f"Saving {name} data...")
        result = pd.DataFrame(
            embed.detach().numpy(),
            index=index,
            columns=list(map(str, range(embed.shape[1]))),
        )
        result.reset_index(inplace=True)
        result.to_feather(osp.join(self.save_file_path, name + ".ftr"))

    def check_task(self) -> bool:
        check_file = True
        # Check embed data
        print("Check the existence of the embed data file...")
        if not osp.exists(osp.join(self.save_file_path, "embed.ftr")):
            print(f"Embed data file does not exist!\n")
            check_file = False
        # Check hidden embed data
        if not osp.exists(osp.join(self.save_file_path, "hidden_embed.ftr")):
            print(f"Hidden embed data file does not exist!\n")
            check_file = False
        for data in self.config["data_type"]:
            if not osp.exists(
                osp.join(self.data_params["processed_data_path"], data + "_recon.ftr")
            ):
                print(f"Reconstructed processed data file does not exist!\n")
                check_file = False

        return check_file

    def run_task(self) -> None:
        pre_trained_model = self.load_model()
        processed_data = self.load_data_dict()
        exist_data = {
            data: th.as_tensor(processed_data[data].to_numpy(), dtype=th.float32)
            for data in self.config["data_type"]
        }
        """
        hidden_data, hidden_idx = self.generate_hide_data(exist_data)
        hidden_idx = np.array(self.data_idx)[hidden_idx].tolist()
        for data in self.config["data_type"]:
            print(f"exist {data} sum: {exist_data[data].sum()}")
            print(f"hidden {data} sum: {hidden_data[data].sum()}\n")
        self.save_pickle(
            hidden_idx, self.save_file_path, self.params["hidden_index_data_name"]
        )
        """
        # Generate embed
        embed_data = pre_trained_model.g_model.encode_from_omics(exist_data)
        #        hidden_embed_data = pre_trained_model.g_model.encode_from_omics(hidden_data)
        #        print(f"exist embed sum: {embed_data.sum()}")
        #        print(f"hidden embed sum: {hidden_embed_data.sum()}\n")
        self.generate_embed(index=self.data_idx, embed=embed_data)
        #        self.generate_embed(
        #            index=self.data_idx, embed=hidden_embed_data, name="hidden_embed"
        #        )
        # Generate reconstructed data
        self.generate_reconstructed_data(
            data_dict=processed_data, embed=embed_data, model=pre_trained_model
        )
