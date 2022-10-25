import itertools
import os.path as osp
from typing import Dict, List, Set, Tuple

import pandas as pd
from pylab import *
from sklearn.preprocessing import LabelEncoder

from module.util import scaling
from .task import Task


class Preprocess(Task):
    def __init__(self, processor=None, **kwargs):
        super(Preprocess, self).__init__(**kwargs)
        # Raw data file
        self.data_dict = {}
        self.scale_type = self.params["scale_type"]
        self.clinical = self.load_feather(
            self.config["raw_data_path"], self.config["clinical_data_name"]
        )
        self.origin_dim = {}

    def generate_scaled_data(self) -> Dict[str, pd.DataFrame]:
        return {k: scaling(self.scale_type, v) for k, v in self.data_dict.items()}

    def generate_virtual_data(
        self, combine_data_idx: Set[str]
    ) -> Dict[str, pd.DataFrame]:
        return {
            k: pd.DataFrame(
                np.zeros((len(combine_data_idx - set(v.index)), len(v.columns))),
                index=list(combine_data_idx - set(v.index)),
                columns=v.columns,
            )
            for k, v in self.data_dict.items()
        }

    def generate_label_and_decode(
        self, data_idx: List[str]
    ) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
        clinical = self.clinical.loc[data_idx]
        label_df = pd.DataFrame(index=clinical.index)
        decode = {}
        # Fill data with label encoding
        encoder = LabelEncoder()
        for label in self.config["label_type"]:
            label_df[label] = encoder.fit_transform(clinical[label].values)
            decode[label] = encoder.classes_

        return label_df, decode

    def generate_virtual_info(
        self, data_idx: List[str], masking_data: Dict[str, pd.DataFrame]
    ) -> Tuple[Dict[str, Dict[str, str]], pd.DataFrame]:
        # Add data authenticity information to existing color data for visualization
        color = self.load_pickle(
            self.config["raw_data_path"], self.config["color_data_name"]
        )
        masking_case = [
            "".join(case)
            for case in itertools.product(
                ["0", "1"], repeat=len(self.config["data_type"])
            )
        ]
        color_map = cm.get_cmap("jet", len(masking_case))
        masking_color = [
            matplotlib.colors.rgb2hex(color_map(color_index))
            for color_index in range(color_map.N)
        ]
        color["authenticity"] = {
            case: color for case, color in zip(masking_case, masking_color)
        }

        # Add data authenticity information to existing clinical data for set label
        clinical = self.clinical.loc[data_idx]
        masking_df = pd.concat(masking_data.values(), axis=1)
        masking_value = masking_df.loc[data_idx].values
        masking_label = []
        for value in masking_value:
            label = ""
            for v in value:
                label += "1" if v == 1 else "0"
            masking_label.append(label)
        clinical["authenticity"] = masking_label

        return color, clinical

    def generate_concat_data(self) -> None:
        # Generate reality scaled data
        reality_data = self.generate_scaled_data()
        combine_data_idx = set(
            np.concatenate([list(data.index) for data in self.data_dict.values()])
        ) & set(self.clinical.index)
        # Generate virtual data
        virtual_data = self.generate_virtual_data(combine_data_idx)
        # Generate combines reality data and virtual data
        masking_label = {
            data: [1] * len(reality_data[data]) + [0] * len(virtual_data[data])
            for data in self.config["data_type"]
        }
        combine_data = {
            data: pd.concat([reality_data[data], virtual_data[data]])
            for data in self.config["data_type"]
        }
        masking_data = {
            data: pd.DataFrame(
                masking_label[data],
                index=list(combine_data[data].index),
                columns=["masking_" + data],
            )
            for data in self.config["data_type"]
        }
        if self.params["virtual_condition"]:
            combine_data = {
                data: pd.concat([combine_data[data], masking_data[data]], axis=1)
                for data in self.config["data_type"]
            }
        combine_data_idx = list(combine_data_idx)
        combine_data = {k: v.loc[combine_data_idx] for k, v in combine_data.items()}
        masking_data = {k: v.loc[combine_data_idx] for k, v in masking_data.items()}
        label, decode = self.generate_label_and_decode(combine_data_idx)
        label = pd.concat([label, pd.concat(masking_data.values(), axis=1)], axis=1)
        color, clinical = self.generate_virtual_info(
            data_idx=combine_data_idx, masking_data=masking_data
        )

        print("Saving processed index data...\n")
        self.save_pickle(
            combine_data_idx, self.save_file_path, self.config["index_data_name"]
        )
        print("Saving processed color data...\n")
        self.save_pickle(color, self.save_file_path, self.config["color_data_name"])
        print("Saving processed clinical data...\n")
        self.save_feather(
            clinical, self.save_file_path, self.config["clinical_data_name"]
        )
        print("Saving processed label data...\n")
        self.save_feather(label, self.save_file_path, self.config["label_data_name"])
        print(f"Saving processed decode data...\n")
        self.save_pickle(decode, self.save_file_path, self.config["decode_data_name"])
        for data in self.config["data_type"]:
            print(f"Saving processed {data} data...\n")
            self.save_feather(combine_data[data], self.save_file_path, data + ".ftr")

    def check_task(self) -> bool:
        check_file = True
        # Check processed omics data
        print("Check the existence of the omics data file...")
        for data in self.config["data_type"]:
            if not osp.exists(osp.join(self.save_file_path, data + ".ftr")):
                print(f"Processed {data} data file does not exist!\n")
                check_file = False
        # Check meta data
        meta_file = ["index", "color", "clinical", "label", "decode"]
        for file in meta_file:
            print(f"Check the existence of the {file} data file...")
            if not osp.exists(
                osp.join(self.save_file_path, self.config[f"{file}_data_name"])
            ):
                print(f"Processed {file} data file does not exist!\n")
                check_file = False

        return check_file

    def run_task(self) -> None:
        self.generate_concat_data()

    def processing(self) -> None:
        start = self.start_task()
        # Set input dimension each data
        self.data_dict = {
            data: self.load_feather(self.config["raw_data_path"], data + ".ftr")
            for data in self.config["data_type"]
        }
        self.origin_dim = {k: len(v.columns) for k, v in self.data_dict.items()}
        # If a virtual condition is applied, the number of dimensions is increased by 1
        if self.params["virtual_condition"]:
            self.origin_dim = {k: v + 1 for k, v in self.origin_dim.items()}
        if not self.check_task():
            print("Start generating data that doesn't exist...\n")
            self.run_task()
        self.end_task(start)
