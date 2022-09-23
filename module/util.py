import os
import os.path as osp
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.preprocessing import LabelEncoder


def check_directory(default_path: str, check_path: str) -> str:
    new_path = osp.join(default_path, check_path)
    if not osp.exists(new_path):
        os.mkdir(new_path)

    return new_path


def check_file(file_path: str) -> bool:
    if osp.exists(file_path):
        return True
    else:
        return False


def set_label(clinical: pd.DataFrame) -> Dict[str, List[str]]:
    print("Setting label...")
    label = {}
    for key in clinical.columns:
        label[key] = clinical[key].values.tolist()

    return label


def set_color_and_legend(
    color: Dict[str, Dict[str, str]], label: Dict[str, List[str]]
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    print("Setting color...")
    color_dict = {}
    legend_dict = {}
    for info in color.keys():
        legend_dict[info] = label[info]
        color_dict[info] = [color[info][x] for x in label[info]]

    return color_dict, legend_dict


def set_desc(label: Dict[str, List[str]]) -> List[str]:
    desc_list = []
    label_len = len(label[list(label.keys())[0]])

    for i in range(label_len):
        desc = ""
        for key in label.keys():
            desc += key + " : " + str(label[key][i]) + "<br>"
        desc_list.append(desc)

    return desc_list


def label_encoding(
    analysis_list: List[str], data: pd.DataFrame
) -> Tuple[Dict[str, List[int]], Dict[str, LabelEncoder]]:
    label_dict = {}
    encoder_dict = {}
    for analysis in analysis_list:
        label_encoder = LabelEncoder()
        label_encoder.fit(data[analysis])
        label_dict[analysis] = label_encoder.transform(data[analysis])
        encoder_dict[analysis] = label_encoder

    return label_dict, encoder_dict
