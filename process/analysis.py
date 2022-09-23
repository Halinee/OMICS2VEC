import os
import os.path as osp
import pickle as pkl
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from tqdm import tqdm

from module.util import (
    check_directory,
    set_label,
    set_color_and_legend,
    label_encoding,
)
from module.visualize import (
    generate_tsne,
    generate_feature_importances,
    generate_confusion_matrix,
)
from .task import Task
from .embed import Embed


class Analysis(Task):
    def __init__(self, processor: Embed, **kwargs):
        super(Analysis, self).__init__(**kwargs)
        self.embedding_params = processor.params
        with open(
            osp.join(self.config["raw_data_path"], self.params["color_data_name"]), "rb"
        ) as f:
            self.color = pkl.load(f)
        with open(
            osp.join(self.config["raw_data_path"], self.params["sample_list_name"]),
            "rb",
        ) as f:
            self.sample_list = pkl.load(f)
        self.clinical = pd.read_feather(
            osp.join(self.config["raw_data_path"], self.config["clinical_data_name"])
        )
        self.clinical.set_index(self.clinical.columns[0], inplace=True)
        self.clinical = self.clinical.loc[self.sample_list]
        self.bokeh_params = self.params["bokeh_params"]
        self.tsne_params = self.params["tsne_params"]
        self.rf_params = self.params["rf_params"]

    def load_embed(self) -> Dict[str, pd.DataFrame]:
        embed_dict = {}
        for t in self.config["data_type"]:
            data = pd.read_feather(
                osp.join(
                    self.embedding_params["save_path"],
                    self.config["experiment"],
                    t + ".ftr",
                )
            )
            data.set_index(data.columns[0], inplace=True)
            embed_dict[t] = data.loc[self.sample_list]
        embed_avg = sum(embed_dict.values()) / len(embed_dict.keys())
        embed_con = pd.concat(embed_dict.values(), axis=1)
        embed_con.columns = list(range(len(embed_con.columns)))
        embed_dict["embed_avg"] = embed_avg
        embed_dict["embed_con"] = embed_con

        return embed_dict

    def load_tsne(self, embedding: pd.DataFrame) -> np.ndarray:
        print("Loading t-SNE...")
        scaler = MinMaxScaler()
        scale_data = scaler.fit_transform(np.array(embedding))
        # t-SNE
        tsne_data = TSNE(
            n_components=self.tsne_params["n_components"],
            n_iter=self.tsne_params["n_iter"],
            random_state=self.tsne_params["random_state"],
        ).fit_transform(scale_data)

        return tsne_data

    def load_prediction(
        self, embed: pd.DataFrame
    ) -> Tuple[
        Dict[str, List[int]],
        Dict[str, LabelEncoder],
        Dict[str, RandomForestClassifier],
        Dict[str, float],
    ]:
        label_dict, encoder_dict = label_encoding(
            analysis_list=self.params["analysis_list"], data=self.clinical
        )
        model_dict = {}
        score_dict = {}
        for label in label_dict.keys():
            x_train, x_test, y_train, y_test = train_test_split(
                embed,
                label_dict[label],
                test_size=self.rf_params["test_size"],
                random_state=self.rf_params["random_state"],
            )
            model = RandomForestClassifier(
                n_estimators=self.rf_params["n_estimators"], class_weight="balanced"
            )
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            score = metrics.f1_score(y_test, y_pred, average="weighted")
            print(
                label.upper() + " prediction(weighted F1-score) result :",
                score,
            )
            model_dict[label] = model
            score_dict[label] = score

        return label_dict, encoder_dict, model_dict, score_dict

    def run_task(self) -> None:
        # Load label information
        label = set_label(self.clinical)
        color, legend = set_color_and_legend(self.color, label)
        data_dict = self.load_embed()
        # Generate t-SNE visualize result
        print("Generate t-SNE result...")
        tsne_file_path = check_directory(self.save_file_path, "t_SNE")
        if len(os.listdir(tsne_file_path)) == 0:
            tsne_dict = {}
            for k, v in tqdm(data_dict.items()):
                tsne_dict[k] = self.load_tsne(v)
            for k, v in tsne_dict.items():
                generate_tsne(
                    data_type=k,
                    tsne=v,
                    label=label,
                    color=color,
                    legend=legend,
                    params=self.bokeh_params,
                    save_path=tsne_file_path,
                )
        print("Generate feature importance and accuracy by cancer type result...")
        feature_importance_file_path = check_directory(
            self.save_file_path, "feature_importance"
        )
        accuracy_file_path = check_directory(self.save_file_path, "accuracy")
        if (
            len(os.listdir(feature_importance_file_path)) == 0
            or len(os.listdir(accuracy_file_path)) == 0
        ):
            for k, v in data_dict.items():
                label_dict, encoder_dict, model_dict, score_dict = self.load_prediction(
                    embed=v
                )
                generate_feature_importances(
                    data_type=k,
                    model_dict=model_dict,
                    features=list(v.columns),
                    save_path=feature_importance_file_path,
                )
                generate_confusion_matrix(
                    data_type=k,
                    label_dict=label_dict,
                    encoder_dict=encoder_dict,
                    model_dict=model_dict,
                    score_dict=score_dict,
                    embed=v,
                    save_path=accuracy_file_path,
                    test_size=self.rf_params["test_size"],
                    random_state=self.rf_params["random_state"],
                )
