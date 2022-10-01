import os.path as osp
import pickle as pkl
from typing import Dict, List, Tuple, Union

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler

from module.util import (
    set_label,
    set_color_and_legend,
)
from module.visualize import (
    generate_confusion_matrix,
    generate_performance_data_frame,
    generate_tsne,
)
from .task import Task
from .embed import Embed


class Analysis(Task):
    def __init__(self, processor: Embed, **kwargs):
        super(Analysis, self).__init__(**kwargs)
        self.embedding_params = processor.params
        self.data_params = processor.data_params
        with open(
            osp.join(
                self.data_params["processed_data_path"], self.config["color_data_name"]
            ),
            "rb",
        ) as f:
            color = pkl.load(f)
        # Extract the label included in label_type
        self.color = {k: v for k, v in color.items() if k in self.config["label_type"]}
        # Label extraction for virtual sample identification visualization
        self.color["authenticity"] = color["authenticity"]
        with open(
            osp.join(
                self.data_params["processed_data_path"], self.config["index_data_name"]
            ),
            "rb",
        ) as f:
            self.data_idx = pkl.load(f)
        with open(
            osp.join(
                self.data_params["processed_data_path"],
                self.config["decode_data_name"],
            ),
            "rb",
        ) as f:
            self.decode_dict = pkl.load(f)
        self.clinical = self.load_feather(
            self.data_params["processed_data_path"], self.config["clinical_data_name"]
        ).loc[self.data_idx]
        self.label = self.load_feather(
            self.data_params["processed_data_path"], self.config["label_data_name"]
        ).loc[self.data_idx]
        self.bokeh_params = self.params["bokeh_params"]
        self.tsne_params = self.params["tsne_params"]
        self.random_forest_params = self.params["random_forest_params"]

    def load_embed(self) -> np.ndarray:
        embed = self.load_feather(
            osp.join(self.embedding_params["save_path"], self.config["experiment"]),
            "embed.ftr",
        ).to_numpy()

        return embed

    def load_tsne(self, embedding: np.ndarray) -> np.ndarray:
        print("Loading t-SNE...")
        scaler = MinMaxScaler()
        scale_data = scaler.fit_transform(embedding)
        # t-SNE
        tsne_data = TSNE(
            n_components=self.tsne_params["n_components"],
            n_iter=self.tsne_params["n_iter"],
            random_state=self.params["random_state"],
        ).fit_transform(scale_data)

        return tsne_data

    def load_prediction(
        self, embed: np.ndarray
    ) -> Tuple[
        Dict[str, Dict[str, Dict[str, List[Union[int, float]]]]],
        Dict[str, Dict[str, float]],
    ]:
        result_dict = {}
        score_dict = {}
        for label in self.config["label_type"]:
            result_dict[label] = {}
            score_dict[label] = {}
            y = self.label[label].values
            k_fold = StratifiedKFold(
                n_splits=self.random_forest_params["k_fold"],
                random_state=self.params["random_state"],
                shuffle=True,
            )
            for k_fold_idx, (train_idx, test_idx) in enumerate(k_fold.split(embed, y)):
                x_train, x_test = embed[train_idx], embed[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                model = RandomForestClassifier(
                    n_estimators=self.random_forest_params["n_estimators"],
                    class_weight=self.random_forest_params["class_weight"],
                )
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
                result = {"true": y_test, "pred": y_pred}
                result_dict[label][f"{k_fold_idx + 1}_fold"] = result
                score = metrics.f1_score(y_test, y_pred, average="weighted")
                print(
                    f"{label.upper()} {k_fold_idx + 1}_fold prediction(weighted F1-score) result : {score}"
                )
                score_dict[label][f"{k_fold_idx + 1}_fold"] = score

        return result_dict, score_dict

    def check_task(self) -> bool:
        return False

    def run_task(self) -> None:
        # Load label information
        label = set_label(self.clinical)
        color, legend = set_color_and_legend(self.color, label)
        embed = self.load_embed()
        # Generate t-SNE visualize result
        print("Generate t-SNE visualization result...")
        tsne = self.load_tsne(embed)
        generate_tsne(
            tsne=tsne,
            label=label,
            color=color,
            legend=legend,
            params=self.bokeh_params,
            save_path=self.save_file_path,
        )
        # Generate confusion matrix visualize result
        result_dict, score_dict = self.load_prediction(embed)
        print("Generate performance evaluation results...")
        generate_performance_data_frame(
            k_fold=self.random_forest_params["k_fold"],
            score_dict=score_dict,
            save_path=self.save_file_path,
        )
        generate_confusion_matrix(
            result_dict=result_dict,
            score_dict=score_dict,
            decode_dict=self.decode_dict,
            save_path=self.save_file_path,
        )
