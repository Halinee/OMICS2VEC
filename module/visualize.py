import os.path as osp
from typing import Any, Dict, List

from bokeh.plotting import ColumnDataSource
from bokeh.models import HoverTool
from bokeh.plotting import figure, output_file, save
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .util import set_desc, check_file


def generate_tsne(
    data_type: str,
    tsne: np.ndarray,
    label: Dict[str, List[str]],
    color: Dict[str, List[str]],
    legend: Dict[str, List[str]],
    params: Dict[str, Any],
    save_path: str,
) -> None:
    for color_key in color.keys():
        save_file_path = osp.join(save_path, data_type + "_" + color_key + ".html")
        if check_file(save_file_path):
            continue
        source = ColumnDataSource(
            data=dict(
                x=tsne[:, 0],
                y=tsne[:, 1],
                desc=set_desc(label),
                legend=legend[color_key],
                color=color[color_key],
            )
        )

        hover = HoverTool(
            tooltips="""
                    <div>
                        <div>
                            <span style="font-size: 17px; font-weight: bold;">@desc</span>
                        </div>
                    </div>
                    """
        )

        interactive_map = figure(
            plot_width=params["plot_width"],
            plot_height=params["plot_height"],
            tools=["reset,box_zoom,wheel_zoom,zoom_in,zoom_out,pan", hover],
            title=data_type + " " + color_key + " " + params["title"],
        )
        interactive_map.circle(
            "x",
            "y",
            source=source,
            color="color",
            size=params["point_size"],
            fill_alpha=params["fill_alpha"],
            legend_field="legend",
        )
        interactive_map.legend.location = "bottom_right"
        interactive_map.legend.click_policy = "hide"
        output_file(save_file_path)
        save(interactive_map)


def generate_feature_importances(
    data_type: str,
    model_dict: Dict[str, RandomForestClassifier],
    features: List[Any],
    save_path: str,
) -> None:
    for label, model in model_dict.items():
        save_file_path = osp.join(save_path, data_type + "_" + label + ".png")
        if check_file(save_file_path):
            continue
        importance = model[2].feature_importances_
        importances = pd.DataFrame()
        importances["Embed Columns"] = features
        importances["Importances"] = importance
        importances.sort_values("Importances", ascending=False, inplace=True)
        importances.reset_index(drop=True, inplace=True)

        plt.figure(figsize=(20, 10))

        sns.barplot(x="Importances", y="Embed Columns", data=importances.iloc[:20])
        plt.title(
            data_type.upper() + " " + label.upper() + " Feature Importances",
            fontsize=36,
        )
        plt.savefig(save_file_path)
        plt.close()


def generate_confusion_matrix(
    data_type: str,
    label_dict: Dict[str, List[int]],
    encoder_dict: Dict[str, LabelEncoder],
    model_dict: Dict[str, RandomForestClassifier],
    score_dict: Dict[str, float],
    embed: pd.DataFrame,
    save_path: str,
    test_size: float,
    random_state: int,
) -> None:
    for label, model in model_dict.items():
        save_file_path = osp.join(save_path, data_type + "_" + label + ".png")
        if check_file(save_file_path):
            continue
        x_train, x_test, y_train, y_test = train_test_split(
            embed, label_dict[label], test_size=test_size, random_state=random_state
        )
        y_pred = model.predict(x_test)
        cm = confusion_matrix(y_test, y_pred)
        percent_of_cm = [i / sum(i) for i in cm]
        plt.figure(figsize=(50, 50))
        ax = sns.heatmap(
            percent_of_cm,
            xticklabels=encoder_dict[label].classes_,
            yticklabels=encoder_dict[label].classes_,
            annot=True,
            fmt=".3f",
            linewidth=0.5,
            annot_kws={"size": 20},
            cbar_kws={"shrink": 0.8},
            square=True,
        )
        ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize=20)
        ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize=20)
        plt.title(
            data_type.upper()
            + " "
            + label.upper()
            + " F1 score(weighted) : "
            + str(score_dict[label]),
            fontsize=36,
        )
        plt.savefig(save_file_path)
        plt.close()
