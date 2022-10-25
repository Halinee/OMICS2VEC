import os.path as osp
from typing import Any, Dict, List, Union

import pandas as pd
from bokeh.plotting import ColumnDataSource
from bokeh.models import HoverTool
from bokeh.plotting import figure, output_file, save
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

from .util import set_desc, check_file


def generate_tsne(
    tsne: np.ndarray,
    label: Dict[str, List[str]],
    color: Dict[str, List[str]],
    legend: Dict[str, List[str]],
    params: Dict[str, Any],
    data_type: str,
    save_path: str,
) -> None:
    for color_key in color.keys():
        save_file_path = osp.join(save_path, f"{data_type}_{color_key}.html")
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
            title=f"{data_type} {color_key} t-SNE result",
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


def generate_confusion_matrix(
    result_dict: Dict[str, Dict[str, Dict[str, List[Union[int, float]]]]],
    score_dict: Dict[str, Dict[str, float]],
    decode_dict: Dict[str, List[str]],
    data_type: str,
    save_path: str,
) -> None:
    for label, k_fold in result_dict.items():
        save_file_path = osp.join(
            save_path, f"{data_type}_{label}_confusion_matrix.png"
        )
        if check_file(save_file_path):
            continue
        cm = None
        for values in k_fold.values():
            if cm is None:
                cm = confusion_matrix(values["true"], values["pred"])
            else:
                cm += confusion_matrix(values["true"], values["pred"])
        percent_of_cm = [i / sum(i) for i in cm]
        plt.figure(figsize=(50, 50))
        ax = sns.heatmap(
            percent_of_cm,
            xticklabels=decode_dict[label],
            yticklabels=decode_dict[label],
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
            f"{data_type} {label} F1 score(weighted): {score_dict[label]}",
            fontsize=36,
        )
        plt.savefig(save_file_path)
        plt.close()


def generate_performance_data_frame(
    k_fold: int, score_dict: Dict[str, Dict[str, float]], data_type: str, save_path: str
):
    save_file_path = osp.join(save_path, f"{data_type}_K-fold_performance.csv")
    index = []
    data = []
    for label, k_fold_data in score_dict.items():
        index.append(label)
        data.append([k_fold_values for k_fold_values in k_fold_data.values()])

    performance_df = pd.DataFrame(
        data, index=index, columns=[f"{idx + 1}_fold" for idx in range(k_fold)]
    )
    performance_df.to_csv(osp.join(save_file_path))
