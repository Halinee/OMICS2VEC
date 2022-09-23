import os.path as osp

import pandas as pd
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)

from .task import Task


class Preprocess(Task):
    def __init__(self, processor=None, **kwargs):
        super(Preprocess, self).__init__(**kwargs)
        # Raw data file
        self.scale_type = self.params["scale_type"]
        self.gene_fpkm_dim = 0
        self.isoform_fpkm_dim = 0
        self.mutation_dim = 0

    def load_raw_data(self, data_type: str) -> pd.DataFrame:
        print(f"Loading raw {data_type} data...")
        raw_omics_data = pd.read_feather(
            osp.join(self.config["raw_data_path"], data_type + ".ftr")
        )
        raw_omics_data.set_index(raw_omics_data.columns[0], inplace=True)

        return raw_omics_data

    def scaling(self, omics: pd.DataFrame) -> pd.DataFrame:
        if self.scale_type is not None:
            if self.scale_type == "zscore":
                scaler = StandardScaler()
            elif self.scale_type == "zeroone":
                scaler = MinMaxScaler()
            elif self.scale_type == "robust":
                scaler = RobustScaler()
            else:
                scaler = None
            scaled_omics = scaler.fit_transform(omics)
            scaled_omics = pd.DataFrame(
                scaled_omics,
                columns=omics.columns,
                index=omics.index,
            )
        else:
            scaled_omics = omics

        return scaled_omics

    def generate_data(self, raw_omics_data: pd.DataFrame, data_type: str) -> None:
        # Scaling data
        scaled_omics_data = self.scaling(raw_omics_data)
        match data_type:
            case "gene_fpkm":
                self.gene_fpkm_dim = len(scaled_omics_data.columns)
            case "isoform_fpkm":
                self.isoform_fpkm_dim = len(scaled_omics_data.columns)
            case "mutation":
                self.mutation_dim = len(scaled_omics_data.columns)
        # Save data
        print(f"Saving processed {data_type} data...\n")
        scaled_omics_data.reset_index(inplace=True)
        scaled_omics_data.to_feather(osp.join(self.save_file_path, data_type + ".ftr"))

    def generate_label(self) -> None:
        clinical = pd.read_feather(
            self.config["raw_data_path"] + self.config["clinical_data_name"]
        )
        clinical.set_index(clinical.columns[0], inplace=True)
        label = pd.DataFrame(index=clinical.index)
        # Fill data with label encoding
        label["cancer_type"] = LabelEncoder().fit_transform(
            clinical["cancer_type"].values
        )
        label["primary_site"] = LabelEncoder().fit_transform(
            clinical["primary_site"].values
        )
        label["sample_type"] = LabelEncoder().fit_transform(
            clinical["sample_type"].values
        )
        label["stage"] = LabelEncoder().fit_transform(clinical["stage"].values)
        label["tumor_event"] = LabelEncoder().fit_transform(
            clinical["tumor_event"].values
        )
        # Save data
        print(f"Saving processed label data...\n")
        label.reset_index(inplace=True)
        label.to_feather(osp.join(self.save_file_path, self.config["label_data_name"]))

    def run_task(self) -> None:
        processed_label_file = osp.join(
            self.save_file_path, self.config["label_data_name"]
        )
        if not osp.exists(processed_label_file):
            self.generate_label()
        for t in self.config["data_type"]:
            processed_file = osp.join(self.save_file_path, t + ".ftr")
            if osp.exists(processed_file):
                omics = pd.read_feather(processed_file)
                match t:
                    case "gene_fpkm":
                        self.gene_fpkm_dim = len(omics.columns) - 1
                    case "isoform_fpkm":
                        self.isoform_fpkm_dim = len(omics.columns) - 1
                    case "mutation":
                        self.mutation_dim = len(omics.columns) - 1
            else:
                raw_omics_data = self.load_raw_data(t)
                self.generate_data(raw_omics_data, t)
