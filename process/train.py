import os.path as osp
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger
import wandb

from module.data_loader import O2VDataModule
from module.model import O2VModule
from module.util import check_file
from .task import Task
from .preprocess import Preprocess


class Train(Task):
    def __init__(self, processor: Preprocess, **kwargs):
        super(Train, self).__init__(**kwargs)
        # Update parameters
        self.project_params = self.update_configuration(
            self.params["project_params"],
            "monitor",
            f"val_p_loss({self.params['model_params']['p_loss']})",
        )
        label_df = pd.read_feather(
            osp.join(processor.save_file_path, self.config["label_data_name"])
        )
        output_dim = (
            [
                len(np.unique(label_df[label].values))
                for label in self.config["label_type"]
            ]
            if self.config["label_type"] is not None
            else []
        )
        self.model_params = self.update_configuration(
            self.params["model_params"],
            [
                "data_type",
                "origin_dim",
                "label_type",
                "output_dim",
            ],
            [
                self.config["data_type"],
                processor.origin_dim,
                self.config["label_type"]
                if self.config["label_type"] is not None
                else [],
                output_dim,
            ],
        )
        self.data_params = self.update_configuration(
            self.params["data_params"],
            [
                "processed_data_path",
                "processed_data_name",
                "processed_label_name",
                "processed_label_list",
            ],
            [
                processor.save_file_path,
                self.config["data_type"],
                self.config["label_data_name"],
                self.config["label_type"],
            ],
        )

    def load_model_and_data(
        self, model_params: Dict[str, Any], data_params: Dict[str, Any]
    ) -> Tuple[pl.LightningModule, pl.LightningDataModule]:
        model = O2VModule(**model_params)
        if self.params["fine_tuning"] is not None:
            print("Fine tuning has been set!")
            model = model.load_from_checkpoint(
                osp.join(
                    self.params["save_path"],
                    self.params["fine_tuning"],
                    self.config["experiment"] + ".ckpt",
                )
            )
        data = O2VDataModule(**data_params)

        return model, data

    def load_trainer(self) -> pl.Trainer:
        # Set wandb
        pl.seed_everything(self.project_params["seed"])
        trainer = pl.Trainer(
            logger=WandbLogger(
                entity=self.project_params["entity"],
                project=self.project_params["project"],
                name=self.config["experiment"],
                log_model=True,
            ),
            gpus=self.project_params["gpus"],
            strategy=self.project_params["strategy"],
            max_epochs=self.project_params["max_epochs"],
            callbacks=[
                EarlyStopping(
                    patience=self.project_params["patience"],
                    monitor=self.project_params["monitor"],
                ),
                LearningRateMonitor(
                    logging_interval=self.project_params["logging_interval"]
                ),
                ModelCheckpoint(
                    dirpath=self.save_file_path,
                    filename=self.config["experiment"],
                    monitor=self.project_params["monitor"],
                    save_top_k=1,
                    mode="min",
                ),
            ],
            gradient_clip_val=self.project_params["clip_norm"],
        )

        return trainer

    def run_task(self) -> None:
        if not check_file(
            osp.join(self.save_file_path, self.config["experiment"] + ".ckpt")
        ):
            net, data = self.load_model_and_data(self.model_params, self.data_params)
            trainer = self.load_trainer()
            trainer.fit(net, data)
            wandb.finish()
