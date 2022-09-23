from typing import Any, Dict, List, Tuple, Union

import pytorch_lightning as pl
import torch as th
import torch.distributions as dist

from .layers import VAE, MLP
from .function import LOSS_FN_FACTORY, OPTIMIZER_FACTORY


class O2VModule(pl.LightningModule):
    def __init__(
        self,
        label_type: List[str],
        predict_label: List[str],
        origin_dim: int = 100,
        g_hidden_dim: List[int] = None,
        p_hidden_dim: List[int] = None,
        latent_dim: int = 100,
        output_dim: List[int] = None,
        g_act: str = "lrelu",
        p_act: str = "lrelu",
        g_loss: str = "mae",
        p_loss: str = "focal",
        g_opt: str = "adabelief",
        p_opt: str = "adabelief",
        kl_coef: float = 0.1,
        g_lr: float = 1e-4,
        p_lr: float = 1e-4,
        g_weight_decay=1e-2,
        p_weight_decay=1e-2,
        g_eps: float = 1e-16,
        p_eps: float = 1e-16,
        **kwargs,
    ):
        super(O2VModule, self).__init__()
        self.save_hyperparameters()
        if g_hidden_dim is None:
            g_hidden_dim = [1024, 512, 256, 128]
        if p_hidden_dim is None:
            p_hidden_dim = [100, 100]
        g_output_act = "sigmoid" if g_loss == "bce" else g_act
        self.label_type = label_type
        self.predict_label = predict_label
        self.g_model = VAE(origin_dim, g_hidden_dim, latent_dim, g_act, g_output_act)
        self.p_model = {
            label: MLP(latent_dim, p_hidden_dim, output_dim[idx], p_act)
            for idx, label in enumerate(predict_label)
        }

    def forward(
        self, x: th.Tensor
    ) -> Tuple[th.Tensor, dist.Normal, dist.Normal, Union[float, th.Tensor]]:
        return self.g_model(x)

    def generation_step(
        self, x: th.Tensor
    ) -> Tuple[th.Tensor, Dict[str, Union[th.Tensor, float]]]:
        x_hat, p, q, z = self(x)
        recon_loss = LOSS_FN_FACTORY[self.hparams.g_loss](x_hat, x)
        kl_loss = th.mean(dist.kl_divergence(q, p)) * float(self.hparams.kl_coef)
        loss = kl_loss + recon_loss

        logs = {
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            f"g_loss({self.hparams.g_loss})": loss,
        }
        return loss, logs

    def prediction_step(
        self, label: str, m: MLP, x: th.Tensor, y: th.Tensor
    ) -> Tuple[th.Tensor, Dict[str, Union[th.Tensor, float]]]:
        z = self.g_model.encode_from_omics(x)
        y_hat = m(z)
        loss = LOSS_FN_FACTORY[self.hparams.p_loss](y_hat, y)

        logs = {f"{label}_p_loss({self.hparams.p_loss})": loss}
        return loss, logs

    def on_fit_start(self) -> None:
        for m in self.p_model.values():
            m.to(self.device)

    def training_step(
        self,
        batch: Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor],
        batch_idx: int,
        optimizer_idx: int,
    ) -> th.Tensor:
        x = batch[0]
        y = batch[1:]
        if optimizer_idx == 0:
            g_loss, logs = self.generation_step(x)
            p_losses = 0.0
            for k, v in self.p_model.items():
                p_loss, _ = self.prediction_step(k, v, x, y[self.label_type.index(k)])
                p_losses += p_loss
            loss = g_loss + p_losses
            logs[f"g_loss({self.hparams.g_loss})"] = loss
            logs[f"p_loss({self.hparams.p_loss})"] = p_losses
        else:
            loss, logs = self.prediction_step(
                self.predict_label[optimizer_idx - 1],
                self.p_model[self.predict_label[optimizer_idx - 1]],
                x,
                y[self.label_type.index(self.predict_label[optimizer_idx - 1])],
            )

        self.log_dict({f"train_{k}": v for k, v in logs.items()})

        return loss

    def validation_step(
        self,
        batch: Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor],
        batch_idx: int,
    ) -> Dict[str, th.Tensor]:
        x = batch[0]
        y = batch[1:]
        _, g_logs = self.generation_step(x)
        p_logs = {}
        p_losses = 0.0
        for k, v in self.p_model.items():
            p_loss, p_log = self.prediction_step(k, v, x, y[self.label_type.index(k)])
            p_losses += p_loss
            p_logs.update(p_log)
        p_logs[f"p_loss({self.hparams.p_loss})"] = p_losses
        logs = dict(g_logs, **p_logs)

        return logs

    def validation_epoch_end(self, outputs: List[Dict[str, th.Tensor]]) -> None:
        logs = {}
        for idx, o in enumerate(outputs):
            if idx == 0:
                for k in o.keys():
                    logs[k] = 0.0
            for k in o.keys():
                logs[k] += o[k]
            if idx == len(outputs) - 1:
                for k in o.keys():
                    logs[k] /= len(outputs)

        self.log_dict({f"val_{k}": v for k, v in logs.items()})

    def configure_optimizers(self) -> Any:
        g_opt = OPTIMIZER_FACTORY[self.hparams.g_opt](
            self.g_model.parameters(),
            lr=float(self.hparams.g_lr),
            weight_decay=self.hparams.g_weight_decay,
            eps=self.hparams.g_eps,
        )
        p_opt = [
            OPTIMIZER_FACTORY[self.hparams.p_opt](
                self.p_model[label].parameters(),
                lr=float(self.hparams.p_lr),
                weight_decay=self.hparams.p_weight_decay,
                eps=self.hparams.p_eps,
            )
            for label in self.predict_label
        ]
        opts = [g_opt] + p_opt
        return opts, []
