from typing import Any, Dict, List, Tuple, Union

import pytorch_lightning as pl
import torch as th
from torch.nn import Module, ModuleDict
import torch.distributions as dist

from .layers import GAE, MLP
from .function import LOSS_FN_FACTORY, OPTIMIZER_FACTORY


class O2VModule(pl.LightningModule):
    def __init__(
        self,
        data_type: List[str],
        label_type: List[str],
        origin_dim: Dict[str, int] = None,
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

        self.g_model = GAE(
            origin_dim=origin_dim,
            hidden_dim=g_hidden_dim,
            latent_dim=latent_dim,
            activation=g_act,
        )
        self.p_model = ModuleDict(
            {
                label: MLP(
                    input_dim=latent_dim,
                    hidden_dim=p_hidden_dim,
                    output_dim=output_dim[idx],
                    activation=p_act,
                )
                for idx, label in enumerate(label_type)
            }
        )

    def forward(
        self, x: Tuple[Any]
    ) -> Tuple[Dict[str, th.Tensor], dist.Normal, dist.Normal]:
        return self.g_model(x)

    def generation_step(
        self, x: Tuple[Any], masking: th.Tensor
    ) -> Tuple[th.Tensor, Dict[str, Union[th.Tensor, float]]]:
        mu, logvar = self(x)
        p, q, z = self.g_model.sample(mu, logvar)
        x_hat = {k: v(z) for k, v in self.g_model.decoder.items()}
        # Only reality data is applied to the reconstruction loss calculation
        recon_loss = {
            data: LOSS_FN_FACTORY[self.hparams.g_loss](
                x_hat[data][masking[:, self.hparams.data_type.index(data)] == 1],
                x[self.hparams.data_type.index(data)][
                    masking[:, self.hparams.data_type.index(data)] == 1
                ],
            )
            for data in self.hparams.data_type
        }
        kl_loss = th.mean(dist.kl_divergence(q, p)) * float(self.hparams.kl_coef)
        loss = kl_loss + sum(list(recon_loss.values()))
        logs = {
            f"{data}_recon_loss({self.hparams.g_loss})": recon_loss[data]
            for data in self.hparams.data_type
        }
        logs.update(
            {
                "kl_loss": kl_loss,
                f"g_loss({self.hparams.g_loss})": loss,
            }
        )
        return loss, logs

    def prediction_step(
        self,
        label: str,
        m: Module,
        x: Tuple[Any],
        y: th.Tensor,
    ) -> Tuple[th.Tensor, Dict[str, Union[th.Tensor, float]]]:
        mu, logvar = self(x)
        p, q, z = self.g_model.sample(mu, logvar)
        y_hat = m(z)
        loss = LOSS_FN_FACTORY[self.hparams.p_loss](y_hat, y)

        logs = {f"{label}_p_loss({self.hparams.p_loss})": loss}
        return loss, logs

    def sharing_step(
        self, data: Tuple[Any], label: th.Tensor, masking: th.Tensor
    ) -> Tuple[th.Tensor, Dict[str, Union[th.Tensor, float]]]:
        g_loss, logs = self.generation_step(data, masking)
        p_losses = 0.0
        for k, v in self.p_model.items():
            y = label[:, self.hparams.label_type.index(k)]
            p_loss, _ = self.prediction_step(k, v, data, y)
            p_losses += p_loss
        loss = g_loss + p_losses
        logs["gae_loss"] = loss
        logs[f"p_loss({self.hparams.p_loss})"] = p_losses

        return loss, logs

    def training_step(
        self,
        batch: Tuple[Tuple[Any], th.Tensor, th.Tensor],
        batch_idx: int,
        optimizer_idx: int = -1,
    ) -> th.Tensor:
        data, label, masking = batch
        # If there is no predictor, only generator(GAE) is trained.
        if len(self.hparams.label_type) == 0:
            loss, logs = self.generation_step(data, masking)
        # Multi semi-supervised learning - Generator(GAE) training only
        elif optimizer_idx == 0:
            loss, logs = self.sharing_step(data, label, masking)
        # Predictor training only
        else:
            current_label = self.hparams.label_type[optimizer_idx - 1]
            y = label[:, self.hparams.label_type.index(current_label)]
            loss, logs = self.prediction_step(
                current_label, self.p_model[current_label], data, y
            )

        self.log_dict({f"train_{k}": v for k, v in logs.items()})

        return loss

    def validation_step(
        self,
        batch: Tuple[Tuple[Any], th.Tensor, th.Tensor],
        batch_idx: int,
    ) -> Dict[str, Union[th.Tensor, float]]:
        data, label, masking = batch
        loss, logs = self.sharing_step(data, label, masking)

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
        # Optimizer
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
            for label in self.hparams.label_type
        ]
        opts = [g_opt] + p_opt

        return opts, []
