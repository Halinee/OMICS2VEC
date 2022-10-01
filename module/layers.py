from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch as th
import torch.distributions as dist
import torch.nn as nn

from .function import ACTIVATION_FN_FACTORY


class Encoder(nn.Sequential):
    def __init__(
        self,
        origin_dim: int = 128,
        hidden_dim: List[int] = None,
        activation: str = "relu",
    ):
        if hidden_dim is None:
            hidden_dim = [1024, 512, 256, 128]
        act_fn = ACTIVATION_FN_FACTORY[activation]
        layers = []

        for idx, dim in enumerate(hidden_dim):
            if idx == 0:
                in_dim = origin_dim
                out_dim = dim
            else:
                in_dim = hidden_dim[idx - 1]
                out_dim = dim

            layers.extend(
                [
                    nn.Linear(in_dim, out_dim),
                    nn.LayerNorm(out_dim),
                    act_fn,
                ]
            )

        super(Encoder, self).__init__(*layers)


class Decoder(nn.Sequential):
    def __init__(
        self,
        latent_dim: int = 256,
        hidden_dim: List[int] = None,
        origin_dim: int = 128,
        activation: str = "relu",
    ):
        if hidden_dim is None:
            hidden_dim = [1024, 512, 256, 128]
        act_fn = ACTIVATION_FN_FACTORY[activation]
        layers = []

        for idx, dim in enumerate(hidden_dim):
            if idx == 0:
                in_dim = latent_dim
                out_dim = dim
            else:
                in_dim = hidden_dim[idx - 1]
                out_dim = dim

            layers.extend(
                [
                    nn.Linear(in_dim, out_dim),
                    nn.LayerNorm(out_dim),
                    act_fn,
                ]
            )
        # Add output layer to generate origin dimension
        layers.extend(
            [
                nn.Linear(hidden_dim[-1], origin_dim),
                nn.LayerNorm(origin_dim),
                act_fn,
            ]
        )
        super(Decoder, self).__init__(*layers)


class GAE(nn.Module):
    def __init__(
        self,
        origin_dim: Dict[str, int] = None,
        hidden_dim: List[int] = None,
        latent_dim: int = 100,
        activation: str = "relu",
    ):
        super(GAE, self).__init__()
        self.data_type = list(origin_dim.keys())
        if hidden_dim is None:
            hidden_dim = [1024, 512, 256, 128]
        concat_dim = hidden_dim[-1] * len(origin_dim.keys())
        act_fn = ACTIVATION_FN_FACTORY[activation]

        self.encoder = nn.ModuleDict(
            {
                data_type: Encoder(
                    origin_dim=dim, hidden_dim=hidden_dim, activation=activation
                )
                for data_type, dim in origin_dim.items()
            }
        )
        self.decoder = nn.ModuleDict(
            {
                data_type: Decoder(
                    latent_dim=latent_dim,
                    hidden_dim=hidden_dim,
                    origin_dim=dim,
                    activation=activation,
                )
                for data_type, dim in origin_dim.items()
            }
        )
        self.concater = nn.Sequential(
            nn.Linear(concat_dim, concat_dim), nn.LayerNorm(concat_dim), act_fn
        )
        self.mu = nn.Sequential(
            nn.Linear(concat_dim, latent_dim), nn.LayerNorm(latent_dim), act_fn
        )
        self.logvar = nn.Sequential(
            nn.Linear(concat_dim, latent_dim), nn.LayerNorm(latent_dim), act_fn
        )

    def forward(self, x: Tuple[Any]) -> Tuple[dist.Normal, dist.Normal]:
        # Encode data
        x = [
            encoder(x[self.data_type.index(data)])
            for data, encoder in self.encoder.items()
        ]
        c_x = th.cat(x, dim=-1)
        c_x = self.concater(c_x)
        mu = self.mu(c_x)
        logvar = self.logvar(c_x)

        return mu, logvar

    @staticmethod
    def sample(
        mu: th.Tensor, logvar: th.Tensor
    ) -> Tuple[dist.Normal, dist.Normal, Union[float, th.Tensor]]:
        std = th.exp(logvar / 2)
        p = dist.Normal(th.zeros_like(mu), th.ones_like(std))
        q = dist.Normal(mu, std)
        z = q.rsample()

        return p, q, z

    def encode_from_omics(
        self, multi_omics: Dict[str, Union[np.ndarray, th.Tensor]]
    ) -> Union[float, th.Tensor]:
        x = [encoder(multi_omics[data]) for data, encoder in self.encoder.items()]
        c_x = th.cat(x, dim=-1)
        c_x = self.concater(c_x)
        mu = self.mu(c_x)

        return mu


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 100,
        hidden_dim: List[int] = None,
        output_dim: int = 33,
        activation: str = "relu",
    ):
        super(MLP, self).__init__()
        if hidden_dim is None:
            hidden_dim = [200, 100]
        dims = [input_dim] + hidden_dim + [output_dim]
        act_fn = ACTIVATION_FN_FACTORY[activation]

        layer_list = []
        for i in range(len(dims) - 2):
            layer_list.extend(
                [
                    nn.Linear(dims[i], dims[i + 1]),
                    nn.LayerNorm(dims[i + 1]),
                    act_fn,
                ]
            )
        layer_list.append(nn.Linear(dims[-2], dims[-1]))
        self.model = nn.Sequential(*layer_list)

    def forward(self, x: th.Tensor):
        return self.model(x)
