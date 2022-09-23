from typing import List, Tuple, Union

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
        base_act: str = "relu",
    ):
        if hidden_dim is None:
            hidden_dim = [1024, 512, 256, 128]
        act_fn = ACTIVATION_FN_FACTORY[base_act]
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
        base_act: str = "relu",
        output_act: str = "sigmoid",
    ):
        if hidden_dim is None:
            hidden_dim = [1024, 512, 256, 128]
        act_fn = ACTIVATION_FN_FACTORY[base_act]
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
                ACTIVATION_FN_FACTORY[output_act],
            ]
        )
        super(Decoder, self).__init__(*layers)


class VAE(nn.Module):
    def __init__(
        self,
        origin_dim: int = 128,
        hidden_dim: List[int] = None,
        latent_dim: int = 256,
        base_act: str = "relu",
        output_act: str = "sigmoid",
    ):
        super(VAE, self).__init__()
        if hidden_dim is None:
            hidden_dim = [1024, 512, 256, 128]
        act_fn = ACTIVATION_FN_FACTORY[base_act]

        self.encoder = Encoder(origin_dim, hidden_dim, base_act)
        self.decoder = Decoder(
            latent_dim, hidden_dim[::-1], origin_dim, base_act, output_act
        )
        self.mu = nn.Sequential(
            nn.Linear(hidden_dim[-1], latent_dim), nn.LayerNorm(latent_dim), act_fn
        )
        self.logvar = nn.Sequential(
            nn.Linear(hidden_dim[-1], latent_dim), nn.LayerNorm(latent_dim), act_fn
        )

    def forward(
        self, x: th.Tensor
    ) -> Tuple[th.Tensor, dist.Normal, dist.Normal, Union[float, th.Tensor]]:
        # Encode data
        x = self.encoder(x)
        mu = self.mu(x)
        logvar = self.logvar(x)
        p, q, z = self.sample(mu, logvar)
        # Decode data
        return self.decoder(z), p, q, z

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
        self, omics: Union[np.ndarray, th.Tensor]
    ) -> Union[float, th.Tensor]:
        x = th.as_tensor(omics, dtype=th.float32)
        x = self.encoder(x)
        mu = self.mu(x)
        logvar = self.logvar(x)
        _, _, z = self.sample(mu, logvar)

        return z


class MLP(nn.Sequential):
    def __init__(
        self,
        input_dim: int = 100,
        hidden_dim: List[int] = None,
        output_dim: int = 33,
        act: str = "relu",
    ):
        super(MLP, self).__init__()
        if hidden_dim is None:
            hidden_dim = [200, 100]
        dims = [input_dim] + hidden_dim + [output_dim]
        act_fn = ACTIVATION_FN_FACTORY[act]

        layers = []
        for i in range(len(dims) - 2):
            layers.extend(
                [
                    nn.Linear(dims[i], dims[i + 1]),
                    nn.LayerNorm(dims[i + 1]),
                    act_fn,
                ]
            )
        layers.append(nn.Linear(dims[-2], dims[-1]))
        super(MLP, self).__init__(*layers)
