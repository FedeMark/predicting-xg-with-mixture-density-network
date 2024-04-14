"""
Implement single distribution network.
"""

from typing import Tuple
import torch.nn as nn
from torch.distributions import LogNormal
import torch
from typing import Sequence


def init_weights(module):
    if isinstance(module, nn.Sequential):
        for sub_module in module:
            init_weights(sub_module)
    elif isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight)
        nn.init.constant_(module.bias, 0.01)


class StdActivation(nn.Module):
    def forward(self, x):
        return nn.functional.relu(x) + 0.1


class xGDN(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int]):
        super(xGDN, self).__init__()
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=hidden_dims[0]),
            # nn.LayerNorm(normalized_shape=hidden_dims[0]),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=hidden_dims[1]),
            # nn.LayerNorm(normalized_shape=hidden_dims[1]),
        )  # layer to reduce/embed the feature space

        init_weights(self.embedding)

        self.mdn_home = LogNormalDN(input_dim=hidden_dims[1])
        self.mdn_away = LogNormalDN(input_dim=hidden_dims[1])

    def forward(self, x):
        latent_space = self.embedding(x)

        return self.mdn_home(latent_space), self.mdn_away(latent_space)

    def predict_proba(self, mu, sigma, y, log: bool = False) -> torch.Tensor:
        model_dist = LogNormal(mu, sigma)

        log_proba = model_dist.log_prob(y + 1e-16)

        if not log:
            log_proba = torch.exp(log_proba)

        return log_proba


class LogNormalDN(nn.Module):
    def __init__(self, input_dim: int):
        super(LogNormalDN, self).__init__()
        self.mean_extractor = nn.Linear(input_dim, 1)
        self.std_extractor = nn.Linear(input_dim, 1)

        nn.init.xavier_normal_(self.mean_extractor.weight)
        nn.init.constant_(self.mean_extractor.bias, 0.05)

        nn.init.kaiming_normal_(self.std_extractor.weight)
        nn.init.constant_(self.std_extractor.bias, 0.05)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        # print("x", torch.sum(torch.isnan(x)))
        mu = nn.Tanh()(self.mean_extractor(x)) * 5
        mu = self.mean_extractor(x)
        sigma = StdActivation()(self.std_extractor(x))

        return mu[..., 0], sigma[..., 0]

    def predict_proba(self, x, y, log: bool = False) -> torch.Tensor:
        mu, sigma = self(x)

        return self._predict_proba(mu=mu, sigma=sigma, y=y, log=log)

    def _predict_proba(self, mu, sigma, y, log: bool = False) -> torch.Tensor:
        model_dist = LogNormal(mu, sigma)

        log_proba = model_dist.log_prob(y + 1e-16)

        if not log:
            log_proba = torch.exp(log_proba)

        return log_proba


class LogNormalLoss(nn.Module):
    def forward(self, y, mu, sigma):
        model_dist = LogNormal(mu, sigma)

        log_proba = model_dist.log_prob(y + 1e-16)

        log_proba = log_proba.clamp_(
            min=torch.log(torch.tensor(1e-3)).to(log_proba.device)
        )

        # print("Loss", torch.mean(-log_proba))
        return torch.mean(-log_proba)
