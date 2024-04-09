"""Implement simple Mixture Density Network

https://deep-and-shallow.com/2021/03/20/mixture-density-networks-probabilistic-regression-for-uncertainty-estimation/"""

from typing import Tuple
import torch.nn as nn
from torch.distributions import Normal, Gamma
import math
import torch
from typing import Sequence


def init_weights(module):
    if isinstance(module, nn.Sequential):
        for sub_module in module:
            init_weights(sub_module)
    elif isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight)
        nn.init.constant_(module.bias, 0.1)


class StdActivation(nn.Module):
    def forward(self, x):
        return nn.functional.relu(x) + 0.1 + 1e-15


class xGMDN(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dims: Sequence[int], num_gaussians: int = 2
    ):
        super(xGMDN, self).__init__()
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=hidden_dims[0]),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=hidden_dims[1]),
        )  # layer to reduce/embed the feature space

        init_weights(self.embedding)

        self.mdn_home = MDN(input_dim=hidden_dims[1], num_gaussians=num_gaussians)
        self.mdn_away = MDN(input_dim=hidden_dims[1], num_gaussians=num_gaussians)

    def forward(self, x):
        latent_space = self.embedding(x)

        return self.mdn_home(latent_space), self.mdn_away(latent_space)


class MDN(nn.Module):
    def __init__(self, input_dim: int, num_gaussians: int = 2):
        super(MDN, self).__init__()
        self.mean_extractor = nn.Linear(
            input_dim, num_gaussians
        )  # using ReLU because xG means can't be negative
        self.std_extractor = nn.Linear(input_dim, num_gaussians)
        self.pi_extractor = nn.Sequential(
            nn.Linear(input_dim, num_gaussians), nn.Softmax(dim=-1)
        )

        init_weights(self.mean_extractor)
        init_weights(self.std_extractor)
        init_weights(self.pi_extractor)

    def forward(self, x) -> Tuple[float, float, float, float]:
        pi = self.pi_extractor(x)
        mu = nn.ReLU()(
            self.mean_extractor(x)
        )  # using ReLU because xG means can't be negative
        sigma = StdActivation()(self.std_extractor(x))

        return mu, sigma, pi


class XGMDNLoss(nn.Module):
    def forward(self, y, pi, sigma, mu):
        return self.calculate_loss(y=y, pi=pi, sigma=sigma, mu=mu)

    def gaussian_probability(self, sigma, mu, target):
        """
        Inspired by https://github.com/sagelywizard/pytorch-mdn/blob/master/mdn/mdn.py
        """
        target = torch.unsqueeze(target, dim=1).expand_as(sigma)
        model_dist = Normal(mu, sigma)

        return model_dist.log_prob(target)

    def log_prob(self, pi, sigma, mu, y):
        log_component_prob = self.gaussian_probability(sigma, mu, y)
        weighted_log_component_prob = pi * log_component_prob
        return torch.logsumexp(weighted_log_component_prob, dim=-1)

    def calculate_loss(self, y, pi, sigma, mu, tag="train"):
        # NLL Loss
        log_prob = self.log_prob(pi, sigma, mu, y)
        loss = torch.mean(-log_prob)

        return loss
