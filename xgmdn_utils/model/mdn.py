"""Implement simple Mixture Density Network"""

from typing import Tuple
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal


class StdActivation(nn.Module):
    def forward(self, x):
        return nn.functional.elu(x) + 1 + 1e-15


class xGMDN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super(xGMDN, self).__init__()
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU()
        )  # layer to reduce/embed the feature space
        self.mean_extractor = nn.Sequential(
            nn.Linear(hidden_dim, 2), nn.ReLU()
        )  # using ReLU because xG means can't be negative
        self.std_extractor = nn.Sequential(
            nn.Linear(hidden_dim, 2), StdActivation()
        )  # using ReLU because xG means can't be negative

    def forward(self, x) -> Tuple[float, float, float, float]:
        embedding = self.embedding(x)
        mu = self.mean_extractor(embedding)
        sigma = self.std_extractor(embedding)

        return mu, sigma


def xgmdn_loss(y_true: Tensor, mus: Tensor, sigmas: Tensor):
    """
    Custom loss function for xGMDN.
    """

    def negative_gaussin_log_likelihood(
        y_true: Tensor, mu: Tensor, sigma: Tensor
    ) -> Tensor:
        # Create Normal distributions using predicted parameters
        dist = Normal(loc=mu, scale=sigma)

        # return the negative log likelihood of the true values under the predicted distributions
        return -dist.log_prob(y_true).mean()  # check batch dim

    return negative_gaussin_log_likelihood(
        y_true=y_true, mu=mus[..., 0], sigma=sigmas[..., 0]
    ) + negative_gaussin_log_likelihood(
        y_true=y_true, mu=mus[..., 1], sigma=sigmas[..., 1]
    )
