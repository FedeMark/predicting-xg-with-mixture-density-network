from pytorch_tabnet.pretraining import TabNetPretrainer
from pytorch_tabnet.tab_network import TabNet
from xgmdn_utils.model.mavi import LogNormalDN
from torch import nn
import torch
from torch.distributions import LogNormal


class xGTabNet(nn.Module):
    def __init__(self, tabnet_pretrainer: TabNetPretrainer) -> None:
        super(xGTabNet, self).__init__()

        pretrainer_network = tabnet_pretrainer.network

        self._embedder = pretrainer_network.embedder
        self._encoder = pretrainer_network.encoder

        self.mdn_home = LogNormalDN(input_dim=pretrainer_network.n_d)
        self.mdn_away = LogNormalDN(input_dim=pretrainer_network.n_d)

    def forward(self, x):
        embedded_input = self._embedder(x)
        steps_output, _ = self._encoder(embedded_input)
        latent_space = torch.sum(torch.stack(steps_output, dim=0), dim=0)

        return self.mdn_home(latent_space), self.mdn_away(latent_space)

    def predict_proba(self, mu, sigma, y, log: bool = False) -> torch.Tensor:
        model_dist = LogNormal(mu, sigma)

        log_proba = model_dist.log_prob(y + 1e-16)

        if not log:
            log_proba = torch.exp(log_proba)

        return log_proba
