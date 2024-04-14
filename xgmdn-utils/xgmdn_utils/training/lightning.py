from typing import Any
import pytorch_lightning as pl
import torch
from xgmdn_utils.model.mdn import XGMDNLoss
from xgmdn_utils.model.mavi import LogNormalLoss, xGDN
from torchmetrics import MeanSquaredError
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from lightning.pytorch.utilities import grad_norm


class xGMDNLightning(pl.LightningModule):
    def __init__(self, model, optimizer: Optimizer, scheduler: LRScheduler):
        super(xGMDNLightning, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = XGMDNLoss()
        self.mean_squared_error = MeanSquaredError()

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, prefix: str, *args: Any, **kwargs: Any):
        x, y_true = batch

        home_preds, away_preds = self.model(x)
        home_loss = self._compute_loss(preds=home_preds, y_true=y_true[..., 0])
        away_loss = self._compute_loss(preds=away_preds, y_true=y_true[..., 1])
        mixture_loss = (home_loss + away_loss) / 2

        home_mu_pred = (home_preds[0] * home_preds[2]).sum(axis=-1)
        away_mu_pred = (away_preds[0] * away_preds[2]).sum(axis=-1)
        home_mse = self.mean_squared_error(home_mu_pred, y_true[..., 0])
        away_mse = self.mean_squared_error(away_mu_pred, y_true[..., 1])

        avg_mse = (home_mse + away_mse) / 2
        # avg_std = (mixture_home_preds[1].mean() + mixture_away_preds[1].mean()) / 2

        loss = mixture_loss  # * 0.7 + avg_mse * 0.3  # + avg_std * 0.1

        self.log(f"{prefix}_loss", loss)
        self.log(f"{prefix}_mixture_loss", mixture_loss)
        self.log(f"{prefix}_home_mse", home_mse)
        self.log(f"{prefix}_away_mse", away_mse)
        self.log(f"{prefix}_avg_mse", avg_mse)
        self.log(f"{prefix}_avg_mse", avg_mse)
        self.log(f"{prefix}_home_proba", torch.exp(home_loss))
        self.log(f"{prefix}_away_proba", torch.exp(away_loss))

        # self.log(f"{prefix}_avg_home_mean", mixture_home_preds[0].mean())
        # self.log(f"{prefix}_avg_home_std", mixture_away_preds[1].mean())
        # self.log(f"{prefix}_avg_away_mean", mixture_home_preds[0].mean())
        # self.log(f"{prefix}_avg_away_std", mixture_away_preds[1].mean())

        return dict(
            loss=loss,
            home_mu_pred=home_mu_pred,
            away_mu_pred=away_mu_pred,
            y=y_true,
        )

    def training_step(self, batch, *args: Any, **kwargs: Any):
        return self._step(batch=batch, prefix="train")

    def validation_step(self, batch, *args: Any, **kwargs: Any):
        return self._step(batch=batch, prefix="val")

    def test_step(self, batch, *args: Any, **kwargs: Any):
        return self._step(batch=batch, prefix="test")

    def predict_step(self, batch, *args: Any, **kwargs: Any) -> Any:
        x, y_true = batch
        home_preds, away_preds = self.model(x)

        home_mu_pred = (home_preds[0] * home_preds[2]).sum(axis=-1)
        away_mu_pred = (away_preds[0] * away_preds[2]).sum(axis=-1)

        return dict(
            home_mu_pred=home_mu_pred,
            away_mu_pred=away_mu_pred,
            y=y_true,
        )

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = grad_norm(self, norm_type=2)
        self.log_dict(norms)

    def configure_optimizers(self):
        return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler}

    def _compute_loss(self, preds, y_true):
        mu, sigma, pi = preds

        loss = self.loss_fn(y=y_true, pi=pi, sigma=sigma, mu=mu)

        return loss


class xGDNLightning(pl.LightningModule):
    def __init__(self, model: xGDN, optimizer: Optimizer, scheduler: LRScheduler):
        super(xGDNLightning, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = LogNormalLoss()
        self.mean_squared_error = MeanSquaredError()

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, prefix: str, *args: Any, **kwargs: Any):
        x, y_true = batch

        home_preds, away_preds = self.model(x)
        # print("Home preds", torch.isnan(home_preds[0]).sum())
        # print("Away preds", torch.isnan(away_preds[0]).sum())

        if bool(home_preds[0].isnan().any()):
            print("a")
        home_loss = self._compute_mixture_loss(preds=home_preds, y_true=y_true[..., 0])
        away_loss = self._compute_mixture_loss(preds=away_preds, y_true=y_true[..., 1])
        mixture_loss = (home_loss + away_loss) / 2

        home_mean_pred = torch.exp(home_preds[0] * home_preds[1] ** 2 / 2)
        away_mean_pred = torch.exp(away_preds[0] * away_preds[1] ** 2 / 2)

        home_mse = self.mean_squared_error(home_mean_pred, y_true[..., 0])
        away_mse = self.mean_squared_error(away_mean_pred, y_true[..., 1])

        avg_mse = (home_mse + away_mse) / 2
        avg_std = (home_preds[1].mean() + away_preds[1].mean()) / 2

        loss = mixture_loss  # * 0.7 + avg_mse * 0.3  # + avg_std * 0.1

        home_proba = self.model.predict_proba(
            mu=home_preds[0], sigma=home_preds[1], y=y_true[..., 0]
        )
        away_proba = self.model.predict_proba(
            mu=away_preds[0], sigma=away_preds[1], y=y_true[..., 1]
        )

        self.log(f"{prefix}_loss", loss)
        self.log(f"{prefix}_mixture_loss", mixture_loss)
        self.log(f"{prefix}_home_mse", home_mse)
        self.log(f"{prefix}_away_mse", away_mse)
        self.log(f"{prefix}_avg_mse", avg_mse)
        self.log(f"{prefix}_avg_mse", avg_mse)
        self.log(f"{prefix}_home_proba", home_proba.mean())
        self.log(f"{prefix}_away_proba", away_proba.mean())
        self.log(f"{prefix}_avg_std", avg_std)

        return dict(
            loss=loss,
            home_mu_pred=home_preds[0],
            away_mu_pred=away_preds[0],
            y=y_true,
        )

    def training_step(self, batch, *args: Any, **kwargs: Any):
        return self._step(batch=batch, prefix="train")

    def validation_step(self, batch, *args: Any, **kwargs: Any):
        return self._step(batch=batch, prefix="val")

    def test_step(self, batch, *args: Any, **kwargs: Any):
        return self._step(batch=batch, prefix="test")

    def predict_step(self, batch, *args: Any, **kwargs: Any) -> Any:
        x, y_true = batch
        home_preds, away_preds = self.model(x)

        home_mean_pred = torch.exp(home_preds[0] * home_preds[1] ** 2 / 2)
        away_mean_pred = torch.exp(away_preds[0] * away_preds[1] ** 2 / 2)

        return dict(
            home_mean=home_mean_pred,
            away_mean=away_mean_pred,
            home_mu=home_preds[0],
            home_sigma=home_preds[1],
            away_mu=away_preds[0],
            away_sigma=away_preds[1],
            y=y_true,
        )

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = grad_norm(self, norm_type=2)
        self.log_dict(norms)

    def configure_optimizers(self):
        return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler}

    def _compute_mixture_loss(self, preds, y_true):
        mu, sigma = preds

        loss = self.loss_fn(y=y_true, mu=mu, sigma=sigma)

        return loss
