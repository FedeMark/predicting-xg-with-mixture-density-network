from typing import Any
import pytorch_lightning as pl
from xgmdn_utils.model.mdn import XGMDNLoss
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
        loss = home_loss + away_loss

        home_mse = self.mean_squared_error(
            (home_preds[0] * home_preds[2]).sum(axis=1), y_true[..., 0]
        )
        away_mse = self.mean_squared_error(
            (away_preds[0] * away_preds[2]).sum(axis=1), y_true[..., 1]
        )

        self.log(f"{prefix}_loss", loss)
        self.log(f"{prefix}_home_mse", home_mse)
        self.log(f"{prefix}_away_mse", away_mse)
        self.log(f"{prefix}_avg_mse", (home_mse + away_mse) / 2)

        return loss

    def training_step(self, batch, *args: Any, **kwargs: Any):
        return self._step(batch=batch, prefix="train")

    def validation_step(self, batch, *args: Any, **kwargs: Any):
        return self._step(batch=batch, prefix="val")

    def test_step(self, batch, *args: Any, **kwargs: Any):
        return self._step(batch=batch, prefix="test")

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
