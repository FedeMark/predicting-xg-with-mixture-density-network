import pytorch_lightning as pl
from xgmdn_utils.model.mdn import xgmdn_loss
from torchmetrics import MeanSquaredError
from torch.optim import Optimizer


class xGMDNLightning(pl.LightningModule):
    def __init__(self, model, optimizer: Optimizer):
        super(xGMDNLightning, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = xgmdn_loss
        self.mean_squared_error = MeanSquaredError()

    def configure_optimizers(self):
        return self.optimizer

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y_true = batch
        mus, sigmas = self.model(x)
        loss = self.loss_fn(y_true, mus, sigmas)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_true = batch
        mus, sigmas = self.model(x)
        loss = self.loss_fn(y_true, mus, sigmas)
        mse = self.mean_squared_error(mus, y_true)
        self.log("val_loss", loss)
        self.log("val_mse", mse)
        return loss
