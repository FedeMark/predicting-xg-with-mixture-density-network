"""Script to perform a training of xGMDN"""
from typing import Dict
from xgmdn_utils.model.mdn import xGMDN
from xgmdn_utils.training.lightning import xGMDNLightning
from torch.optim import AdamW
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from torchinfo import summary

CFG = {
    "model_args": {"input_dim": 180, "hidden_dim": 20},
    "optimizer_args": {"lr": 1e-3},
    "trainer_args": {"max_epochs": 100},
}


def main(cfg: Dict = CFG):
    model = xGMDN(**cfg["model_args"])
    summary(model, input_size=(32, 180))
    lightning_module = xGMDNLightning(
        model=model, optimizer=AdamW(**cfg["optmizer_args"])
    )
    logger = TensorBoardLogger("logs", name="xgmdn_experiment")

    # Initialize PyTorch Lightning Trainer with TensorBoard logger

    trainer = Trainer(logger=logger, **cfg["trainer_args"])

    # train_loader = DataLoader(train_set)
    # val_loader = DataLoader(val_set)
    # test_loader = DataLoader(test_set)

    # # Train the model
    # trainer.fit(
    #     lightning_module, train_dataloader=train_loader, val_dataloaders=val_loader
    # )


if __name__ == "__main__":
    main()
