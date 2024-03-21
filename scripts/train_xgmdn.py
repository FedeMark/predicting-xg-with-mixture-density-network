"""Script to perform a training of xGMDN"""

from typing import Dict, Tuple, Generator
from xgmdn_utils.model.mdn import xGMDN
from xgmdn_utils.training.lightning import xGMDNLightning
from torch.optim import AdamW
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchinfo import summary
import pandas as pd
import torch
import random
from xgmdn_utils.data.paths import SIGMAEFFE_DATA
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from torch.optim.lr_scheduler import LinearLR, SequentialLR, StepLR
import mlflow
import time
from sklearn.preprocessing import RobustScaler

torch.set_float32_matmul_precision("medium")

CFG = {
    "dataloader_kwargs": {"batch_size": 128, "num_workers": 8},
    "early_stopping": {"patience": 200, "monitor": "val_loss"},
    "model_args": {"input_dim": 176, "hidden_dims": [50, 20]},
    "optimizer_args": {"lr": 1e-4, "weight_decay": 0.05},
    "trainer_args": {
        "max_epochs": 1000,
        "accelerator": "gpu",
        "gradient_clip_val": 1,
    },
    "DATA": {
        "path": SIGMAEFFE_DATA
        / "Fbref"
        / "Pred_matches"
        / "training_dataset_5l_base_28_01_2024.csv"
    },
}


def leave_one_league_out(
    cfg: Dict,
) -> Generator[Tuple[Dataset, Dataset, Dataset], None, None]:
    df = pd.read_csv(cfg["DATA"]["path"], index_col=0)

    print(f"Dataset shape: {df.shape}")
    feat_cols = df.columns[:-6]
    y_cols = ["npxg_home", "npxg_away"]

    leagues = df.League.unique()
    for test_league in leagues:
        test_data = df[df.League == test_league]
        train_data = df[df.League != test_league]

        scaler = RobustScaler()
        train_data.loc[:, feat_cols] = scaler.fit_transform(train_data[feat_cols])
        test_data.loc[:, feat_cols] = scaler.fit_transform(test_data[feat_cols])

        train_leagues = list(set(leagues) - set([test_league]))
        validation_league = random.choice(train_leagues)

        validation_data = train_data[train_data.League == validation_league]
        train_data = train_data[train_data.League != validation_league]

        train_X = torch.from_numpy(train_data[feat_cols].to_numpy()).float()
        train_y = torch.from_numpy(train_data[y_cols].to_numpy()).float()
        validation_X = torch.from_numpy(validation_data[feat_cols].to_numpy()).float()
        validation_y = torch.from_numpy(validation_data[y_cols].to_numpy()).float()
        test_X = torch.from_numpy(test_data[feat_cols].to_numpy()).float()
        test_y = torch.from_numpy(test_data[y_cols].to_numpy()).float()

        train_dataset = TensorDataset(train_X, train_y)
        validation_dataset = TensorDataset(validation_X, validation_y)
        test_dataset = TensorDataset(test_X, test_y)

        yield train_dataset, validation_dataset, test_dataset


def main(cfg: Dict = CFG):
    model = xGMDN(**cfg["model_args"])
    summary(model, input_size=(32, 176))

    experiment_id = mlflow.create_experiment(f"xGMDN_CrossValidation_{time.time()}")

    losses = []
    test_mse = []
    for i, (train_dataset, validation_dataset, test_dataset) in enumerate(
        leave_one_league_out(cfg=cfg)
    ):
        with mlflow.start_run(run_name=f"Fold_{i}", experiment_id=experiment_id):
            model = xGMDN(**cfg["model_args"])
            optimizer = AdamW(params=model.parameters(), **cfg["optimizer_args"])
            lightning_module = xGMDNLightning(
                model=model,
                optimizer=optimizer,
                scheduler=SequentialLR(
                    optimizer=optimizer,
                    schedulers=[
                        LinearLR(
                            optimizer=optimizer,
                            start_factor=1e-6,
                            end_factor=1,
                            total_iters=100,
                        ),
                        StepLR(optimizer=optimizer, step_size=50, gamma=0.9),
                    ],
                    milestones=[100],
                ),
            )

            train_loader = DataLoader(
                dataset=train_dataset,
                persistent_workers=True,
                shuffle=True,
                **cfg["dataloader_kwargs"],
            )
            validation_loader = DataLoader(
                dataset=validation_dataset,
                persistent_workers=True,
                shuffle=True,
                **cfg["dataloader_kwargs"],
            )
            test_loader = DataLoader(dataset=test_dataset, **cfg["dataloader_kwargs"])

            logger = TensorBoardLogger("logs", name="xgmdn_experiment_new")
            # Initialize PyTorch Lightning Trainer with TensorBoard logger

            checkpoint_callback = ModelCheckpoint(mode="min", monitor="val_loss")
            trainer = Trainer(
                logger=logger,
                callbacks=[
                    EarlyStopping(**cfg["early_stopping"]),
                    LearningRateMonitor(logging_interval="step"),
                    checkpoint_callback,
                ],
                **cfg["trainer_args"],
            )

            # Train the model
            trainer.fit(
                model=lightning_module,
                train_dataloaders=train_loader,
                val_dataloaders=validation_loader,
            )

            best_model_path = checkpoint_callback.best_model_path
            lightning_module.load_state_dict(torch.load(best_model_path)["state_dict"])

            test_metrics = trainer.test(
                model=lightning_module, dataloaders=test_loader
            )[0]
            test_metrics = {k: round(v, 2) for k, v in test_metrics.items()}
            mlflow.log_metrics(test_metrics)

            train_metrics = trainer.test(
                model=lightning_module, dataloaders=train_loader
            )[0]
            train_metrics = {
                k.replace("test_", "train_"): round(v, 2)
                for k, v in train_metrics.items()
            }
            mlflow.log_metrics(train_metrics)

            validation_metrics = trainer.test(
                model=lightning_module, dataloaders=validation_loader
            )[0]
            validation_metrics = {
                k.replace("test_", "validation_"): round(v, 2)
                for k, v in validation_metrics.items()
            }
            mlflow.log_metrics(validation_metrics)

            losses.append(test_metrics["test_loss"])
            test_mse.append(test_metrics["test_avg_mse"])

    print(
        f"Average loss {sum(losses)/len(losses)}, average MSE {sum(test_mse)/len(test_mse)}"
    )


if __name__ == "__main__":
    main()
