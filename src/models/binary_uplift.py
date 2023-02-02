from torch import optim, nn, utils, Tensor
import torch.nn.functional as F
import pytorch_lightning as pl
import torch


from typing import Any
from argparse import ArgumentParser

class BinaryUplift(pl.LightningModule):
    """PyTorch Lightning implementation of a MLP
    Args:
        hidden_dim (int, optional): dimension of hidden layer (default: ``128``).
        learning_rate (float, optional): optimizer learning rate (default: ``1e-3``).
    """
    @staticmethod
    def fetch_model(input_dim, output_dim, hidden_dim, num_hidden_layers = 1):
        layers = []
        for i in range(num_hidden_layers):
            layers.append(nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        model = nn.Sequential(*layers)
        return model 

    def __init__(self, input_dim: int, output_dim: int,
                 hidden_dim: int = 128, num_hidden_layers: int = 1, 
                 learning_rate: float = 1e-3, **kwargs: Any) -> None:
        super().__init__()
        self.save_hyperparameters()
        
        self.model = BinaryUplift.fetch_model(self.hparams.input_dim, 
                                              self.hparams.output_dim,
                                              self.hparams.hidden_dim,
                                              self.hparams.num_hidden_layers
                                             )

        self.criterion = torch.nn.BCEWithLogitsLoss()
        
    def forward(self, x: Tensor) -> Tensor:
        out = x.view(x.size(0), -1)
        out = self.model(out)
        return out

    def shared_step(self, batch: Any, batch_idx: int, step: str) -> Tensor:
        x, y, t = batch
        z = t*y + (1-t)*(1-y)
        z_hat = self(x)
        loss = self.criterion(z_hat.squeeze(), z)

        if step == "train":
            self.log("train_loss", loss)
        elif step == "val":
            self.log("val_loss", loss)
        elif step == "test":
            self.log("test_loss", loss)
        else:
            raise ValueError(f"Step {step} is not recognized. Must be 'train', 'val', or 'test'.")

        return loss

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        return self.shared_step(batch, batch_idx, "train")

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        self.shared_step(batch, batch_idx, "val")

    def test_step(self, batch: Any, batch_idx: int) -> None:
        self.shared_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--hidden_dim", type=int, default=128)
        parser.add_argument("--learning_rate", type=float, default=1e-3)
        return parser
