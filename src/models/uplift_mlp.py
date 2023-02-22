from torch import optim, nn, utils, Tensor
import torch.nn.functional as F
import pytorch_lightning as pl
import torch

from typing import Any
from argparse import ArgumentParser

    
class UpliftMLP(pl.LightningModule):
    """PyTorch Lightning implementation of a MLP
    Args:
        hidden_dim (int, optional): dimension of hidden layer (default: ``128``).
        learning_rate (float, optional): optimizer learning rate (default: ``1e-3``).
    """
    @staticmethod
    def fetch_model(input_dim, output_dim, hidden_dim, num_hidden_layers=1, use_layer_norm=False, dropout_prob=0.0):
        layers = []
        for i in range(num_hidden_layers):
            layers.append(nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            if dropout_prob > 0:
                layers.append(nn.Dropout(p=dropout_prob))          
        layers.append(nn.Linear(input_dim if num_hidden_layers == 0 else hidden_dim,
                                output_dim))
        model = nn.Sequential(*layers)
        return model

    def __init__(self, input_dim: int, output_dim: int,
                 hidden_dim: int = 128, num_hidden_layers: int = 1, 
                 l2_weight: float = 0, l2_diff: float = 0, grad_glip: float = 0.9,
                 learning_rate: float = 1e-3, optimizer: str = "Adam", 
                 use_layer_norm: bool = False, dropout_prob: float = 0.0,
                 lr_scheduler: str = None, **kwargs: Any) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.model_t = UpliftMLP.fetch_model(self.hparams.input_dim, 
                                              self.hparams.output_dim,
                                              self.hparams.hidden_dim,
                                              self.hparams.num_hidden_layers,
                                              self.hparams.use_layer_norm,
                                              self.hparams.dropout_prob
                                             )
        
        self.model_c = UpliftMLP.fetch_model(self.hparams.input_dim, 
                                      self.hparams.output_dim,
                                      self.hparams.hidden_dim,
                                      self.hparams.num_hidden_layers,
                                      self.hparams.use_layer_norm,
                                      self.hparams.dropout_prob
                                     )


        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.l2_weight = l2_weight
        self.l2_diff = l2_diff
        self.l2_loss = nn.MSELoss()
        
    def forward(self, x: Tensor) -> Tensor:
        out = x.view(x.size(0), -1)
        out_t = self.model_t(out)
        out_c = self.model_c(out)
        return out_t - out_c
    
    def regularization(self):
        reg_loss = 0
        for param_t, param_c in zip(self.model_t.parameters(), self.model_c.parameters()):
            reg_loss += self.l2_loss(param_t, param_c)        
        return reg_loss
    
    def l2_regularization(self, params):
        reg_loss = 0
        for param in params:
            reg_loss += self.l2_loss(param, torch.zeros_like(param))
        return reg_loss
        

    def shared_step(self, batch: Any, batch_idx: int, step: str) -> Tensor:
        x, y, t = batch
        t_indx, c_indx = torch.where(t==1)[0], torch.where(t==0)[0]
        x_t, x_c = x[t_indx,:], x[c_indx,:]
        y_t, y_c = y[t_indx], y[c_indx]
        
        y_t_pred = self.model_t(x_t)
        y_c_pred = self.model_c(x_c)
        
        loss_t = self.criterion(y_t_pred, y_t.view(-1,1))
        loss_c = self.criterion(y_c_pred, y_c.view(-1,1))
        loss = loss_t + loss_c
        if self.l2_diff > 0:
            loss += self.l2_diff*self.regularization()
        if self.l2_weight > 0:
            loss += self.l2_weight*self.l2_regularization(self.model_t.parameters())
            loss += self.l2_weight*self.l2_regularization(self.model_c.parameters())

        if step == "train":
            self.log("train_loss", loss)
            opt = self.optimizers()
            opt.zero_grad()
            self.manual_backward(loss)
            self.clip_gradients(opt, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
            opt.step()
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

    def configure_scheduler(self, optimizer, lr_scheduler):
        scheduler = getattr(torch.optim.lr_scheduler, lr_scheduler)(optimizer, 100)
        return scheduler

    def configure_optimizers(self):
        # ref: https://github.com/Lightning-AI/lightning/issues/7576
        optimizer = getattr(torch.optim, self.hparams.optimizer)(
            self.parameters(),
            lr=self.hparams.learning_rate,
        )
        if self.hparams.lr_scheduler is None:
            return optimizer
        scheduler = self.configure_scheduler(optimizer, self.hparams.lr_scheduler)
        return [optimizer], [scheduler]         

    @staticmethod
    def add_model_specific_args(parent_parser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--hidden_dim", type=int, default=128)
        parser.add_argument("--learning_rate", type=float, default=1e-3)
        return parser    
    