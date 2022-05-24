import pytorch_lightning as pl
import os
import math
import torch
from torch import Tensor, optim

class VAEXperiment(pl.LightningModule):

    def __init__(self,
                 vae_model,
                 params: dict) -> None:
        super(VAEXperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        self.prepare_data_per_node = False

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        
        results = self.forward(batch)
        train_loss, loss_log = self.model.loss_function(batch,*results)

        self.log_dict({key: val.item() for key, val in loss_log.items()}, sync_dist=True)

        return train_loss

    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        results = self.forward(batch)
        train_loss, loss_log = self.model.loss_function(batch,*results)

        self.log_dict({f"val_{key}": val.item() for key, val in loss_log.items()}, sync_dist=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        return optimizer



