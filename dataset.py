import os
import torch
from torch import Tensor
from pathlib import Path
from typing import List, Optional, Sequence, Union, Any, Callable
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import zipfile
from imitation.data import types
import numpy as np

class VAEDataSet(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        return self.data[idx]

# Add your custom dataset class here
class VAEDataModule(LightningDataModule):
    def __init__(self, transitions: types.Transitions, batch_size=1024):
        super().__init__()
        self.data = transitions
        self.batch_size = batch_size
        self.prepare_data_per_node = False

    def setup(self,stage: Optional[str] = None):
        ids = np.arange(self.__len__(), dtype=int)
        np.random.shuffle(ids)
        train_id  = [0, self.__len__()*80//100]
        val_id = [train_id[-1] , train_id[-1]+self.__len__()*10//100]
        test_id = [val_id[-1], self.__len__()]
        feat_data = np.concatenate((self.data.obs, self.data.acts.reshape((self.data.obs.shape[0],-1))), axis=-1, dtype=np.float32)
        # print(feat_data.dtype)
        self.train_dataset = VAEDataSet(feat_data[ids[train_id[0]: train_id[1]]])
        self.val_dataset = VAEDataSet(feat_data[ids[val_id[0]: val_id[1]]])
        self.test_dataset = VAEDataSet(feat_data[ids[test_id[0]: test_id[1]]])

        print(f"Size : Train-{self.train_dataset.__len__()}, Val-{self.val_dataset.__len__()}, Test-{self.test_dataset.__len__()}")
    
    
    def __len__(self):
        if isinstance(self.data,list):
            print('Traj-shape:',self.data[0].obs.shape, self.data[0].acts.shape)
            return self.data[0].obs.shape[0]
        else:
            print('Traj-shape:',self.data.obs.shape, self.data.acts.shape)
            return self.data.obs.shape[0]

    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, self.batch_size)


    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, self.batch_size)