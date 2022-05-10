from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from abc import abstractmethod


class BaseVAE(nn.Module):
    def __init__(self):
        super(BaseVAE, self).__init__()
        
    def encoder(self, x):
        """encode input x and return mean, log_var

        Args:
            x (_type_): _description_

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError
    
    def sampling(self, mu, log_var):
        """Generate a random sample from mu and log_var

        Args:
            mu (_type_): _description_
            log_var (_type_): _description_

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError
        
    def decoder(self, z):
        """Decode sampled z

        Args:
            z (_type_): _description_

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError
    
    @abstractmethod
    def forward(self, x):
        pass
        

# Test model
# model = BaseVAE([50, 25, 10], 8)
# print(model)