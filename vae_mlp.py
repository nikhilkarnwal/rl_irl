from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from vae_base import BaseVAE


class MLPVAE(BaseVAE):
    def __init__(self, in_dim, enc_dims: List, z_dim):
        super(MLPVAE, self).__init__()
        
        enc = []
        enc.append(nn.Linear(in_dim, enc_dims[0]))

        for i in range(len(enc_dims)-1):
            enc.append(nn.Sequential(
                nn.Linear(enc_dims[i], enc_dims[i+1]),
                # nn.BatchNorm1d(enc_dims[i+1]),
                nn.LeakyReLU()
            ))

        self.enc = nn.Sequential(*enc)
        
        self.fc_mu = nn.Linear(enc_dims[-1], z_dim)
        self.fc_var = nn.Linear(enc_dims[-1], z_dim)

        enc_dims.reverse()
        dec_dims = enc_dims
        dec = []
        dec.append(nn.Linear(z_dim,dec_dims[0]))
        for i in range(len(dec_dims)-1):
            dec.append(nn.Sequential(
                nn.Linear(dec_dims[i], dec_dims[i+1]),
                nn.BatchNorm1d(dec_dims[i+1]),
                nn.LeakyReLU()
            ))
        
        dec.append(nn.Linear(dec_dims[-1], in_dim))

        self.dec =nn.Sequential(*dec)


        # self.loss_func = torch.nn.SmoothL1Loss()
        self.loss_func = torch.nn.MSELoss()
        
    def encoder(self, x):
        h = self.enc(x)
        return self.fc_mu(h), self.fc_var(h) # mu, log_
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # return z sample
        
    def decoder(self, z):
        return self.dec(z)
    
    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var

    def loss_function(self,x,x_cons, mu, log_var):
        mse_loss  = self.loss_func(x,x_cons)
        kld_loss = -0.5 * torch.mean(torch.sum(1 + log_var - mu.pow(2) - log_var.exp(),-1))
        final_loss = mse_loss +  100000000*kld_loss
        loss_log = {'mse_loss':mse_loss, 'kld_loss': kld_loss, 'loss': final_loss}
        return final_loss, loss_log

    
# Test model
# model = MLPVAE(45, [50, 25, 10], 8)
# print(model)