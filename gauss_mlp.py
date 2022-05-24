from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Normal

from vae_base import BaseVAE


class GaussMLP(BaseVAE):
    def __init__(self, in_dim, enc_dims: List, z_dim):
        super(GaussMLP, self).__init__()
        
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

        self.mu_pr = torch.zeros(z_dim)
        self.std_pr = torch.ones(z_dim)

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
        mu = mu
        std = torch.exp(0.5*log_var)
        std = std.clamp(0.000001,7)
        dist = Normal(mu,torch.ones_like(mu)*std)
        z = dist.rsample()
        # z = Normal(torch.zeros_like(mu),torch.ones_like(mu)).sample()
        return z, dist.entropy()  # return z sample
        # return z, Normal(mu, std).log_prob(z)
        
    def decoder(self, z):
        return self.dec(z)
    
    def forward(self, x):
        mu, log_var = self.encoder(x)
        z, log_prob = self.sampling(mu, log_var)
        z = self.decoder(z)
        return (log_prob,z)

    def forward_2(self,x):
        mu = self.fc_mu(self.enc(x))
        x =  self.fc_final(mu)
        x = F.sigmoid(x)
        return (x,)
        

    def loss_function(self,x,log_prob,z):
        loss_val =  torch.mean(log_prob)
        cons_loss = self.loss_func(x,z)
        f_loss = loss_val+cons_loss
        return f_loss, {"loss_prob" : loss_val, "cons_loss": cons_loss, "loss": f_loss}


    
# Test model
# model = MLPVAE(45, [50, 25, 10], 8)
# print(model)