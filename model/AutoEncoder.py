import torch.nn as nn
from model.ResidualNet import ResidualLayers
import torch
from torch.distributions import MultivariateNormal, Normal, Independent

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Encoder(nn.Module):
    def __init__(self, in_dim, h_dim, res_h_dim, n_res_layers):
        super().__init__()
        kernel = 4
        stride = 2
        #Maybe remove batch norms?
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_dim, h_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(h_dim // 2),
            nn.ReLU(),
            nn.Conv2d(h_dim // 2, h_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(h_dim),
            nn.ReLU(),
            nn.Conv2d(h_dim, h_dim, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(h_dim),
            ResidualLayers(h_dim, h_dim, res_h_dim, n_res_layers)
        ) 

    def forward(self, x):
      return self.conv_block(x)
    

class Decoder(nn.Module):
    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim, out_channel=3):
        super(Decoder, self).__init__()
        kernel = 4
        stride = 2

        self.inverse_conv_stack = nn.Sequential(
            nn.ConvTranspose2d(
                in_dim, h_dim, kernel_size=kernel-1, stride=stride-1, padding=1),
            ResidualLayers(h_dim, h_dim, res_h_dim, n_res_layers),
            nn.ConvTranspose2d(h_dim, h_dim // 2,
                               kernel_size=kernel, stride=stride, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(h_dim//2, out_channel, kernel_size=kernel,
                               stride=stride, padding=1)
        )

    def forward(self, x):
        return self.inverse_conv_stack(x)
    
class AE(nn.Module):
    def __init__(self, h_dim=64, res_h_dim=32, n_res_layers=2, embedding_dim=64, lin_dim=256):
        super().__init__()

        self.h_dim = h_dim
        self.code_size = 8
        # encode image into continuous latent space
        self.encoder = Encoder(3, h_dim, n_res_layers, res_h_dim)

        #FC Projections
        self.fc1 = nn.Linear(h_dim*self.code_size*self.code_size, lin_dim)
        self.fc2 = nn.Linear(lin_dim, h_dim*self.code_size*self.code_size)

        # decode the discrete latent representation
        self.decoder = Decoder(embedding_dim, h_dim, n_res_layers, res_h_dim)

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(-1, self.h_dim*self.code_size*self.code_size)
        return self.fc1(x)

    def decode(self, z):
        z = self.fc2(z)
        z = z.view(-1, self.h_dim, self.code_size, self.code_size)
        return self.decoder(z)
    
    def train_step(self, optimizer, x_in, x_star):
        z = self.encode(x_in)
        x_hat = self.decode(z)

        loss = self.loss(x_star, x_hat)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss
    
    def test_step(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        
        loss = self.loss(x, x_hat)
        return loss
    @staticmethod
    def loss(x, x_hat):
        #Mean-Squared Error Reconstruction Loss
        criterion = nn.MSELoss()
        return criterion(x, x_hat)
    

class VAE(nn.Module):
    def __init__(self, h_dim=64, res_h_dim=32, n_res_layers=2, embedding_dim=64, lin_dim=256):
        super().__init__()
        self.z_mean = Encoder(3, h_dim, n_res_layers, res_h_dim)
        self.z_log_std = Encoder(3, h_dim, n_res_layers, res_h_dim)
        self.decoder = Decoder(embedding_dim, h_dim, n_res_layers, res_h_dim)

        self.h_dim = h_dim
        self.code_size = 8
        #FC Projections
        self.fc1 = nn.Linear(h_dim*self.code_size*self.code_size, lin_dim)
        self.fc2 = nn.Linear(lin_dim, h_dim*self.code_size*self.code_size)
    
    def _encode(self, x):
        x_mean = self.z_mean(x)
        x_flat = x_mean.view(-1, self.h_dim*self.code_size*self.code_size)
        z_mean = self.fc1(x_flat)
        

        x_std = self.z_log_std(x)
        x_flat = x_std.view(-1, self.h_dim*self.code_size*self.code_size)
        z_log_std = self.fc1(x_flat)
        z_log_std = nn.Sigmoid()(z_log_std)
        # reparameterization trick
        z_std = torch.exp(z_log_std)

        eps = torch.randn_like(z_std)
        z = z_mean + eps * z_std
        # log prob
        # 'd' not sampled on purpose
        # to show reparameterization trick
        d = Independent(Normal(z_mean, z_std), 1)
        log_prob = d.log_prob(z)
        
        return z_mean + eps * z_std, log_prob
    
    def encode(self, x):
        z, _ = self._encode(x)
        return z

    def decode(self, z):
        z = self.fc2(z)
        z = z.view(-1, self.h_dim, self.code_size, self.code_size)
        return self.decoder(z)
    
    def train_step(self, optimizer, x_in, x_star):
        z, log_prob = self._encode(x_in)
        x_hat = self.decode(z)
        loss = self.loss(x_star, x_hat, z, log_prob)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss
    
    def test_step(self, x):
        z, log_prob = self._encode(x)
        x_hat = self.decode(z)
        
        loss = self.loss(x, x_hat, z, log_prob)
        return loss

    @staticmethod
    def loss(x, x_hat, z, log_prob, kl_weight=.0001):
        criterion = nn.MSELoss()
        reconst_loss = criterion(x, x_hat)

        z_dim = z.shape[-1]
        standard_normal = MultivariateNormal(torch.zeros(z_dim).to(device), 
                                             torch.eye(z_dim).to(device))
        #print(MultivariateNormal.device)
        kld_loss = (log_prob - standard_normal.log_prob(z)).mean()
        
        return reconst_loss + kl_weight * kld_loss