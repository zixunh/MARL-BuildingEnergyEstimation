import torch.nn as nn
from ResidualNet import ResidualLayers

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
    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim):
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
            nn.ConvTranspose2d(h_dim//2, 3, kernel_size=kernel,
                               stride=stride, padding=1)
        )

    def forward(self, x):
        return self.inverse_conv_stack(x)