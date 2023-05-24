import torch.nn as nn
import torch.nn.functional as F

class ResidualLayerBlock(nn.Module):
    def __init__(self, in_dim, h_dim, res_h_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=in_dim, out_channels = res_h_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(res_h_dim),
            nn.ReLU(),
            nn.Conv2d(in_channels=res_h_dim, out_channels=h_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(h_dim)
        )

    def forward(self, x):
        out = x + self.block(x)
        return out

class ResidualLayers(nn.Module):
    def __init__(self, in_dim, h_dim, res_h_dim, n_res_layers):
        super().__init__()
        self.n_res_layers = n_res_layers
        self.layers = nn.ModuleList(
            [ResidualLayerBlock(in_dim, h_dim, res_h_dim)] * n_res_layers
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return F.relu(x)