import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def add_noise(tensor, mean=0., std=1., noise_weight=0.5):
    noise = torch.randn(tensor.size()).to(device) * std + mean
    return torch.clip(tensor + noise_weight * noise, 0., 1.)

