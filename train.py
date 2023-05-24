from model.VQAE import VQAE
from utils import device, add_noise
from tqdm import tqdm
import torch
from data.FloorPlanLoader import *
import torch.nn.functional as F
import random

#Reproducability Checks:
random.seed(0) #Python
torch.manual_seed(0) #Torch
np.random.seed(0) #NumPy

#Hyperparameter
batch_size = 64
n_hiddens = 64
n_residual_hiddens = 32
n_residual_layers = 2
embedding_dim = 64
n_embeddings = 512
beta = .25
lr = 3e-3
epochs = 6
noise=False
noise_weight=0.1

if __name__ == "__main__":
    vqvae = VQAE(n_hiddens, n_residual_hiddens, n_residual_layers,
                n_embeddings, embedding_dim, 
                beta).to(device)
    train_loader = torch.utils.data.DataLoader(training_data, batch_size = batch_size, shuffle = True)
    validation_loader = torch.utils.data.DataLoader(validation_data,batch_size= batch_size,shuffle=True)
    optimizer = torch.optim.Adam(vqvae.parameters(), lr=lr, amsgrad=False)

    train_res_recon_error = []
    test_res_recon_error = []

    for epoch in range(epochs):
        with tqdm(train_loader, unit="batch") as tepoch:
            vqvae.train()
            for data, target in tepoch:
                data_no_noise = data.to(device)
                optimizer.zero_grad()
                
                if noise:
                    data = add_noise(data_no_noise, noise_weight=noise_weight)
                else:
                    data = data_no_noise
                vq_loss, data_recon, perplexity = vqvae(data_no_noise)
                recon_error = F.mse_loss(data_recon, data_no_noise) / data_variance
                loss = recon_error + vq_loss
                loss.backward()

                optimizer.step()
                tepoch.set_postfix(loss=float(loss.detach().cpu()))
                train_res_recon_error.append(recon_error.item())

        avg_loss = 0
        vqvae.eval()
        with torch.no_grad():
            for data, target in validation_loader:
                data = data.to(device)

                vq_loss, data_recon, perplexity = vqvae(data)
                recon_error = F.mse_loss(data_recon, data) / data_variance
                loss = recon_error.item() * batch_size

                avg_loss += loss / len(validation_data)
                test_res_recon_error.append(loss)
        
        print(f'Validation Loss: {avg_loss}')

