from model.VQAE import VQAE
from utils import device, add_noise
from tqdm import tqdm
import torch
from data.FloorPlanLoader import *
import torch.nn.functional as F
import random
import json

#Reproducability Checks:
random.seed(0) #Python
torch.manual_seed(0) #Torch
np.random.seed(0) #NumPy

#Hyperparameter
batch_size = 128
n_hiddens = 32
n_residual_hiddens = 32
n_residual_layers = 1
embedding_dim = 64
n_embeddings = 218
beta = .25
lr = 3e-3
epochs = 100
noise=False
noise_weight=0.05
img_channel=1

def train_vqae():
    vqvae = VQAE(n_hiddens, n_residual_hiddens, n_residual_layers,
                n_embeddings, embedding_dim, 
                beta, img_channel).to(device)
    optimizer = torch.optim.Adam(vqvae.parameters(), lr=lr, amsgrad=False)
    train_res_recon_error = []
    test_res_recon_error = []
    best_loss = 2
    for epoch in range(epochs):
        with tqdm(train_loader, unit="batch") as tepoch:
            vqvae.train()
            for data in tepoch:
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
            for data in validation_loader:
                data = data.to(device)

                vq_loss, data_recon, perplexity = vqvae(data)
                recon_error = F.mse_loss(data_recon, data) / data_variance
                loss = recon_error.item() * batch_size

                avg_loss += loss / val_len
                test_res_recon_error.append(loss)
        
        if epoch%5==0 and avg_loss<best_loss:
            best_loss = avg_loss
            torch.save(vqvae.state_dict(), f"checkpoint/{epoch}-vqae-{avg_loss}.pt")
            with open(f"checkpoint/{epoch}-vqae_train-{avg_loss}.json", 'w', encoding ='utf8') as json_file:
                json.dump(train_res_recon_error, json_file, ensure_ascii = False)
            with open(f"checkpoint/{epoch}-vqae_test-{avg_loss}.json", 'w', encoding ='utf8') as json_file:
                json.dump(test_res_recon_error, json_file, ensure_ascii = False)

        print(f'Validation Loss: {avg_loss}')

    return vqvae, train_res_recon_error, test_res_recon_error

if __name__ == "__main__":
    #Load Dataset
    data_variance, train_set, val_set, val_len = load_FloorPlan()
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True)
    validation_loader = torch.utils.data.DataLoader(val_set, batch_size = batch_size, shuffle = False)

    vqvae, train_res_recon_error, test_res_recon_error = train_vqae()


