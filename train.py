from model.VQAE import VQAE
from model.MARL import MARL
from utils import device, add_noise
from tqdm import tqdm
import torch
from data.FloorPlanLoader import *
import torch.nn.functional as F
import random
import json



USE_MULTISCALE = True
USE_MULTITASK = True

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
img_channel=3 if USE_MULTISCALE else 1


def train_marl(train_loader=None, validation_loader=None, 
               data_variance=None, val_len=None, year_label_num=None, category_num=None,
               get_pretrain=True, use_multi_task=USE_MULTITASK):
    
    vqae = VQAE(n_hiddens, n_residual_hiddens, n_residual_layers,
                n_embeddings, embedding_dim, 
                beta, img_channel).to(device)
    if get_pretrain:
        vqae.load_state_dict(torch.load("./best_checkpoint/final/55-vqae-0.04753296934928414.pt"))

    marl = MARL(vqae, USE_MULTITASK, year_label_num, category_num)
    optimizer = torch.optim.Adam(marl.parameters(), lr=lr, amsgrad=False)
    train_recon_error = []
    train_height_error = []
    train_age_error = []
    train_usage_error = []
    test_recon_error = []
    test_height_error = []
    test_age_error = []
    test_usage_error = []


    best_loss = 1e10
    for epoch in range(0, epochs):
        with tqdm(train_loader, unit="batch") as tepoch:
            marl.train()
            for data_dict in tepoch:
                data = data_dict['image_tensor']
                bs = data.shape[0]
                data_no_noise = data.to(device)
                optimizer.zero_grad()

                if noise:
                    data = add_noise(data_no_noise, noise_weight=noise_weight)
                else:
                    data = data_no_noise
                pred = marl(data)

                # recon loss
                vq_loss, data_recon, perplexity = pred['vqae']
                recon_error = F.mse_loss(data_recon, data) / data_variance
                train_recon_error.append(recon_error.item())
                

                if USE_MULTITASK:
                    # height infer
                    height_pred = pred['height']
                    height_error = F.mse_loss(height_pred, data_dict['height'].to(device).view(bs,-1))
                    train_height_error.append(height_error.item())
                    # age infer
                    age_pred = pred['age']
                    labels = data_dict['age_label'].to(device).long()
                    age_error = F.cross_entropy(age_pred, labels)*0.3
                    train_age_error.append(age_error.item())
                    # category infer
                    category_pred = pred['category']
                    labels = data_dict['cate_onehot'].to(device)
                    criterion = torch.nn.BCEWithLogitsLoss()
                    category_error = criterion(category_pred, labels)*0.7
                    train_usage_error.append(category_error.item())

                loss = (recon_error + vq_loss) + height_error + age_error + category_error
                loss.backward()
                optimizer.step()
                tepoch.set_postfix(recon_error=float((recon_error+ vq_loss).detach().cpu()),
                                   height_error=float(height_error.detach().cpu()),
                                   age_error=float(age_error.detach().cpu()),
                                   category_error=float(category_error.detach().cpu()))
                
        avg_loss = 0
        marl.eval()
        with torch.no_grad():
            for data_dict in validation_loader:
                data = data_dict['image_tensor']
                bs = data.shape[0]
                data = data.to(device)

                pred = marl(data)
                # recon loss
                vq_loss, data_recon, perplexity = pred['vqae']
                recon_error = F.mse_loss(data_recon, data) / data_variance
                test_recon_error.append(recon_error.item())

                if USE_MULTITASK:
                    # height infer
                    height_pred = pred['height']
                    height_error = F.mse_loss(height_pred, data_dict['height'].to(device).view(bs,-1))
                    test_height_error.append(height_error.item())
                    # age infer
                    age_pred = pred['age']
                    labels = data_dict['age_label'].to(device).long()
                    age_error = F.cross_entropy(age_pred, labels)
                    test_age_error.append(age_error.item())
                    # category infer
                    category_pred = pred['category']
                    labels = data_dict['cate_onehot'].to(device)
                    criterion = torch.nn.BCEWithLogitsLoss()
                    category_error = criterion(category_pred, labels)
                    test_usage_error.append(category_error.item())

                loss = (recon_error.item() \
                        + height_error.item()\
                        + age_error.item()\
                        + category_error.item()\
                        ) * batch_size
                avg_loss += loss / val_len
                
                
        if avg_loss<best_loss:
            best_loss = avg_loss
            best_epoch = epoch
            torch.save(marl.state_dict(), f"./checkpoint/{best_epoch}-marl-{best_loss}.pt")
            torch.save(optimizer.state_dict(), f"./checkpoint/{best_epoch}-adam-{best_loss}.pt")
            if USE_MULTITASK:
                error = {
                    'train_recon_error': train_recon_error,
                    'train_height_error': train_height_error,
                    'train_age_error': train_age_error,
                    'train_usage_error': train_usage_error,
                    'test_recon_error': test_recon_error,
                    'test_height_error': test_height_error,
                    'test_age_error': test_age_error,
                    'test_usage_error': test_usage_error
                }
            else:
                error = {
                    'train_recon_error': train_recon_error,
                    'test_recon_error': test_recon_error
                }
            with open(f"./checkpoint/{best_epoch}-error-{best_loss}.json", 'w', encoding ='utf8') as json_file:
                json.dump(error, json_file, ensure_ascii = False)

        print(f'Validation Loss: {avg_loss}')


if __name__ == "__main__":
    #Load Dataset
    floor = FloorPlanDataset(multi_scale=True, root='./data/data_root/data00/', data_config='./data/data_config/', preprocess=True)
    data_variance = floor.var
    val_len = int(len(floor)/10)
    train_set, val_set = torch.utils.data.random_split(floor, [len(floor)-val_len, val_len])

    print(f"data shape: {floor[0]['image_tensor'].shape}, dataset size: {len(floor)}, data variance: {data_variance}")
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True)
    validation_loader = torch.utils.data.DataLoader(val_set, batch_size = batch_size, shuffle = False)

    train_marl(train_loader, validation_loader, \
               floor.var, int(len(floor)/10), floor.age_label_num, floor.category_num)


