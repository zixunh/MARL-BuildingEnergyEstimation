import torch.nn as nn
import torch
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class MetaInfer(nn.Module):
    def __init__(self, out_dim):
        super(MetaInfer, self).__init__()
        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels = 1, kernel_size=7, stride=3),
            nn.BatchNorm2d(1)
        )
        self.fc = nn.Linear(64, out_dim)
    def forward(self, x):
        x = self.conv(x)
        bs, *_ = x.shape
        x = self.fc(x.view(bs,-1))
        return x

class MARL(nn.Module):
    def __init__(self, vqae, add_downstream=False, year_label_num=None, category_num=None):
        super(MARL, self).__init__()
        self.vqae = vqae
        self.use_downstream = add_downstream
        if add_downstream:
            self.height_infer = MetaInfer(1).to(device)
            self.age_infer = MetaInfer(year_label_num).to(device)
            self.category_infer = MetaInfer(category_num).to(device)

    def forward(self, x):
        if not self.use_downstream:
            return {'vqae': self.vqae(x)}
        
        # recon
        latent = self.vqae.encode(x)
        vq_loss, data_recon, perplexity = self.vqae.decode(latent)
        # downstream
        height_pred = self.height_infer(latent)
        age_pred = F.softmax(self.age_infer(latent), dim=1)
        category_pred = torch.sigmoid(self.category_infer(latent))

        return {
                'vqae': [vq_loss, data_recon, perplexity],
                'height': height_pred,
                'age': age_pred,
                'category': category_pred
               }
        
