import torchvision
from torchvision import transforms
import numpy as np
import torch
from utils import add_noise
import os
from PIL import Image
from utils import *

# Demo Data
def load_CIFAR10():
    training_data = torchvision.datasets.CIFAR10(root="../data", train=True, download=False,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
                                    ]))

    validation_data = torchvision.datasets.CIFAR10(root="../data", train=False, download=False,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
                                    ]))

    data_variance = np.var(training_data.data / 255.0)

    return training_data, validation_data, data_variance

def load_FloorPlan(multi_scale=False):
    #Load Dataset
    floor = FloorPlanDataset(multi_scale=multi_scale)

    val_len = int(len(floor)/10)
    train_set, val_set = torch.utils.data.random_split(floor, [len(floor)-val_len, val_len])
    data_variance = get_data_variance(train_set)

    print(f"data shape: {floor[0].shape}, dataset size: {len(floor)}, data variance: {data_variance}")
    return data_variance, train_set, val_set, val_len

# Floor Plan
class FloorPlanDataset(torch.utils.data.Dataset):
    def __init__(self, root='../data/floorplan_crop/', subset=None, 
                 add_noise=False, multi_scale=False, preprocess=False):
        self.data_root = root
        self.subset = subset
        self.add_noise = add_noise
        self.multi_scale = multi_scale
        self.preprocess = preprocess
        self._init_config()
        self._init_data_info()

    def _init_config(self):
        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.composed = transforms.Compose([
                                            transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0)),
                                            transforms.Grayscale(1),
                                            transforms.Resize(112),
                                            transforms.CenterCrop(56),
                                          ])
        if self.multi_scale:
            self.composed_0 = transforms.Compose([
                                            transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0)),
                                            transforms.Grayscale(1)])
            self.composed_1 = transforms.Compose([
                                            transforms.Resize(56)])
            self.composed_2 = transforms.Compose([
                                            transforms.CenterCrop(112),
                                            transforms.Resize(56)])
            self.composed_3 = transforms.Compose([
                                            transforms.CenterCrop(56)])

    def _init_data_info(self):
        self.all_data_dirs = os.listdir(self.data_root)
        newlist = []
        subset_idx = None
        if self.subset is not None:
            import pandas as pd
            subset_idx = list(pd.read_excel(self.subset).to_numpy().flatten())

        for names in self.all_data_dirs:
            if names.endswith(".png" if not self.preprocess else ".pt"):
                if subset_idx is not None and not int(names[:-3]) in subset_idx:
                    continue
                newlist.append(names)
        self.all_data_dirs = [self.data_root + name for name in newlist]

        
    def data_variance(self):
        value =  np.var(np.array([self.preload(i).numpy() for i in range(0,self.__len__())]))
        self.preprocess = True
        return value

    def __len__(self):
        return len(self.all_data_dirs)
    
    def __getitem__(self, index):
        if self.preprocess:
            return torch.load(self.all_data_dirs[index])
        img = Image.open(self.all_data_dirs[index])
        if self.add_noise:
            img = self.trancolor(img)
        img = np.array(img)/255.0
        img = np.transpose(img[:, :, :3], (2, 0, 1))
        img_tensor = torch.from_numpy(img.astype(np.float32))
        if not self.multi_scale:
            return self.composed(img_tensor)
        else:
            img_tensor = self.composed_0(img_tensor)
            channel_1 = self.composed_1(img_tensor)
            channel_2 = self.composed_2(img_tensor)
            channel_3 = self.composed_3(img_tensor)
            
            return torch.cat([channel_1,channel_2,channel_3], dim=0)
        
    def preload(self, index):
        data = self[index]
        if not self.preprocess:
            torch.save(data, self.data_root + str(index) + '.pt')
        return data

    



        

    
