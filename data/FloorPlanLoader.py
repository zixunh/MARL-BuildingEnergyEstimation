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
    training_data = torchvision.datasets.CIFAR10(root="data", train=True, download=False,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
                                    ]))

    validation_data = torchvision.datasets.CIFAR10(root="data", train=False, download=False,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
                                    ]))

    data_variance = np.var(training_data.data / 255.0)

    return training_data, validation_data, data_variance

def load_FloorPlan():
    #Load Dataset
    floor = FloorPlanDataset()

    val_len = int(len(floor)/10)
    train_set, val_set = torch.utils.data.random_split(floor, [len(floor)-val_len, val_len])
    data_variance = get_data_variance(train_set)

    print(f"data shape: {floor[0].shape}, dataset size: {len(floor)}, data variance: {data_variance}")
    return data_variance, train_set, val_set, val_len

# Floor Plan
class FloorPlanDataset(torch.utils.data.Dataset):
    def __init__(self, root='data/floorplan_crop/', add_noise=False):
        self.data_root = root
        self.add_noise = add_noise
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

    def _init_data_info(self):
        self.all_data_dirs = os.listdir(self.data_root)
        self.all_data_dirs = [self.data_root + str(i) + '.png' for i in range(0,self.__len__())]
        
    def data_variance(self):
        return np.var(np.array([self[i] for i in range(0,self.__len__())])/255.0)

    def __len__(self):
        return len(self.all_data_dirs)
    
    def __getitem__(self, index):
        img = Image.open(self.all_data_dirs[index])
        if self.add_noise:
            img = self.trancolor(img)
        img = np.array(img)/255.0
        img = np.transpose(img[:, :, :3], (2, 0, 1))
        return self.composed(torch.from_numpy(img.astype(np.float32)))
    



        

    
