from torchvision import transforms
import numpy as np
import torch
from utils import add_noise
import os
from PIL import Image
from utils import *
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


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
    def __init__(self, root='../data/data_root/', subset=None, 
                 data_config='../data/data_config/', 
                 add_noise=False, multi_scale=False, 
                 preprocess=False):
        
        self.data_root = root
        self.data_config = data_config
        self.subset = subset
        self.add_noise = add_noise
        self.multi_scale = multi_scale
        self.preprocess = preprocess
        self._init_config()
        self._init_data_info()
        self.var = self.data_variance()

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
                                transforms.Grayscale(1),
                                transforms.CenterCrop(700)])
            self.composed_1 = transforms.Compose([
                                transforms.Resize(112)])
            self.composed_2 = transforms.Compose([
                                transforms.CenterCrop(224),
                                transforms.Resize(112)])
            self.composed_3 = transforms.Compose([
                                transforms.CenterCrop(112)])

    def _init_data_info(self):
        all_file_names = os.listdir(self.data_root)
        self.meta_info = pd.read_csv(os.path.join(self.data_config, 'meta.csv'), index_col='OBJECTID')
        self.height_info = pd.read_csv(os.path.join(self.data_config, 'height.csv'), index_col='OBJECTID')
        self.meta_info['AgeLabel'] = LabelEncoder().fit_transform(self.meta_info['YearBuilt1'])
        self.meta_info['CateOneHot'] = OneHotEncoder().fit_transform(self.meta_info.UseDescription.values.reshape(-1,1)).toarray().tolist()

        self.all_data_dirs = []
        self.all_building_idx = []
        if self.subset is not None:
            subset_idx = list(pd.read_excel(self.subset).to_numpy().flatten())
            self.all_data_dirs = [self.data_root+str(idx)+('.pt' if self.preprocess else '.png') for idx in subset_idx]
            return
        for name in all_file_names:
            if name.endswith(".png" if not self.preprocess else ".pt"):
                self.all_data_dirs.append(self.data_root + name)
                self.all_building_idx.append(int(name[:-3]))

    def data_variance(self):
        if not self.preprocess:
            value = np.var(np.array([self.preload(i).numpy() for i in range(0,self.__len__())]))
            self.preprocess = True
            torch.save(value, os.path.join(self.data_root, 'var_pt'))
        value = torch.load(os.path.join(self.data_root, 'var_pt'))
        return value
    
    def preload(self, index):
        data_dict = self[index]
        data = data_dict['image_tensor']
        invalid_check = (data_dict['year_built']!=0)
        if not self.preprocess and invalid_check:
            self.all_data_dirs[index] = self.all_data_dirs[index][:-4]+'.pt'
            torch.save(data, self.all_data_dirs[index])
        return data
    
    def __len__(self):
        return len(self.all_data_dirs)
    
    def __getitem__(self, index):
        # meta info loading
        obj_idx = self.all_building_idx[index]
        meta_info = self.meta_info.loc[[obj_idx]]
        height = self.height_info.at[obj_idx, 'HEIGHT_norm']

        year_built = meta_info.at[obj_idx, 'YearBuilt1']
        category = meta_info.at[obj_idx, 'UseDescription']
        age_label = meta_info.at[obj_idx, 'AgeLabel']
        cate_onehot = meta_info.at[obj_idx, 'CateOneHot']
  

        # image loading
        if self.preprocess:
            img_tensor = torch.load(self.all_data_dirs[index])
        else:
            img = Image.open(self.all_data_dirs[index])
            if self.add_noise:
                img = self.trancolor(img)
            img = np.array(img)/255.0
            img = np.transpose(img[:, :, :3], (2, 0, 1))
            img_tensor = torch.from_numpy(img.astype(np.float32))
            if not self.multi_scale:
                img_tensor = self.composed(img_tensor)
            else:
                img_tensor = self.composed_0(img_tensor)
                channel_1 = self.composed_1(img_tensor)
                channel_2 = self.composed_2(img_tensor)
                channel_3 = self.composed_3(img_tensor)
                img_tensor = torch.cat([channel_1,channel_2,channel_3], dim=0)
        return {
                    'image_tensor': img_tensor,
                    'year_built': year_built,
                    'age_label': age_label,
                    'height': height,
                    'category': category,
                    'cate_onehot': cate_onehot
               }
        




        

    
