import torch
from PIL import Image
import numpy as np
from torchvision import transforms
import os

# Data utils
class Centeralize(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple): Desired output size.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int))
        self.output_size = output_size

    def __call__(self, image):
        tmp = image
        image = torch.sum(image, dim=0)/3
        h, w = image.shape
        half_h = half_w = int(self.output_size/2)
        grid_x = torch.FloatTensor([[i for i in range(0,w)] for j in range(0,h)])
        grid_y = torch.FloatTensor([[j for i in range(0,w)] for j in range(0,h)])
        img_reverse = (image<1).float()
        center_w = int(torch.sum(img_reverse*grid_x)/torch.sum(img_reverse))
        center_h = int(torch.sum(img_reverse*grid_y)/torch.sum(img_reverse))
        top = center_h - half_h
        left = center_w - half_w
        image = tmp[:, top: top + self.output_size,
                      left: left + self.output_size]
        return image

def preprocess_data(data_root):
    composed = transforms.Compose([
                                    Centeralize(1000),
                                    transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0)),
                                    transforms.Grayscale(1),
                                    transforms.Resize(112),
                                    transforms.CenterCrop(56),
                                ])
    def saveto(item, data_root, i):
        img = Image.open(item)
        img = np.array(img)/255.0
        img = np.transpose(img[:, :, :3], (2, 0, 1))
        img_tensor = torch.from_numpy(img.astype(np.float32))
        torch.save(composed(img_tensor), data_root + str(i) + '.pt')
    #read all data path
    all_file_names = os.listdir(data_root)
    all_data_dirs = []
    for name in all_file_names:
        if name.endswith(".png"):
            all_data_dirs.append(data_root + name)

    i=0
    for item in all_data_dirs:
        try:
            saveto(item, data_root, i)
        except:
            pass
        i+=1
        break

