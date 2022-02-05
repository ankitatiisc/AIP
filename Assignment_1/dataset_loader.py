import os
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

import pdb 

def get_img_paths(folder, id):
    img_paths = []
    img_label = []
    for file_ in os.listdir(folder):
        if os.path.splitext(file_)[1] in ['.jpg']:
            img_paths.append(os.path.join(folder,file_))
            img_label.append(id)
    return img_paths, img_label
        
class ImagesDataLoader(Dataset):
    def __init__(self, folder, dataset_type, categories_to_id, img_res):
        super(ImagesDataLoader, self).__init__()  

        self.img_paths = []
        self.img_labels = []
        self.dataset_type = dataset_type
        for category in categories_to_id.keys():
           curr_img_paths, curr_img_labels = get_img_paths(os.path.join(folder,category+"_"+ dataset_type), categories_to_id[category])
           self.img_paths = self.img_paths + curr_img_paths
           self.img_labels = self.img_labels + curr_img_labels
        print('Loaded {} number of images ({})'.format(len(self.img_paths),dataset_type))

    def __len__(self):
        return len(self.img_paths)

    def get_rgb_img(self, img_path):
        input_image = Image.open(img_path)
        preprocess = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        img = preprocess(input_image)
        return img

    def __getitem__(self, index):
        item = {}
        item['img'] = self.get_rgb_img(self.img_paths[index])
        item['label'] = self.img_labels[index]
        return item