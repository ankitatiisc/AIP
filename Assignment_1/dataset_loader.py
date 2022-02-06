import os
from PIL import Image
import pandas as pd

import torch
from torch.utils.data import Dataset
from torchvision import transforms

import pdb 

     
class ImagesDataLoader(Dataset):
    """Image Data Loader
    """
    def __init__(self, csv_file, dataset_type, categories_to_id, img_res):
        """Constructor

        Args:
            csv_file (str): csv file
            dataset_type (str): type of dataset (train/test)
            categories_to_id (dict): categories to id mapping
            img_res (tuple): image resolution
        """
        super(ImagesDataLoader, self).__init__()  
        self.data_set = pd.read_csv(csv_file)
        #pdb.set_trace()
        self.dataset_type = dataset_type
        print('Loaded {} number of images ({})'.format(len(self.data_set), dataset_type))

    def __len__(self):
        """Length of the dataset.
        """
        return len(self.data_set)

    def get_rgb_img(self, img_path):
        """Get RGB Image

        Args:
            img_path (str): path of the input image

        Returns:
            [tensor]: processed image tensor
        """
        input_image = Image.open(img_path)
        preprocess = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        img = preprocess(input_image)
        return img

    def __getitem__(self, index):
        """Gets item given an index.

        Args:
            index (int): index

        Returns:
            [dict]: dataset given an index.
        """
        item = {}
        item['img'] = self.get_rgb_img(self.data_set.iloc[index]['Image Paths'])
        item['label'] = int(self.data_set.iloc[index]['Label'])
        item['img_path'] = self.data_set.iloc[index]['Image Paths']
        return item