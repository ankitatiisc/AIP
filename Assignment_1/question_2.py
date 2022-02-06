import os
import argparse
import yaml 
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader

from sklearn.metrics import classification_report, confusion_matrix

from nearest_neighbour import NearestNeighbor
from dataset_loader import ImagesDataLoader

from torchsummary import summary

import pdb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def freeze_weights(model):
    """Freeze weights of the model

    Args:
        model (nn.Module): A valid torch model
    """
    for param in model.parameters():
        param.requires_grad = False

class FineTunedModel(nn.Module):
    """Fine tuned CNN model
    """
    def __init__(self, feature_extractor):
        """Constructor

        Args:
            feature_extractor (nn.Module): feature extractor module
        """
        super(FineTunedModel,self).__init__()
        self.feature_extractor = feature_extractor
        freeze_weights(self.feature_extractor)
        self.linear = nn.Linear(4096, 6, bias=False)

    def forward(self, x):
        """forward method

        Args:
            x (tensor): input tensor

        Returns:
            [tensor]: output from the model
        """
        x = self.feature_extractor(x)
        return self.linear(x)

class SimpleModel(nn.Module):
    """Simple CNN model.
    """
    def __init__(self):
        super(SimpleModel,self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, bias=False)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, bias=False)
        self.conv3 = nn.Conv2d(32, 64, 3, bias=False)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=(1,1), bias=False)
        self.fc1 = nn.Linear(64 * 13 * 13, 36, bias=False)
        self.fc2 = nn.Linear(36, 6, bias=False)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        """forward method

        Args:
            x (tensor): input tensor

        Returns:
            [tensor]: output from the model
        """
        x = self.pool(self.leaky_relu(self.conv1(x)))
        #print(x.shape)
        x = self.pool(self.leaky_relu(self.conv2(x)))
        #print(x.shape)
        x = self.pool(self.leaky_relu(self.conv3(x)))
        x = self.pool(self.leaky_relu(self.conv4(x)))
        #print(x.shape)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x

def init_weights(m):
    """Initializes weights of the model.

    Args:
        m (layer): layer in the pytorch model.
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        #m.bias.data.fill_(0.01)
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)
        #m.bias.data.fill_(0.01)

def load_backbone(model_name:str, layer_num:int):
    """Loads backbone from the pretrained models

    Args:
        model_name (str): name of the pretrained model architecture
        layer_num (int): id of the layer from feature to be extracted
    Returns:
        [torch model]: features from the pre-trained module
    """
    if model_name == 'vgg16':
        m = models.vgg16(pretrained=True)
        m.classifier = torch.nn.Sequential(*(list(m.classifier.children())[:layer_num]))
        return m


def prepare_dataset_ML(cfg, backbone_model, categories_to_id, img_res,  dataset_type):
    """Prepares dataset for the ML module

    Args:
        cfg (dict): configuration file
        backbone_model (str): name of the backbone model
        categories_to_id (dict): categories to id map
        img_res (tuple): Input image resolution
        dataset_type (str): train/test
    Returns:
        [typle]: features, labels, image-paths
    """
    dataset = ImagesDataLoader(cfg[dataset_type], dataset_type, categories_to_id, img_res)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=1)
    tqdm_iterator = tqdm(data_loader, desc='Eval', total=len(data_loader))
    features = np.zeros((len(dataset), 4096), dtype=np.float32)
    labels = np.zeros((len(dataset)), dtype=np.int32)
    image_paths = []
    for step, batch in enumerate(tqdm_iterator):
        with torch.no_grad():
            batch_images = batch['img']
            batch_labels = batch['label']
            batch_features = backbone_model(batch_images)
            curr_batch_size = batch_features.shape[0]
            offset = step * 32 #hard-coded
            features[offset:offset+curr_batch_size] = batch_features.data.cpu().numpy()
            labels[offset:offset+curr_batch_size] = batch_labels.data.cpu().numpy()
            image_paths = image_paths + batch['img_path']
    return features, labels, image_paths

def find_wrong_predictions(gt_labels, pred_labels, test_img_paths, id_to_category):
    """Find wrong predictions. 

    Args:
        gt_labels (np.array): ground-truth labels
        pred_labels (np.array): predicted labels
        test_img_paths (list): test image paths
        id_to_category (dict): id to categories map
    """
    for i in range(gt_labels.shape[0]):
        if gt_labels[i] != pred_labels[i]:
            print('GT Label : {} , Predicted Label : {}'.format(id_to_category[gt_labels[i]], id_to_category[pred_labels[i]]))
            print(test_img_paths[i])

def train_NN(cfg, backbone_model, categoreies_to_id, id_to_categories, img_res):
    """Train nearest neighbor ML module.

    Args:
        cfg (dict):configuration file
        backbone (str): name of the backbone model
        categories_to_id (dict): categories to id map
        id_to_categories (dict): id to categories map
        img_res (tuple): image resolution
    """
    NN = NearestNeighbor()
    train_features, train_labels, train_img_paths = prepare_dataset_ML(cfg, backbone_model, categoreies_to_id, img_res, 'train')
    test_features, test_labels, test_img_paths = prepare_dataset_ML(cfg, backbone_model, categoreies_to_id, img_res, 'test')
    NN.train(train_features, train_labels)
    pred_test_labels = NN.predict(test_features)
    find_wrong_predictions(test_labels, pred_test_labels, test_img_paths, id_to_categories)
    target_names = [ id_to_categories[i] for i in range(len(categoreies_to_id.keys()))]
    
    print(confusion_matrix(test_labels, pred_test_labels))
    print(classification_report(test_labels, pred_test_labels, target_names=target_names))

def predict_from_CNN(cfg, model, categoreies_to_id, id_to_categories, img_res):
    """Predicts from a CNN module

    Args:
        cfg (dict): configuration file
        model (nn.Module):CNN model
        categories_to_id (dict): categories to id map
        id_to_categories (dict): id to categories map
        img_res (tuple): Input image resolution
    """
    #training dataset
    test_dataset = ImagesDataLoader(cfg['test'], 'test', categories_to_id, img_res)
    test_data_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=1)
    pred_ = np.zeros((len(test_dataset)), dtype=np.int32)
    gt_ = np.zeros((len(test_dataset)), dtype=np.int32)
    model.eval()
    #tqdm_iterator = tqdm(test_data_loader, desc='Train', total=len(test_data_loader))
    img_paths = []
    prev_index = 0
    for step, batch in enumerate(test_data_loader):
        images = batch['img'].to(device)
        curr_labels = batch['label']
        img_paths = img_paths + batch['img_path']
        output = model(images)
        _, curr_pred = torch.max(output,1)
        curr_batch_size = images.shape[0]
        #pdb.set_trace()
        pred_[prev_index:prev_index+curr_batch_size] = curr_pred.data.cpu().numpy()
        gt_[prev_index:prev_index+curr_batch_size] = curr_labels.data.cpu().numpy()
        prev_index = prev_index + curr_batch_size

    #pdb.set_trace()
    find_wrong_predictions(gt_, pred_, img_paths, id_to_categories)
    target_names = [ id_to_categories[i] for i in range(len(categoreies_to_id.keys()))]
    #print(target_names)
    print(confusion_matrix(gt_, pred_))
    print(classification_report(gt_, pred_, target_names=target_names))

def finetuneCNN(cfg, backbone, categories_to_id, id_to_categories):
    """Runs fine-tune CNN method

    Args:
       cfg (dict): configuration file
        backbone (str): name of the backbone model
        categories_to_id (dict): categories to id map
        id_to_categories (dict): id to categories map
    """
    backbone_model = load_backbone(backbone, layer_num = -1)
    finetuned_model = FineTunedModel(backbone_model)
    finetuned_model = nn.DataParallel(finetuned_model).to(device)
    img_res = (224, 224)

    #training dataset
    train_dataset = ImagesDataLoader(cfg['train'], 'train', categories_to_id, img_res)
    train_data_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    # set up optimizer
    optimizer = torch.optim.Adam(finetuned_model.parameters(), lr=0.0001)
    finetuned_model.train()

    criterion = nn.CrossEntropyLoss()

    for epoch in range(10):
        acc_loss = 0
        num_data = 0
        tqdm_iterator = tqdm(train_data_loader, desc='Train', total=len(train_data_loader))
        for step, batch in enumerate(tqdm_iterator):
            optimizer.zero_grad()
            images = batch['img'].to(device)
            labels = batch['label'].to(device)
            output = finetuned_model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            acc_loss = acc_loss + loss.item()
            num_data = num_data + batch['img'].shape[0]
        print('Epoch : {} Loss : {}'.format(epoch, acc_loss/num_data))
    
    predict_from_CNN(cfg, finetuned_model, categories_to_id, id_to_categories, img_res)

def simpleCNN(cfg, backbone, categories_to_id, id_to_categories):
    """Runs simple cnn method

    Args:
        cfg (dict): configuration file
        backbone (str): name of the backbone model
        categories_to_id (dict): categories to id map
        id_to_categories (dict): id to categories map
    """
    model = SimpleModel()
    model.apply(init_weights)
    print(summary(model.cuda(), (3,224,224)))
    model = nn.DataParallel(model).to(device)
    img_res = (224, 224)

    batch_size=32
    #training dataset
    train_dataset = ImagesDataLoader(cfg['train'], 'train', categories_to_id, img_res)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    
    # set up optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    model.train()

    criterion = nn.CrossEntropyLoss()

    for epoch in range(50):
        acc_loss = 0
        num_data = 0
        tqdm_iterator = tqdm(train_data_loader, desc='Train', total=len(train_data_loader))
        for step, batch in enumerate(tqdm_iterator):
            optimizer.zero_grad()
            images = batch['img'].to(device)
            labels = batch['label'].to(device)
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            acc_loss = acc_loss + loss.item()
            num_data = num_data + batch['img'].shape[0]
        print('Epoch : {} Loss : {}'.format(epoch, acc_loss/num_data))
    
    predict_from_CNN(cfg, model, categories_to_id, id_to_categories, img_res)
    torch.save(model.state_dict(), './simple_cnn_model.pth')

def get_id_to_categories(mapping):
    """Get id for the input categories

    Args:
        mapping (dict): mapping of the categories to id

    Returns:
        [dict]: id to categories mapping
    """
    reverse_mapping = {}
    for k in mapping.keys():
        reverse_mapping[int(mapping[k])] = k
    return reverse_mapping

def run_NN_method(cfg, backbone, categories_to_id, id_to_categories):
    """Run the nearest neighbour method. 

    Args:
        cfg (dict): configuration file
        backbone (str): name of the backbone model
        categories_to_id (dict): categories to id map
        id_to_categories (dict): id to categories map
    """
    backbone_model = load_backbone(backbone, layer_num = -3)
    backbone_model = nn.DataParallel(backbone_model).to(device)
    backbone_model.eval()
    img_res = (224, 224)
    train_NN(cfg, backbone_model, categories_to_id, id_to_categories, img_res)

if __name__ == '__main__':
    #Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg_file', type=str)
    args = parser.parse_args()

    #read yaml file
    with open(args.cfg_file,'r') as file:
        cfg= yaml.safe_load(file)

    categories_to_id = cfg['categories']
    id_to_categories = get_id_to_categories(categories_to_id)

    print('Nearest Neighbor')
    run_NN_method(cfg, 'vgg16', categories_to_id, id_to_categories)

    print('Fine-Tune')
    finetuneCNN(cfg, 'vgg16', categories_to_id, id_to_categories)

    print('Simple CNN')
    simpleCNN(cfg, 'vgg16', categories_to_id, id_to_categories)