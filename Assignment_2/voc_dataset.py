import os
import math

import numpy as np
import cv2
from tqdm import tqdm

from PIL import Image

import torch
from torch import nn
from torchvision import datasets, io
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from focal_loss import FocalLoss

import pdb 

VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]
            
VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

def voc_colormap2label():
    colormap2label = torch.zeros(256**3, dtype=torch.long)
    for index, color in enumerate(VOC_COLORMAP):
        hash_index = color[2] + 256 * (color[1] + 256 * color[0])
        colormap2label[hash_index] = index
    return colormap2label

def colormap2indices(mask, lut):
    #pdb.set_trace()
    idx = ((mask[0,:,:] * 256 + mask[1,:,:]) * 256 + mask[2,:,:])
    return lut[idx]

def label2colormap(mask):
    mask = mask.numpy()
    img = np.zeros( (mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            color = VOC_COLORMAP[mask[y,x]]
            img[y,x,0] = color[2]
            img[y,x,1] = color[1]
            img[y,x,2] = color[0]
    return img

def evaluate_metrics(list_true_label, list_pred_label, num_classes):
    #Input is of list type
    #From paper : nij be the number of pixels of class i predicted to belong to class j
    data_ = np.zeros((num_classes, num_classes))
    for true_label, pred_label in tqdm(zip(list_true_label, list_pred_label),desc='Evaluation', total=len(list_true_label)):
        for i in range(num_classes):
            mask = pred_label[true_label == i]
            for elem in mask:
                data_[i][elem] += 1

    pixel_accuracy = np.diag(data_).sum()/data_.sum()
    acc_per_cls = np.diag(data_) / data_.sum(axis=1)
    iu = np.diag(data_) / (data_.sum(axis=1) + data_.sum(axis=0) - np.diag(data_))
    #freq_weighted = np.multiply(data_.sum(axis=1) , iu)
    #for i in range(num_classes):
        #pixel_accuracy[i] = data_[i,i]/(np.sum(data_[i,:]))
    #mean_accuracy = np.mean(pixel_accuracy)
    freq = data_.sum(axis=1) / data_.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return pixel_accuracy, np.nanmean(acc_per_cls), np.nanmean(iu),  fwavacc
       

class VOCSegmentationDataset(Dataset):
    def __init__(self, is_train=True, voc_dir='./voc_data/VOCdevkit/VOC2012/', aug=True, is_finetuned=False, is_test=False):
        self.data_set = self.get_file_names(os.path.join(voc_dir, 'ImageSets', 'Segmentation', 'train.txt' if is_train else 'val.txt'))
        if is_test:
            self.data_set = self.get_file_names('test.txt')
        #self.data_set = self.data_set[0:40]
        self.voc_dir = voc_dir
        self.is_train = is_train
        self.aug = aug
        self.lut = voc_colormap2label()
        self.is_finetuned = is_finetuned

    def get_file_names(self, txt_fname):
        with open(txt_fname, 'r') as f:
            images = f.read().split()
            return images

    def transform(self, image, mask, aug):
        if not self.is_train and self.is_finetuned:
            #Add Padding
            p_w = 0
            p_h = 0
            if image.size[0] % 32 != 0:
                p_w = math.ceil(image.size[0]/32) * 32 - image.size[0]
            if image.size[1] % 32 != 0:
                p_h = math.ceil(image.size[1]/32) * 32 - image.size[1]
 
            padding_sequence = [p_w//2, p_h//2, p_w//2, p_h//2]
            if p_w % 2 != 0:
                padding_sequence[2] = padding_sequence[2] + 1

            if p_h % 2 != 0:
                padding_sequence[3] = padding_sequence[3] + 1

            padding_sequence = tuple(padding_sequence)
            #print(padding_sequence)
            #pdb.set_trace()
            image = T.functional.pad(image,padding_sequence)
            mask = T.functional.pad(mask,padding_sequence)

        if aug:
            #Add Padding
            p_w = max(0,320 - image.size[0])
            p_h = max(0,480 - image.size[1])
            padding_sequence = [p_w//2, p_h//2, p_w//2, p_h//2]
            if p_w % 2 != 0:
                padding_sequence[2] = padding_sequence[2] + 1
            
            if p_h % 2 != 0:
                padding_sequence[3] = padding_sequence[3] + 1

            padding_sequence = tuple(padding_sequence)
            #print(padding_sequence)
            #pdb.set_trace()
            image = T.functional.pad(image,padding_sequence)
            mask = T.functional.pad(mask,padding_sequence)

            # Random crop
            #pdb.set_trace()
            i, j, h, w = T.RandomCrop.get_params(image, output_size=(480, 320))
            
            image = T.functional.crop(image, i, j, h, w)
            mask = T.functional.crop(mask, i, j, h, w)

        #convert image to tensor
        image = T.functional.to_tensor(image)
        mask = T.functional.to_tensor(mask)*255.
        mask = mask.long()
        
        #normalize the image
        transform = T.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image = transform(image)
        return image, mask

    def __getitem__(self, idx):
        #Read Image
        fname = self.data_set[idx]
        #pdb.set_trace()
        img = Image.open(os.path.join(self.voc_dir, 'JPEGImages', f'{fname}.jpg'))
        label = Image.open(os.path.join(self.voc_dir, 'SegmentationClass' ,f'{fname}.png')).convert('RGB')

        img, label = self.transform(img, label, self.aug)

        label = colormap2indices(label, self.lut)
        return img, label

    def __len__(self):
        return len(self.data_set)
        
def visualize(voc_seg_datset, index):
    img, label = voc_seg_datset[index]
    
    img = img.detach().cpu().numpy()
    img = np.transpose(img, (1,2,0) )
    img[:,:,0] = (( img[:,:,0] + (0.485/0.229) ) * 0.229)*255
    img[:,:,1] = (( img[:,:,1] + (0.456/0.224) ) * 0.224)*255
    img[:,:,2] = (( img[:,:,2] + (0.406/0.225) ) * 0.225)*255
    img = img[:,:,::-1].astype(np.uint8)
    label = label2colormap(label)#np.asarray(label)
    #label = np.transpose(label, (1,2,0) ) 
    #label = label[:,:,::-1].astype(np.uint8)

    #pdb.set_trace()
    padding = np.zeros((img.shape[0],10,3),dtype=np.uint8)
    padding[:,:] = [255,255,255]
    img = np.concatenate((img,padding),axis=1) 
    img = np.concatenate((img,label),axis=1)
    cv2.imwrite('./dataset_sanity_check/image_'+str(index)+'.jpg', img)

def visualize_dataset(voc_seg_datset, n=5):
    for i in range(n):
        print(i)
        visualize(voc_seg_datset, i)

def test_evaluate_metrics():
    true_labels = np.random.randint(low=0, high=4, size=(4,4))
    pred_labels = np.random.randint(low=0, high=4, size=(4,4))
    print(evaluate_metrics(true_labels, pred_labels, 4))

def evaluate_metrics_voc2012(model, device, is_pretrained=False):
    #model.eval()
    if is_pretrained:
        val_dataset = VOCSegmentationDataset(is_train=False, aug=False, is_finetuned=False)
    else:
        val_dataset = VOCSegmentationDataset(is_train=False, aug=False, is_finetuned=True)
    val_data_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)
    
    tqdm_iterator = tqdm(val_data_loader, desc='Test', total=len(val_data_loader))
    list_true = []
    list_pred = []
    val_acc = 0
    num_data_points = 0
    for step, batch in enumerate(tqdm_iterator):
        images = batch[0].to(device)
        labels = batch[1].to(device)
        if is_pretrained:
            output = model(images)['out'] 
        else:
            output = model(images)
       
        num_data_points += 1
        labels = labels.squeeze().detach().cpu().numpy()
        output = torch.argmax(output.squeeze(), dim=0).detach().cpu().numpy()
        val_acc += ((output==labels).sum())
        num_data_points += (labels.shape[0] * labels.shape[1])
        list_true.append(labels)
        list_pred.append(output)

        #if step % 10 == 0:
            #break
       
    print('Average Accuracy is ', val_acc/num_data_points)
    print(evaluate_metrics(list_true, list_pred, 21))

def decode_segmap( image, nc=21):
    label_colors = np.array([(0, 0, 0),  # 0=background
                # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]
    rgb = np.stack([b, g, r], axis=2)
    return rgb

def generate_visual_outputs(model, device, is_pretrained=False):
    #model.eval()
    if is_pretrained:
        val_dataset = VOCSegmentationDataset(is_train=False, aug=False, is_finetuned=False, is_test=True)
    else:
        val_dataset = VOCSegmentationDataset(is_train=False, aug=False, is_finetuned=True, is_test=True)
    val_data_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)
    
    tqdm_iterator = tqdm(val_data_loader, desc='Test Visual', total=len(val_data_loader))

    for step, batch in enumerate(tqdm_iterator):
        images = batch[0].to(device)
        labels = batch[1].to(device)
        visualize(val_dataset, step)
        if is_pretrained:
            output = model(images)['out'] 
        else:
            output = model(images)
       
        labels = labels.squeeze().detach().cpu().numpy()
        output = torch.argmax(output.squeeze(), dim=0).detach().cpu().numpy()
       
        out_img = decode_segmap(output)
        cv2.imwrite('./visual_output/image_'+str(step)+'.png',out_img)
        #pdb.set_trace()

if __name__ == '__main__':
    voc_seg = VOCSegmentationDataset(is_train=True)
    #visualize(voc_seg, 18)
    #visualize_dataset(voc_seg, n=len(voc_seg))
    test_evaluate_metrics()

