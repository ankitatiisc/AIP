import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm

from voc_dataset import VOCSegmentationDataset
from voc_dataset import evaluate_metrics_voc2012, generate_visual_outputs
from focal_loss import FocalLoss

import pdb 
import argparse

from matplotlib import pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def freeze_weights(model):
    """Freeze weights of the model

    Args:
        model (nn.Module): A valid torch model
    """
    for param in model.parameters():
        param.requires_grad = False

def xavier_initialization(layer):
    nn.init.xavier_normal_(layer.weight)

class FCNResNet18(nn.Module):
    def __init__(self):
        super(FCNResNet18,self).__init__()
        #Resnet has adaptive-average and linear layer at the end. removing them for the backbone.
        self.backbone = nn.Sequential( *list((torchvision.models.resnet18(pretrained=True)).children())[0:-2] )
        freeze_weights(self.backbone) #Freeze the weights of the backbone

        #create 1x1 kernel
        self.final_conv = nn.Conv2d(512, 21, kernel_size=1, bias=False) #output channels from resnet 512 and 32 times spatial reduction
        self.upsampling = nn.ConvTranspose2d(21, 21, kernel_size=64, padding=16, stride=32, bias=False) 
        xavier_initialization(self.final_conv)
        xavier_initialization(self.upsampling)

    def forward(self, x):
        x = self.backbone(x)
        x = self.final_conv(x)
        return self.upsampling(x)

def loss_criterion(inputs, targets):
    return F.cross_entropy(inputs, targets, reduction='mean')

def train_FCNResNet18(num_epochs=100, batch_size=32, learning_rate=1e-4):
    fcn_resnet18 = nn.DataParallel(FCNResNet18()).to(device)
    voc_seg = VOCSegmentationDataset(is_train=True)
    
    train_dataset = VOCSegmentationDataset(is_train=True)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    
    val_dataset = VOCSegmentationDataset(is_train=False, aug=False, is_finetuned=True)
    val_data_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)

    optimizer = torch.optim.Adam(fcn_resnet18.parameters(), lr=learning_rate)
    fcn_resnet18.train()

    criterion = FocalLoss(gamma = 3.0)
    best_val_acc_model = 0
    acc_val = 0
    list_train_loss = []
    list_val_loss = []
    list_val_accuracy = []
    for epoch in range(num_epochs):
        fcn_resnet18.train()
        train_loss = 0
        val_loss = 0
        num_data = 0
        tqdm_iterator = tqdm(train_data_loader, desc='Train', total=len(train_data_loader))
        for step, batch in enumerate(tqdm_iterator):
            optimizer.zero_grad()
            images = batch[0].to(device)
            labels = batch[1].to(device)
            output = fcn_resnet18(images)
            loss = criterion(output, labels)
            #pdb.set_trace()
            loss.backward()
            optimizer.step()
            train_loss = train_loss + loss.item()*batch[0].shape[0]
            num_data = num_data + batch[0].shape[0]
        
        tqdm_iterator = tqdm(val_data_loader, desc='Val', total=len(val_data_loader))
        fcn_resnet18.eval()
        val_loss = 0
        val_acc = 0
        num_val = 0
        num_val_acc = 0
        for step, batch in enumerate(tqdm_iterator):
            images = batch[0].to(device)
            labels = batch[1].to(device)
            output = fcn_resnet18(images)
            loss = criterion(output, labels)
            val_loss = val_loss + loss.item()*batch[0].shape[0]
            num_val = num_val + batch[0].shape[0]
            val_acc += ((torch.argmax(output.squeeze(),dim=0)==labels.squeeze()).sum())
            #pdb.set_trace()
            num_val_acc += (labels.shape[0] * labels.shape[1] * labels.shape[2])

        average_val_accuracy = (val_acc * 1.)/num_val_acc
        avg_train_loss = train_loss/num_data
        avg_val_loss = val_loss/num_val
        list_train_loss.append(avg_train_loss)
        list_val_loss.append(avg_val_loss)
        list_val_accuracy.append(average_val_accuracy)
        #print(val_acc)
        print('Epoch : {} Train Loss : {} Val Loss : {} Val Accuracy : {}'.format(epoch, avg_train_loss, avg_val_loss, average_val_accuracy))

        if epoch % 10 == 0:
            generate_visual_outputs(fcn_resnet18, device, is_pretrained=False)
        if average_val_accuracy > best_val_acc_model:
            best_val_acc_model = average_val_accuracy
            torch.save(fcn_resnet18.state_dict(), './trained_models/my_fcn_model_' + str(epoch) +'.pth')

    torch.save(fcn_resnet18.state_dict(), './trained_models/my_fcn_model_final.pth')

    epochs = [i for i in range(num_epochs)]
    plt.plot(epochs,list_train_loss,color='g',label='Train Loss')
    plt.plot(epochs,list_val_loss,color='b',label='Val Loss')
    plt.legend()
    plt.title('Training Statistics')
    plt.xlabel('epochs')
    plt.ylabel('Value')
    plt.savefig('training_plot.png')
    plt.close()

    plt.plot(epochs,list_val_accuracy,color='r',label='Val Accuracy')
    plt.legend()
    plt.title('Training Statistics')
    plt.xlabel('epochs')
    plt.ylabel('Value')
    plt.savefig('training_plot_accuracy.png')

    fcn_resnet18.eval()
    evaluate_metrics_voc2012(fcn_resnet18, device, is_pretrained=False)
    generate_visual_outputs(fcn_resnet18, device, is_pretrained=False)

def evaluate_model(model_path):
    fcn_resnet18 = FCNResNet18()
    fcn_resnet18.load_state_dict(torch.load(model_path), strict=False)
    fcn_resnet18 = fcn_resnet18.to(device)
    fcn_resnet18.eval()
    #fcn_resnet18 =  nn.DataParallel(fcn_resnet18).to(device)
    evaluate_metrics_voc2012(fcn_resnet18, device, is_pretrained=False)
    generate_visual_outputs(fcn_resnet18, device, is_pretrained=False)

if __name__ == '__main__':
    #Parse Arguments
    parser = argparse.ArgumentParser(description='FCN : Finetuning with pretrained backbone')
    parser.add_argument('--train', default=0, type=int, help='if 1 then trains the FCN with resnet18 backbone' )
    parser.add_argument('--test', type=str, help='if path of the model is provided then it tests the perfomance on VOC dataset.' )
    args = parser.parse_args()

    if args.train:
        train_FCNResNet18()
    
    if args.test:
        evaluate_model(args.test)