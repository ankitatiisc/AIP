import os, sys
import argparse

import numpy as np

import torch
from torchvision import models, datasets
import torchvision.transforms as T

from torchsummary import summary
from PIL import Image

from voc_dataset import evaluate_metrics_voc2012, generate_visual_outputs
import cv2
import pdb

def get_fcn_resnet101():
    model = models.segmentation.fcn_resnet101(pretrained=True)
    torch.save(model, 'resnet50.pt')
    #model.load_state_dict(torch.load('./models/resnet101-5d3b4d8f.pth'))
    return model

class PreTrainedFCN:
    def __init__(self, device, backbone_type):
        self.device = device
        if backbone_type == 'resnet50':
            self.model = torch.load('./models/fcn_resnet50.pt').to(device).eval()
        elif backbone_type == 'resnet101':
            self.model = torch.load('./models/fcn_resnet101.pt').to(device).eval()
    
    def get_transformation(self):
        return T.Compose([T.Resize(512), T.ToTensor(), T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])]) 

    def decode_segmap(self, image, nc=21):
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
        rgb = np.stack([r, g, b], axis=2)
        return rgb

    def predict(self, img_path):
        tfn = self.get_transformation()
        inp = tfn(Image.open(img_path)).unsqueeze(0)
        out = self.model(inp.to(self.device))['out'] 
        om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
        decoded_img =   self.decode_segmap(om)
        cv2.imwrite('output.jpg', decoded_img)

if __name__ == '__main__':
    #Parse Arguments
    parser = argparse.ArgumentParser(description='FCN : Pretrained Network')
    parser.add_argument('--model_type', default='resnet50', type=str, help='resenet50/resnet101' )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fcn = PreTrainedFCN(device, args.model_type)
    fcn.predict('horse.jpg')
    evaluate_metrics_voc2012(fcn.model, device, is_pretrained=True)
    generate_visual_outputs(fcn.model, device, is_pretrained=True)