#!/usr/bin/env python
# coding=utf-8
'''
Description: 
Author: yangyuxiang
Date: 2021-03-19 00:03:13
LastEditors: yangyuxiang
LastEditTime: 2021-04-17 22:53:31
FilePath: /Assignment2-3/data/transform_images.py
'''

import os
from os.path import dirname
import sys
import torch
from torch import device
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from torchvision import models, transforms

sys.path.append('..')
import config


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.resnet = models.resnet101(pretrained=False)
        self.resnet.load_state_dict(torch.load("./resnet101-5d3b4d8f.pth"))
        self.w = nn.Linear(2048, config.img_vec_dim)

    def forward(self, x):
        output = self.resnet.conv1(x)
        output = self.resnet.bn1(output)
        output = self.resnet.relu(output)
        output = self.resnet.maxpool(output)
        output = self.resnet.layer1(output)
        output = self.resnet.layer2(output)
        output = self.resnet.layer3(output)
        output = self.resnet.layer4(output)
        output = self.resnet.avgpool(output)
        output = self.w(output.permute(0, 3, 2, 1))
        output = torch.tanh(output)
        return output


def extract_vectors(img_path, net, device):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(1,1,1)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    img = Image.open(img_path)
    img = img.convert("RGB")
    img = transform(img)
#     print(img.shape)
    
    img = img.unsqueeze(0).float()
    img = img.to(device)
#     print(img)
    output = net(img)
    return output


if __name__ == "__main__":
    device = torch.device("cuda" if config.is_cuda else "cpu")
    model = Net()
    model.to(device)
    image_data = []
    for path, dirnames, filenames in os.walk(config.data_folder):
        for filename in filenames:
            if filename.endswith("jpg"):
                image_data.append(os.path.join(path, filename))

    print("has %s images." % len(image_data))
    with open(config.img_vecs, mode="w", encoding='utf-8') as f:
        for img_path in tqdm(image_data):
#             print(img_path)
            output = extract_vectors(img_path, model, device)
            output = output.cpu()
            output = output.reshape(-1)
            output = output.data.numpy()
#             print(output.shape)
            img_name = img_path.split("/")[-1]
            f.write(img_name + '\t')
            f.write(" ".join([str(v) for v in output.tolist()]))
            f.write("\n")
            
            
    




    