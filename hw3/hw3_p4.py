import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import csv
import sys

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

test_path = sys.argv[1]
target_domain = sys.argv[2]
output_path = sys.argv[3]

class Allen(Dataset): 
    def __init__(self, root, transform = None):
        self.transform = transform
        self.common_path = root
        self.filename = sorted(os.listdir(self.common_path))
        self.len = len(self.filename)

    def __getitem__(self, index):
        path = os.path.join(self.common_path, str(self.filename[index]))
        name = str(self.filename[index])
        im = Image.open(path)
        if self.transform is not None:
            im = self.transform(im)

        return im, name

    def __len__(self):

        return(self.len)

testset = Allen(test_path, transform = transforms.Compose([
    transforms.Resize((32,32)), # 隨機將圖片水平翻轉
    transforms.ToTensor(), # 將圖片轉成 Tensor，並把數值 normalize 到 [0,1] (data normalization)
    transforms.Normalize([0.0,],[1.0,])
]))
testloader = DataLoader(testset, batch_size = 64)

class source_encode(nn.Module):
    def __init__(self):
        super(source_encode, self).__init__()
        self.source_encoder_conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  #64*28*28
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, 1, 1),  #128*28*28
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, 1, 1), #256*12*12
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 256, 3, 1, 1), #256*10*10
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, 1, 1), #512*8*8
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
    def forward(self, x):
        feature = self.source_encoder_conv(x)
        feature = feature.view(-1, 512)
        return feature

class target_encode(nn.Module):
    def __init__(self):
        super(target_encode, self).__init__()
        self.target_encoder_conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  #64*28*28
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, 1, 1),  #128*28*28
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, 1, 1), #256*12*12
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 256, 3, 1, 1), #256*10*10
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, 1, 1), #512*8*8
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
    def forward(self, x):
        feature = self.target_encoder_conv(x)
        feature = feature.view(-1, 512)
        return feature

class shared_encode(nn.Module):
    def __init__(self):
        super(shared_encode, self).__init__()
        self.shared_encoder_conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  #64*28*28
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, 1, 1),  #128*28*28
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, 1, 1), #256*12*12
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 256, 3, 1, 1), #256*10*10
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, 1, 1), #512*8*8
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
    def forward(self, x):
        feature = self.shared_encoder_conv(x)
        feature = feature.view(-1, 512)
        return feature

class classification(nn.Module):
    def __init__(self):
        super(classification, self).__init__()
        self.shared_encoder_pred_class = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10),
        )
    def forward(self, x):
        x = self.shared_encoder_pred_class(x)
        return x

class domainclass(nn.Module):
    def __init__(self):
        super(domainclass, self).__init__()
        self.shared_encoder_pred_domain = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
    def forward(self, x):
        x = self.shared_encoder_pred_domain(x)
        return x

class decoder(nn.Module):
    def __init__(self, code_size=512):
        super(decoder, self).__init__()
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 3, 4, 2, 1),
        )
    def forward(self, x):
        x = self.layer(x)
        return x

if target_domain == 'mnistm':
    load_path_C = 'p4_model/U_M_C'
    load_path_shared = 'p4_model/U_M_shared'
elif target_domain == 'svhn':
    load_path_C = 'p4_model/M_S_C'
    load_path_shared = 'p4_model/M_S_shared'
else:
    load_path_C = 'p4_model/S_U_C'
    load_path_shared = 'p4_model/S_U_shared'

shared = shared_encode().to(device)
C = classification().to(device)

torch.manual_seed(70)

if use_cuda == True:
    C = torch.load(load_path_C)
    shared = torch.load(load_path_shared)
else:
    C = torch.load(load_path_C, map_location=torch.device('cpu'))
    shared = torch.load(load_path_shared, map_location=torch.device('cpu'))

C.eval()
shared.eval()
with torch.no_grad():
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image_name', 'label'])
        for batch_idx, data in enumerate(testloader):
            im, name = data
            im = im.to(device)
            im = im.expand(im.data.shape[0], 3, 32, 32)

            output = shared(im)
            output = C(output)

            pred = output.max(1, keepdim = True)[1]
            pred = pred.cpu().numpy()
            for idx in range(len(pred)):
                ans = pred[idx][0]
                picture_name = name[idx]
                writer.writerow([picture_name, ans])
