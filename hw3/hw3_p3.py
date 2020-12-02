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
from torch.autograd import Function
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

testset = Allen(test_path, transform = transforms.ToTensor())
testloader = DataLoader(testset, batch_size = 64)

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class CNNModel(nn.Module):

    def __init__(self):
        super(CNNModel, self).__init__()

        self.feature = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  #64*28*28
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, 1, 1),  #128*28*28
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, 2, 1), #256*12*12
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, 3, 2, 1), #256*10*10
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 512, 3, 2, 1), #512*8*8
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.class_classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(512, 10),
        )

        self.domain_classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 1),
        )

    def forward(self, input_data, alpha):
        input_data = input_data.expand(input_data.data.shape[0], 3, 28, 28)
        feature = self.feature(input_data)
        feature = feature.view(-1, 512)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output

if target_domain == 'mnistm':
    load_path = 'p3_model/U_M_dann'
elif target_domain == 'svhn':
    load_path = 'p3_model/M_S_dann'
else:
    load_path = 'p3_model/S_U_dann'
model = CNNModel().to(device)

torch.manual_seed(70)

if use_cuda == True:
    model = torch.load(load_path)
else:
    model = torch.load(load_path, map_location=torch.device('cpu'))

model.eval()
with torch.no_grad():
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image_name', 'label'])
        for batch_idx, data in enumerate(testloader):
            im, name = data
            im = im.to(device)
            output, _ = model(im, 0)
            pred = output.max(1, keepdim = True)[1]
            pred = pred.cpu().numpy()
            for idx in range(len(pred)):
                ans = pred[idx][0]
                picture_name = name[idx]
                writer.writerow([picture_name, ans])
