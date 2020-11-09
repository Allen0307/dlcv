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

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

test_path = sys.argv[1]
output_path = sys.argv[2]
class Allen(Dataset): 
    def __init__(self, transform = None):
        self.transform = transform
        self.common_path = test_path
        self.filename = sorted(os.listdir(self.common_path))
        self.len = len(self.filename)

    def __getitem__(self, index):
        path = os.path.join(self.common_path + str(self.filename[index]))
        im = Image.open(path)

        if self.transform is not None:
            im = self.transform(im)
        
        return im, str(self.filename[index])

    def __len__(self):

        return(self.len)

testset = Allen(transform = transforms.ToTensor())
testset_loader = DataLoader(testset, batch_size = 128)

class Net(nn.Module):
    def __init__(self, model):
        super(Net, self).__init__()
        self.layer1 = model
        self.fc1 = nn.Sequential(
            nn.Linear(1000, 256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc2 = nn.Linear(256, 50)
    
    def forward(self, x):
      x = self.layer1(x)
      x = self.fc1(x)
      x = self.fc2(x)
      return x
backbond = torchvision.models.vgg16_bn(pretrained = True)
model = Net(backbond).to(device)

if use_cuda == True:
    model = torch.load('p1_vgg16_bn')
else:
    model = torch.load('p1_vgg16_bn', map_location=torch.device('cpu'))

model.eval()

with torch.no_grad():
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image_id', 'label'])
        for batch_id, data in enumerate(testset_loader):
            im, name = data
            im = im.to(device)
            output = model(im)
            pred = output.max(1, keepdim = True)[1]
            pred = pred.cpu().numpy()
            for idx in range(len(pred)):
                ans = pred[idx][0]
                picture_name = name[idx]
                writer.writerow([picture_name, ans])
        

