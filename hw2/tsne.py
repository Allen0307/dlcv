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
import sys
import matplotlib.pyplot as plt
import csv
from sklearn.manifold import TSNE

use_cuda = torch.cuda.is_available()
torch.manual_seed(123)
device = torch.device("cuda" if use_cuda else "cpu")

test_path = 'hw2_data/p1_data/val_50/'
# output_path = sys.argv[2]
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
    #   x = self.fc1(x)
    #   x = self.fc2(x)
      return x
backbond = torchvision.models.vgg16_bn(pretrained = True)
model = Net(backbond).to(device)

if use_cuda == True:
    model = torch.load('p1_vgg16_bn')
else:
    model = torch.load('p1_vgg16_bn', map_location=torch.device('cpu'))

model.eval()

with torch.no_grad():
    for batch_id, data in enumerate(testset_loader):
        im, name = data
        im = im.to(device)
        output = model(im)
        if batch_id == 0:
            ans = output
        else:
            ans = np.concatenate((ans, output), axis=0)

print(np.size(ans,0))
print(np.size(ans,1))

X_embedded = TSNE(n_components=2).fit_transform(ans)
print(X_embedded.shape)
plt.scatter(X_embedded[:,0],X_embedded[:,1])
plt.show()