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
import matplotlib.pyplot as plt
from PIL import Image
import scipy.misc

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

test_path = sys.argv[1]
output_path = sys.argv[2]

class Allen(Dataset): 
    def __init__(self, transform = None):
        self.transform = transform
        self.common_path = test_path
        self.filename = [file for file in os.listdir(self.common_path) if file.endswith('.jpg')]
        self.filename.sort()
        self.len = len(self.filename)

    def __getitem__(self, index):

        sat_path = os.path.join(self.common_path, str(self.filename[index]))
        sat_im = Image.open(sat_path)

        if self.transform is not None:
            sat_im = self.transform(sat_im)

        return sat_im, str(self.filename[index][0:4])

    def __len__(self):

        return(self.len)

testset = Allen(transform = transforms.ToTensor())
testset_loader = DataLoader(testset, batch_size = 8, shuffle = False)

cls_color = {
    0:  [0, 255, 255],
    1:  [255, 255, 0],
    2:  [255, 0, 255],
    3:  [0, 255, 0],
    4:  [0, 0, 255],
    5:  [255, 255, 255],
    6: [0, 0, 0],
}

def visual(A):
    #將512*512的nparray 轉成512*512*3
    A = list(A.reshape(512*512))

    for index in range(512*512):

        entry = np.array(cls_color[int(A[index])])

        A[index] = entry

    A = np.array(A)
    A = A.reshape((512,512,3))
    return A

class Net(nn.Module):
    def __init__(self, model):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(*list(model.children())[:-2][0][:-14])       
        self.layer2 = nn.Sequential(*list(model.children())[:-2][0][17:24])       
        self.layer3 = nn.Sequential(*list(model.children())[:-2][0][24:])
        self.x2 = nn.Sequential(
            nn.ConvTranspose2d(512,256,4,2,1),
            nn.ReLU(inplace=True)
        )
        self.x4 = nn.Sequential(
            nn.ConvTranspose2d(512,256,8,4,2),
            nn.ReLU(inplace=True)            
        )
        self.fcn8 = nn.Sequential(
            nn.Conv2d(256, 4096, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(4096,4096,1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(4096,7,1),
            nn.ReLU(),
            nn.ConvTranspose2d(7,7,16,8,4)
        )

    def forward(self, x):
      x = self.layer1(x)
      y = x
      x = self.layer2(x)
      z = x
      x = self.layer3(x)
      x = self.x4(x) + self.x2(z) + y
      x = self.fcn8(x)
      return x

backbond = torchvision.models.vgg16(pretrained = True)
model = Net(backbond).to(device)

if use_cuda == True:
    model = torch.load('p2_improve')
else:
    model = torch.load('p2_improve', map_location=torch.device('cpu'))

model.eval()
with torch.no_grad():
    for batch_id, data in enumerate(testset_loader):
        im, name = data
        im = im.to(device)
        output = model(im)
        pred = output.max(1, keepdim = True)[1]
        pred = pred.cpu().numpy().reshape((np.size(pred,0),512,512))
        if batch_id == 0:
            ans = pred
        else:
            ans = np.concatenate((ans, pred), axis=0)

namelist = []
for i in range(len(testset)):
  namelist.append(testset.__getitem__(i)[1])

for i in range(len(testset)):
    pngname = namelist[i] + '_mask.png'
    photo = visual(ans[i])
    target_path = os.path.join(output_path, pngname)
    scipy.misc.imsave(target_path, photo)