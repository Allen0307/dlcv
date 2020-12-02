import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.utils as vutils
import numpy as np
from PIL import Image
from scipy import misc
import sys

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(168)

output_path = sys.argv[1]

# Generator
class Generator(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(inputSize, hiddenSize*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(hiddenSize*8),
            nn.ReLU(True),

            nn.ConvTranspose2d(hiddenSize*8, hiddenSize*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hiddenSize*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(hiddenSize*4, hiddenSize*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hiddenSize*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(hiddenSize*2, hiddenSize, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hiddenSize),
            nn.ReLU(True),

            nn.ConvTranspose2d(hiddenSize, outputSize, 4, 2, 1, bias=False),
            nn.Tanh())

    def forward(self, input):
        return self.main(input)

neyG = Generator(100,64,3)

if use_cuda == True:
    netG = torch.load('p2_model/netG_epoch10')
else:
    netG = torch.load('p2_model/netG_epoch10', map_location=torch.device('cpu'))

# netG.eval()

with torch.no_grad():
    noise = torch.randn(32, 100, 1, 1).to(device)
    output = netG(noise).cpu()

    pred = np.transpose(vutils.make_grid(output, padding=1, normalize=True), (1, 2, 0)).numpy()
    misc.imsave(output_path, pred)