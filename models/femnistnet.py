import torch.nn as nn
import torchvision.transforms as transforms
import math
from .binarized_modules import  BinarizeLinear,BinarizeConv2d

__all__ = ['femnistnet']

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.tanh1 = nn.Tanh()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.tanh2 = nn.Tanh()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(7*7*64, 2048)
        self.bn3 = nn.BatchNorm1d(2048)
        self.tanh3 = nn.Tanh()
        self.fc2 = nn.Linear(2048, 62)
        self.bn4 = nn.BatchNorm1d(62)
        self.logsoftmax=nn.LogSoftmax()
        self.regime = {
            0: {'optimizer':'Adam', 'lr':3e-4},
            30:{'lr':1e-4},
            60:{'lr':5e-5},
        }
        '''
        self.regime = {
            0: {'optimizer':'Adam', 'lr':5e-2},
            30:{'lr':2e-2},
            60:{'lr':1e-2},
            90:{'lr':5e-3},
            120:{'lr':2e-3}
        }
        '''

    def forward(self, x):
        x = x.view(-1,1,28,28)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.tanh1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.tanh2(x)
        x = self.maxpool2(x)
        x = x.view(-1,7*7*64)
        x = self.fc1(x)
        x = self.bn3(x)
        x = self.tanh3(x)
        x = self.fc2(x)
        x = self.bn4(x)
        return self.logsoftmax(x)


def femnistnet(**model_config):
    return Net()
