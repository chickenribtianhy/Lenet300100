import os
import sys

import torch 
import torch.nn as nn
import torch.nn.functional as F
import math

from pytorch_apis import MatrixMuliplication
from myLinear import Linear

cwd = os.getcwd()
sys.path.append(cwd+'../')


class LeNet_300_100(nn.Module):
    def __init__(self, prune):
        super(LeNet_300_100, self).__init__()
        self.linear1 = Linear(28*28, 300)
        self.relu_ip1 = nn.ReLU(inplace=True)
        self.linear2 = Linear(300, 100)
        self.relu_ip2 = nn.ReLU(inplace=True)
        self.ip3 = nn.Linear(100, 10)
        self.device = torch.device('cuda')
        return
    

    def forward(self, x):
        x = x.view(x.size(0), 28*28)
        # x = MatrixMuliplication(x, self.weight1, x.size(0), 300, self.device) + self.bias1
        x = self.linear1(x)
        x = self.relu_ip1(x)
        
        # x = MatrixMuliplication(x, self.weight2, x.size(0), 100, self.device) + self.bias2
        x = self.linear2(x)
        x = self.relu_ip2(x)
        
        x = self.ip3(x)
        # x = F.softmax(x, dim=1)
        return x