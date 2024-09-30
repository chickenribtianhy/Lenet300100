import os
import sys

import torch 
import torch.nn as nn
import torch.nn.functional as F
import math

from pytorch_apis import MatrixMuliplication

cwd = os.getcwd()
sys.path.append(cwd+'../')


class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()
        self._weight = nn.Parameter(torch.empty(in_features, out_features))
        self._bias = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()
        self.device = torch.device('cuda')
        return
    
    def reset_parameters(self) -> None:
        # kaiming_uniform_ initializes the weight matrix with a uniform distribution
        nn.init.kaiming_uniform_(self._weight, a=math.sqrt(5))
        # bias:
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self._weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self._bias, -bound, bound)
        return
    
    def forward(self, x):
        return torch.matmul(x, self._weight) + self._bias
    

        

            