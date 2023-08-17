
#pytorch libraries
import torch
from torch import nn
from torch.nn import init, Linear, ReLU, Softmax
import torch.nn.functional as F

from torchvision.models import wide_resnet50_2
wresn50 = wide_resnet50_2(pretrained=False, progress = True)

#adjust Wide-ResNet-50 to my dataset
class wr50(nn.Module):
    def __init__(self, pretrained_model):
        super(wr50,self).__init__()
        self.wrn50 = pretrained_model
        self.fl1 = nn.Linear(1000, 256)
        self.fl2 = nn.Linear(256,10)
        
    def forward(self, X):
        X = self.wrn50(X)
        X = F.relu(self.fl1(X))
        X = F.dropout(X, p=0.25)
        X = self.fl2(X)
        return X