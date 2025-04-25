import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math


class Unit2D(nn.Module):
    def __init__(self,
                 D_in,
                 D_out,
                 kernel_size,
                 stride=1,
                 dim=2, #一直是2。
                 dropout=0,
                 bias=True):
        super(Unit2D, self).__init__()
        pad = int((kernel_size - 1) / 2)
        if dim == 2: #一直是2
            self.conv = nn.Conv2d(D_in,D_out,kernel_size=(kernel_size, 1),padding=(pad, 0),stride=(stride, 1),bias=bias)
        elif dim == 3: #没执行
            self.conv = nn.Conv2d(
                D_in,
                D_out,
                kernel_size=(1, kernel_size),
                padding=(0, pad),
                stride=(1, stride),
                bias=bias)
        else:
            raise ValueError()

        self.bn = nn.BatchNorm2d(D_out)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # initialize
        conv_init(self.conv)

    def forward(self, x):  #太简单，dropout、2D卷积、批归一化、relu
        x = self.dropout(x)
        x = self.relu(self.bn(self.conv(x)))
        return x


def conv_init(module):
    # he_normal
    n = module.out_channels
    for k in module.kernel_size:
        n *= k
    module.weight.data.normal_(0, math.sqrt(2. / n))
# 该函数使用 He 正态初始化方法来初始化卷积层的权重。它通过计算卷积核的总输入特征数来确定标准差，
# 然后用正态分布初始化权重。这样可以帮助保持前向传播过程中激活值的方差稳定，特别适合使用 ReLU 激活函数的网络。

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
