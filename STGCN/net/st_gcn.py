import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
from .net import Unit2D, conv_init, import_class
from .unit_gcn import unit_gcn


default_backbone = [(64, 64, 1), (64, 64, 1), (64, 64, 1), (64, 128,2), (128, 128, 1),(128, 128, 1), (128, 256, 2), (256, 256, 1), (256, 256, 1)]

class Model(nn.Module):
    """ 时空图卷积网络
    用于基于骨架的动作识别。
    输入形状：
        输入形状应该是 (N, C, T, V, M)
        其中 N 是样本数量，
              C 是输入数据的通道数，S
              T 是序列长度，
              V 是关节或图节点的数量，
              M 是人数。
    参数：
        关于形状：
            channel (int): 输入数据中的通道数
            num_class (int): 分类类别数
            window_size (int): 输入序列的长度
            num_point (int): 关节或图节点的数量
            num_person (int): 人数
        关于网络：
            use_data_bn: 如果为 true，则数据首先输入到批量归一化层
            backbone_config: 主干网络的结构
        关于图卷积：
            graph: 骨架的图，由邻接矩阵表示
            graph_args: 图的参数
            mask_learning: 如果为 true，则使用掩码矩阵重新加权邻接矩阵
            use_local_bn: 如果为 true，则图中的每个节点都有特定的批量归一化层参数
        关于时间卷积：
            multiscale: 如果为 true，则使用多尺度时间卷积
            temporal_kernel_size: 时间卷积的核大小
            dropout: 每个时间卷积层前的 dropout 层的丢弃率
    """
    def __init__(self,
                 channel,  #输入数据的通道数。
                 num_class,  #输入数据的通道数。
                 window_size,  #输入序列的长度。
                 num_point,  # 关节点数或图节点数。
                 use_data_bn=False,  #是否在数据输入前使用批量归一化。
                 backbone_config=None,  #主干网络的结构配置。就是none
                 graph=None,  #骨架的图结构，用邻接矩阵表示。
                 graph_args=dict(),  #图的其他参数。
                 mask_learning=False,  #是否学习掩码矩阵以重新加权邻接矩阵。 #执行为true
                 use_local_bn=False,  #是否在图的每个节点上使用局部批量归一化。#执行为false
                 multiscale=False,  #是否使用多尺度时间卷积。 #执行为false
                 temporal_kernel_size=9,  # 时间卷积核的大小。
                 dropout=0.5):  #Dropout 层的丢弃率。
        super(Model, self).__init__()  #
        if graph is None: 
            raise ValueError()
        else:
            Graph = import_class(graph) 
            self.graph = Graph(**graph_args)
            self.A = torch.from_numpy(self.graph.A).float()

            #self.A的3个矩阵，恒定，且后2个完全一样。
        self.num_class = num_class  #
        self.use_data_bn = use_data_bn  # true

        self.multiscale = multiscale  #
        # Different bodies share batchNorma parameters or not
        self.M_dim_bn = True  #
        if self.M_dim_bn:  #根据 M_dim_bn 参数，初始化批量归一化层。
            self.data_bn = nn.BatchNorm1d(channel * num_point)
        else:  #没执行
            self.data_bn = nn.BatchNorm1d(channel * num_point)  #

        kwargs = dict(A=self.A,mask_learning=mask_learning,use_local_bn=use_local_bn,dropout=dropout,kernel_size=temporal_kernel_size)
        if self.multiscale:  #没执行
            unit = TCN_GCN_unit_multiscale  #
        else:#执行了
            unit = TCN_GCN_unit  #

        # backbone
        if backbone_config is None:  #执行了。以下根据 backbone_config 配置构建网络层。每层都是一个时空图卷积单元（TCN_GCN_unit 或 TCN_GCN_unit_multiscale），用于提取时空特征。
            backbone_config = default_backbone  #
        #default_backbone = [(64, 64, 1), (64, 64, 1), (64, 64, 1), (64, 128,2), (128, 128, 1),(128, 128, 1), (128, 256, 2), (256, 256, 1), (256, 256, 1)]

        backbone_in_c = backbone_config[0][0]  #获取主干网络第一层的输入通道数。第2个[0]，因为输入通道参数
        backbone_out_c = backbone_config[-1][1]  #获取主干网络最后一层的输出通道数。第2个[1]，因为输出通道参数
        #backbone_out_t = window_size  #设置主干网络输出的时间维度（即序列长度）为window_size，这通常在模型初始化时作为参数传入。  =150

        backbone = []  #空列表，用于存储主干网络的每一层。

        #len(backbone_config) = 9
        for in_c, out_c, stride in backbone_config:  #遍历backbone_config中的每个元素，每个元素包含一层的输入通道数（in_c）、输出通道数（out_c）和步长（stride）。#stride=1、2

            backbone.append(unit(in_c, out_c, stride=stride, **kwargs))

            #将每一层添加到backbone列表中。unit是构建网络层的函数，可能是TCN_GCN_unit或TCN_GCN_unit_multiscale，取决于是否使用多尺度时间卷积。**kwargs是传递给unit函数的其他参数。
            # if backbone_out_t % stride == 0:  #如果当前输出时间维度能被步长整除，直接除以步长。执行了。
            #     backbone_out_t = backbone_out_t // stride
            # else: #如果不能整除，除以步长后加1，确保输出时间维度至少保留一个单位。没执行。倒数第3次执行了。
            #     backbone_out_t = backbone_out_t // stride + 1

        self.backbone = nn.ModuleList(backbone)

        # head
        self.gcn0 = unit_gcn(channel,backbone_in_c,self.A,mask_learning=mask_learning,use_local_bn=use_local_bn) 
        self.tcn0 = Unit2D(backbone_in_c, backbone_in_c, kernel_size=9) 

        # tail
        self.person_bn = nn.BatchNorm1d(backbone_out_c) 
        self.fcn = nn.Conv1d(backbone_out_c, num_class, kernel_size=1) 
        conv_init(self.fcn)  #

    def forward(self, x): #
        N, C, T, V = x.size()
        if self.use_data_bn:
            if self.M_dim_bn:
                x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)

            else:
                x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
            x = self.data_bn(x)
            x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()

        else: 
            x = x.permute(0, 1, 2, 3).contiguous().view(N, C, T, V)
        x = self.gcn0(x) 
        x = self.tcn0(x) 
        for m in self.backbone: 
            x = m(x)
        # V pooling
        x = F.avg_pool2d(x, kernel_size=(1, V))
        
        c = x.size(1)
        t = x.size(2)
        x = x.view(N, c, t)
        # T pooling
        x = F.avg_pool1d(x, kernel_size=x.size()[2])
        # C fcn
        x = self.fcn(x)  

        x = F.avg_pool1d(x, x.size()[2:])  

        x = x.view(N, self.num_class)
        return x

class TCN_GCN_unit(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 A,
                 kernel_size=9,
                 stride=1,
                 dropout=0.5,
                 use_local_bn=False,
                 mask_learning=False):
        super(TCN_GCN_unit, self).__init__()
        half_out_channel = out_channel / 2
        self.A = A

        self.gcn1 = unit_gcn(in_channel,out_channel,A,use_local_bn=use_local_bn,mask_learning=mask_learning)

        self.tcn1 = Unit2D(out_channel,out_channel,kernel_size=kernel_size,dropout=dropout,stride=stride)

        if (in_channel != out_channel) or (stride != 1): #当输入输出通道不一致时，如4、7，backbone。执行Unit2D。否则不执行Unit2D
            self.down1 = Unit2D(in_channel, out_channel, kernel_size=1, stride=stride)
        else:
            self.down1 = None

    def forward(self, x): #以第4次为例
        # N, C, T, V = x.size()
        #所以当4、7，backbone时，原始x过了一层Unit2D，及conv2D，然后结果再加
        y = x if (self.down1 is None) else self.down1(x) #输入的x.shape = [128, 64, 300, 18]。输出的 y.shape = [128, 128, 150, 18]。self.down1是Unit2D。区别只是输入输出维度不一样

        #self.down1 = Unit2D(in_channel, out_channel, kernel_size=1, stride=stride)  #kernel_size=1
        #self.tcn1 = Unit2D(out_channel,out_channel,kernel_size=kernel_size,dropout=dropout,stride=stride) #kernel_size=9
        #只是输入输出维度调一下、kernel_size调一下，就OK了

        z = self.gcn1(x) #输入的x.shape = [128, 64, 300, 18]。输出的 z.shape = [128, 128, 300, 18]。gcn1改的就是通道数。

        P = self.tcn1(z) #输出的 P.shape = [128, 128, 150, 18].所以关键在于tcn1，将序列长度减半了。self.tcn1也是Unit2D。区别只是输入输出维度不一样。
        # 而且这也说明了，本质上不是self.down1造成的。因为无论哪层，都一定要经过self.tcn1。维度变了，序列长度就变了。tcn1或Unit2D改的就是序列长度。

        x = P + y #输出的 x.shape = [128, 128, 150, 18]
        return x


class TCN_GCN_unit_multiscale(nn.Module):#没执行
    def __init__(self,
                 in_channels,
                 out_channels,
                 A,
                 kernel_size=9,
                 stride=1,
                 **kwargs):
        super(TCN_GCN_unit_multiscale, self).__init__()
        self.unit_1 = TCN_GCN_unit(
            in_channels,
            out_channels / 2,
            A,
            kernel_size=kernel_size,
            stride=stride,
            **kwargs)
        self.unit_2 = TCN_GCN_unit(
            in_channels,
            out_channels - out_channels / 2,
            A,
            kernel_size=kernel_size * 2 - 1,
            stride=stride,
            **kwargs)

    def forward(self, x):
        return torch.cat((self.unit_1(x), self.unit_2(x)), dim=1)