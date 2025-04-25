# The based unit of graph convolutional networks.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .net import conv_init


class unit_gcn(nn.Module):
    def __init__(self,in_channels,out_channels,A,use_local_bn=False,kernel_size=1,stride=1,mask_learning=False):
        super(unit_gcn, self).__init__()

        # ==========================================
        # number of nodes
        self.V = A.size()[-1]

        # the adjacency matrixes of the graph
        self.A = Variable(A.clone(), requires_grad=False).view(-1, self.V, self.V)

        # number of input channels
        self.in_channels = in_channels

        # number of output channels
        self.out_channels = out_channels

        # if true, use mask matrix to reweight the adjacency matrix
        self.mask_learning = mask_learning

        # number of adjacency matrix (number of partitions)
        self.num_A = self.A.size()[0]

        # if true, each node have specific parameters of batch normalizaion layer.
        # if false, all nodes share parameters.
        self.use_local_bn = use_local_bn
        # ==========================================

        self.conv_list = nn.ModuleList([ 
            nn.Conv2d(
                self.in_channels,
                self.out_channels,
                kernel_size=(kernel_size, 1),
                padding=(int((kernel_size - 1) / 2), 0),
                stride=(stride, 1)) for i in range(self.num_A)
        ])

        if mask_learning:
            self.mask = nn.Parameter(torch.ones(self.A.size())) #18*18的矩阵，随机初始化，然后可学习。深度学习点就在这个self.mask = nn.Parameter

        if use_local_bn:
            self.bn = nn.BatchNorm1d(self.out_channels * self.V)
        else:
            self.bn = nn.BatchNorm2d(self.out_channels)

        self.relu = nn.ReLU()

        # initialize
        for conv in self.conv_list:
            conv_init(conv)

    def forward(self, x):  #次与次之间shape一样。同一次内会进行10次。  以下shape，以backbone第4层为例，输入[128, 64, 300, 18]、 输出[128, 128, 300, 18] ，    (64, 128,2)
        N, C, T, V = x.size() #[128, 64, 300, 18]
        self.A = self.A.cuda(x.get_device())

        A = self.A  #self.A的形状：(num_matrices, num_nodes, num_nodes)，num_matrices 表示邻接矩阵的数量（可能是多个不同的图表示），num_nodes 是每个图中的节点数量。#输出的 A.shape = [3, 18, 18]
        # reweight adjacency matrix

        if self.mask_learning:
            A = A * self.mask #输出的 A.shape = [3, 18, 18]
            #self.mask = nn.Parameter(torch.ones(self.A.size())) #18*18的矩阵，随机初始化，然后可学习。深度学习点就在这个self.mask = nn.Parameter

        # graph convolution #就这5行，就是那个图卷积操作，找到了

        for i, a in enumerate(A): #遍历A中的每一个邻接矩阵a
                                    #总共会遍历3次
                                    #enumerate(A) 会遍历 A 的第一个维度，即邻接矩阵的数量。i 是当前矩阵的索引，a 是当前的邻接矩阵。具体来说：
                                    # i：当前邻接矩阵的索引，范围从 0 到 num_matrices - 1。a：当前索引 i 对应的邻接矩阵，其形状为 (num_nodes, num_nodes)。
            xa = x.view(-1, V).mm(a).view(N, C, T, V) #输出的 xa.shape = [128, 128, 150, 18]

            #x.view(-1, V)，将输入张量 x 从形状 (N, C, T, V) 重塑为 (N * C * T, V)。这一步骤将图卷积的计算从节点维度和时间维度的卷积转换为矩阵乘法。

            #mm(a)：对重塑后的 x 进行矩阵乘法，其中 a 是邻接矩阵。这一步骤将节点特征图 x 和邻接矩阵 a 相乘。矩阵乘法的结果是 (N * C * T, num_nodes) 形状的张量，表示每个节点的特征被邻接矩阵加权。

            #view(N, C, T, V)：将乘法结果重新塑形为 (N, C, T, V)，即恢复为与输入数据相同的形状

            #以下if，即使在第4层backbone下，无论i == 0还是1、2，输出的y的shape都不变，一直一致
            if i == 0: #有执行.对于第一个邻接矩阵，直接将卷积结果存储在 y 中
                y = self.conv_list[i](xa)  # 输出的 y.shape = [128, 256, 150, 18]
                #print(f'i = {i},y.shape = {y.shape}')
                #self.conv_list[i] 是一个卷积层，作用在 xa 上，生成的 y 的形状为 (N, D_out, T, V)，其中 D_out 是卷积层的输出通道数

            else:  #也有执行.对于后续的邻接矩阵，将当前的卷积结果与之前的 y 相加。
                y = y + self.conv_list[i](xa) ##输出的 y.shape = [128, 128, 150, 18]
                #print(f'i = {i},y.shape = {y.shape}')
                #对每个邻接矩阵的卷积结果进行累加。self.conv_list[i](xa) 是对当前邻接矩阵 i 使用相应的卷积层进行卷积操作，并将其结果累加到 y 中。最终，y 包含所有邻接矩阵的卷积结果的累加。

            #图卷积的计算流程。具体步骤如下：
        # 图卷积计算：
        # 输入张量 x 被重塑以适应矩阵乘法操作。
        # 每个邻接矩阵 a 用于与输入特征图 x 进行矩阵乘法，得到加权后的特征图 xa。
        # 将加权后的特征图 xa 恢复为原始形状，并进行卷积操作。
        # 卷积结果累加：
        # 对第一个邻接矩阵，直接将卷积结果赋值给 y。
        # 对后续邻接矩阵，将卷积结果累加到 y 中。这种累加操作允许融合来自不同邻接矩阵的信息。
        #通过循环遍历不同的邻接矩阵并对每个邻接矩阵进行卷积，最后将结果进行累加，从而结合了不同的图结构信息。

            #           输出的xa.shape       i==0的输出的y.shape      i！=0的输出的y.shape
            #L = 1:  [128, 64, 300, 18] ， [128, 128, 300, 18] ，            \
            #L = 2:  [128, 64, 300, 18] ，         \           ，   [128, 128, 300, 18]
            #L = 3:  [128, 64, 300, 18] ，         \           ，   [128, 128, 300, 18]


        # batch normalization
        if self.use_local_bn: #10次均不执行
            y = y.permute(0, 1, 3, 2).contiguous().view(N, self.out_channels * V, T)
            y = self.bn(y)
            y = y.view(N, self.out_channels, V, T).permute(0, 1, 3, 2)
        else: #执行了
            y = self.bn(y)  #输出的y.shape =[128, 128, 300, 18]

        # nonliner
        y = self.relu(y) # 输出的 y.shape = [128, 128, 300, 18]
        return y
