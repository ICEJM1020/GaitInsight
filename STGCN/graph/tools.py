import numpy as np


def edge2mat(link, num_node): #将边列表转换为邻接矩阵.link: 边的列表，格式为 (origin, neighbor).num_node: 图中的节点总数
    A = np.zeros((num_node, num_node)) #全零的邻接矩阵 A，尺寸为 (num_node, num_node)
    for i, j in link: #遍历 link 中的每一对 (i, j)，将邻接矩阵中 (j, i) 的位置设置为 1，表示存在一条从 j 到 i 的边。
        A[j, i] = 1
    return A
#如果 link = [(1, 0), (2, 1)] 和 num_node = 3，则 edge2mat(link, num_node) 将返回。左上角（0,0）。图的邻接矩阵A[i,j]表示第j个节点到第i个节点是否有边。
#[(1, 0), (2, 1)，（i,j）]表示第i个节点到第j个节点是否有边.所以才正好反着的。
# [[0, 1, 0],
#  [0, 0, 1],
#  [0, 0, 0]]

def normalize_digraph(A): #对有向图的邻接矩阵进行归一化处理。具体来说，这是计算每个节点的入度的逆，然后将邻接矩阵乘以这个逆度矩阵来进行归一化。
    Dl = np.sum(A, 0) #计算每个节点的入度，得到入度矩阵 Dl。Dl[i] 是第 i 列的和，即节点 i 的入度。即射入到第i个点的和。
    num_node = A.shape[0] #

    Dn = np.zeros((num_node, num_node)) #以下4行创建对角矩阵 Dn，对角线上的元素是 Dl 中对应元素的倒数
    for i in range(num_node): #
        if Dl[i] > 0: #
            Dn[i, i] = Dl[i]**(-1) #-1次方

    AD = np.dot(A, Dn) #使用矩阵乘法将原始邻接矩阵 A 乘以归一化的对角矩阵 Dn
    return AD

def get_spatial_graph(num_node, self_link, inward, outward): #self.A的3个矩阵，恒定，且后2个完全一样。
    #生成一个表示空间图的邻接矩阵堆栈。它包括自连接、入边和出边的邻接矩阵。
    I = edge2mat(self_link, num_node) #生成自连接矩阵 I
    # print('==I==')
    # print(I)
    In = normalize_digraph(edge2mat(inward, num_node)) #生成入边的归一化邻接矩阵 In
    # print('==In==')
    # print(In)
    Out = normalize_digraph(edge2mat(outward, num_node)) #生成出边的归一化邻接矩阵 Out
    # print('==Out==')
    # print(Out)
    #这三个函数都一样，edge2mat，都是在将边生成矩阵。区别只是输入不同。self_link和inward、outward。然后I原封表示。In和out再乘以归一化的对角矩阵。
    A = np.stack((I, In, Out)) #将 I、In 和 Out 堆叠在一起，形成一个三维数组 A。说明这个的本来是有向图。
    return A
#假设 num_node = 3，self_link = [(0, 0), (1, 1), (2, 2)]，inward = [(1, 0), (2, 1)]，outward = [(0, 1), (1, 2)]，则：
# I 是一个对角矩阵。
# In 和 Out 是经过归一化处理的入边和出边矩阵。
# A 将是一个 (3, 3, 3) 形状的三维数组，其中包含了三个邻接矩阵的堆栈
#这说明在同一条边，因为可以有2个方向，所以要设2个矩阵。如果无向图，则两个矩阵一样。无向图，不仅两个矩阵一样，在1个矩阵内也是对称的。



def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD


def get_uniform_graph(num_node, self_link, neighbor):
    A = normalize_digraph(edge2mat(neighbor + self_link, num_node))
    return A


def get_uniform_distance_graph(num_node, self_link, neighbor):
    I = edge2mat(self_link, num_node)
    N = normalize_digraph(edge2mat(neighbor, num_node))
    A = I - N
    return A


def get_distance_graph(num_node, self_link, neighbor):
    I = edge2mat(self_link, num_node)
    N = normalize_digraph(edge2mat(neighbor, num_node))
    A = np.stack((I, N))
    return A





def get_DAD_graph(num_node, self_link, neighbor):
    A = normalize_undigraph(edge2mat(neighbor + self_link, num_node))
    return A


def get_DLD_graph(num_node, self_link, neighbor):
    I = edge2mat(self_link, num_node)
    A = I - normalize_undigraph(edge2mat(neighbor, num_node))
    return A
