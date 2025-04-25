# sys
import os
import sys
import numpy as np
import random
import pickle

# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms

# visualization
import time

# operation
from . import tools
 

class Feeder(torch.utils.data.Dataset): #加载和预处理骨架动作识别数据集
    def __init__(self, #义了类的初始化方法，接受数据路径、标签路径、随机选择、随机偏移、随机移动、窗口大小、归一化和调试模式作为参数。
                 data_path, #以前是路径"./data/NTU-RGB-D/xview/val_data.npy"。现在直接是[587,3,40,32]
                 label_path, #以前是路径"./data/NTU-RGB-D/xview/val_label.pkl"。现在直接是[587]
                 random_choose=False,  #执行时为false
                 random_shift=False,  #执行时为false
                 random_move=False,  #执行时为false
                 window_size=-1,
                 normalization=False,  #执行时为false
                 debug=False
                 ):
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.N, self.C, self.T, self.V = -1,-1,-1,-1

        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move

        self.window_size = window_size
        self.normalization = normalization

        self.load_data()
        if normalization:
            self.get_mean_map()

    def load_data(self): #
        # data: N C T V
        # load label
        self.label = list(np.load(self.label_path, allow_pickle=True))
        self.data = np.load(self.data_path, allow_pickle=True)

        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
        self.N, self.C, self.T, self.V = self.data.shape

    def get_mean_map(self):
        data = self.data
        N, C, T, V  = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 1, 3)).reshape((N * T , C * V)).std(axis=0).reshape((C, 1, V, 1)) 

    def __getitem__(self, index): 

        data_numpy = self.data[index]

        label = self.label[index]

        # normalization
        if self.normalization:
            data_numpy = (data_numpy - self.mean_map) / self.std_map 
        # processing
        if self.random_shift:
            data_numpy = tools.random_shift(data_numpy)
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size) 
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)

        return data_numpy, label

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)] #
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self


def test(data_path, label_path, vid=None):
    import matplotlib.pyplot as plt
    loader = torch.utils.data.DataLoader(
        dataset=Feeder(data_path, label_path),
        batch_size=64,
        shuffle=False,
        num_workers=2)

    if vid is not None:
        data, label = loader.dataset[0]
        data = data.reshape((1, ) + data.shape)

        # for batch_idx, (data, label) in enumerate(loader):
        N, C, T, V = data.shape

        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)

        pose, = ax.plot(np.zeros(V), np.zeros(V), 'g^')
        ax.axis([-1, 1, -1, 1])

        for n in range(N):
            for t in range(T):
                x = data[n, 0, t, :]
                y = data[n, 1, t, :]
                z = data[n, 2, t, :]
                pose.set_xdata(x)
                pose.set_ydata(y)
                fig.canvas.draw()
                plt.pause(1)


if __name__ == '__main__':
    data_path = "/Users/timberzhang/Documents/Project/2023-GaitParameter/TestOutput/gait_data.npy"
    label_path = "/Users/timberzhang/Documents/Project/2023-GaitParameter/TestOutput/gait_label.npy"

    test(data_path, label_path, vid='S003C001P017R001A044')
