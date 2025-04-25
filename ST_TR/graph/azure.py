import numpy as np
from . import tools

# Joint index:
# {0,  "Pelvis"},
# {1,  "SpineNavel"},
# {2,  "SpineChest"},
# {3,  "Neck"},
# {4,  "ClavicleLeft"},
# {5,  "ShoulderLeft"},
# {6,  "ElbowLeft"},
# {7,  "WristLeft"},
# {8,  "HandLeft"},
# {9,  "HandTipLeft"},
# {10, "ThumbLeft"},
# {11, "ClavicleRight"},
# {12, "ShoulderRight"},
# {13, "ElbowRight"},
# {14, "WristRight"},
# {15, "HandRight"},
# {16, "HandTipRight"},
# {17, "ThumbRight"},
# {18, "HipLeft"},
# {19, "KneeLeft"},
# {20, "AnkleLeft"},
# {21, "FootLeft"},
# {22, "HipRight"},
# {23, "KneeRight"},
# {24, "AnkleRight"},
# {25, "FootRight"},
# {26, "Head"},
# {27, "Nose"},
# {28, "EyeLeft"},
# {29, "EarLeft"},
# {30, "EyeRight"},
# {31, "EarRight"}

# Edge format: (origin, neighbor)
num_node = 32
self_link = [(i, i) for i in range(num_node)]
inward = [
    (0, 1), (1, 2), (2, 3), (2, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (7, 10),
    (2, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (14, 17),
    (0, 18), (18, 19), (19, 20), (20, 21),
    (0, 22), (22, 23), (23, 24), (24, 25),
    (3, 26), (26, 27), (26, 28), (26, 29), (26, 30), (26, 31)
]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward

class Graph():
    """ The Graph to model the skeletons extracted by the openpose

    Arguments:
        labeling_mode: must be one of the follow candidates
            uniform: Uniform Labeling
            dastance*: Distance Partitioning*
            dastance: Distance Partitioning
            spatial: Spatial Configuration
            DAD: normalized graph adjacency matrix
            DLD: normalized graph laplacian matrix

    For more information, please refer to the section 'Partition Strategies' in our paper.

    """
    def __init__(self, labeling_mode='uniform'):#执行了
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'uniform':
            A = tools.get_uniform_graph(num_node, self_link, neighbor)
        elif labeling_mode == 'distance*':
            A = tools.get_uniform_distance_graph(num_node, self_link, neighbor)
        elif labeling_mode == 'distance':
            A = tools.get_distance_graph(num_node, self_link, neighbor)
        elif labeling_mode == 'spatial': 
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        elif labeling_mode == 'DAD':
            A = tools.get_DAD_graph(num_node, self_link, neighbor)
        elif labeling_mode == 'DLD':
            A = tools.get_DLD_graph(num_node, self_link, neighbor)
        else:
            raise ValueError()
        return A


def main():
    mode = ['uniform', 'distance*', 'distance', 'spatial', 'DAD', 'DLD']  #用的spatial，其他没用
    np.set_printoptions(threshold=np.nan)
    for m in mode:
        print('=' * 10 + m + '=' * 10)
        print(Graph(m).get_adjacency_matrix())


if __name__ == '__main__':
    main()
