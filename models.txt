import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import GCNConv

from layers import GCN, HGPSLPool
import numpy as np
import pandas as pd


class Model(torch.nn.Module):
    def __init__(self, args, i):
        super(Model, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.num_nodes = args.num_nodes
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio
        self.sample = args.sample_neighbor
        self.sparse = args.sparse_attention
        self.sl = args.structure_learning
        self.lamb = args.lamb

        self.conv1 = GCNConv(self.num_features, self.nhid)  # https://blog.csdn.net/qq_41987033/article/details/103377561;num_features咱自己设定的为1，我猜是特征的维数，code本来用了node的attribute和label（onehot），但咱node没label所以就只有一维。nhid默认128，应该是输出特征维度。相当于每个点得到128维向量
        self.conv2 = GCN(self.nhid, self.nhid)
        self.conv3 = GCN(self.nhid, self.nhid)
        # self.conv4 = GCN(self.nhid, self.nhid)
        # self.conv5 = GCN(self.nhid, self.nhid)

        self.pool1 = HGPSLPool(self.nhid, self.pooling_ratio, self.sample, self.sparse, self.sl, self.lamb, score_shape=self.num_nodes[i])
        self.pool2 = HGPSLPool(self.nhid, self.pooling_ratio, self.sample, self.sparse, self.sl, self.lamb, score_shape=round(self.num_nodes[i] / 2)) # TODO 测试一层池化层（作为消融实验）
        # self.pool3 = HGPSLPool(self.nhid, self.pooling_ratio, self.sample, self.sparse, self.sl, self.lamb, score_shape=round(self.num_nodes[i] / 4)) # TODO 测试三层池化层（作为消融实验）
        # self.pool4 = HGPSLPool(self.nhid, self.pooling_ratio, self.sample, self.sparse, self.sl, self.lamb, score_shape=round(self.num_nodes[i] / 8)) # TODO 测试四层池化层（作为消融实验）

        self.lin1 = torch.nn.Linear(self.nhid*2, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid // 2)
        self.lin3 = torch.nn.Linear(self.nhid // 2, self.num_classes)  # num_classes最终的label有几类就是几
        # self.edge_index = torch.nn.Parameter(adj_matrix, requires_grad=True)
        self.layernorm1 = torch.nn.LayerNorm(self.nhid)  # 为什么虽然每个layernorm的输入大小都是nhid，但是每个layernorm的参数都不一样？是因为每个layernorm的输入都不一样吗？是的，因为每个layernorm的输入都是上一层的输出，而上一层的输出是不一样的，所以每个layernorm的参数都不一样
        self.layernorm2 = torch.nn.LayerNorm(self.nhid)
        self.layernorm3 = torch.nn.LayerNorm(self.nhid)
        # self.layernorm4 = torch.nn.LayerNorm(self.nhid)
        # self.layernorm5 = torch.nn.LayerNorm(self.nhid)

    def forward(self, data, is_train=True):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = None
        x = x[:,0].unsqueeze(1)  # 因为我们没用到node label，本来有三列的，后两列就是node label的one hot结果。没用所以就删了，此时x为6400*1
        x = F.relu(self.layernorm1(self.conv1(x, edge_index, edge_attr)))  # x变成6400*128了  brca 为16000*1  todo 为啥要加layernorm是因为要对输出进行归一化（因为之前画了一下池化完后的特征重要性柱状图，只有极个别的特征重要性很高，其他的都很低，所以要归一化）
        x, edge_index, edge_attr, batch = self.pool1(x, edge_index, edge_attr, batch, is_train, num_i=1)  # brca x形状为16000*128
        # x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)  # mix pooling,形状为32*256  # todo 为啥要注释掉是因为，咱是用的最后一层网络的结构，所以没有x1+x2+x3的需求了，也就不用计算x1 x2 x3了

        x = F.relu(self.layernorm2(self.conv2(x, edge_index, edge_attr)))  # TODO 测试一层池化层（作为消融实验）
        x, edge_index, edge_attr, batch = self.pool2(x, edge_index, edge_attr, batch, is_train, num_i=2)  # TODO 测试一层池化层（作为消融实验）
        # x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)  # 形状为32*256

        # x = F.relu(self.layernorm3(self.conv3(x, edge_index, edge_attr)))  # TODO 测试三层池化层（作为消融实验）
        # x, edge_index, edge_attr, batch = self.pool3(x, edge_index, edge_attr, batch, is_train, num_i=3)  # TODO 测试三层池化层（作为消融实验）

        # x = F.relu(self.layernorm4(self.conv4(x, edge_index, edge_attr)))  # TODO 测试四层池化层（作为消融实验）
        # x, edge_index, edge_attr, batch = self.pool4(x, edge_index, edge_attr, batch, is_train, num_i=4)  # TODO 测试四层池化层（作为消融实验）

        x = F.relu(self.layernorm3(self.conv3(x, edge_index, edge_attr)))  # TODO layernorm和conv要和前边的都不同
        # x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        # x = F.relu(x1) + F.relu(x2) + F.relu(x3)
        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)  # 因为输出的事最大池化和平均池化的拼接，所以是2倍的nhid
        x = F.relu(self.lin1(x))  # 所以lin1的输入是2倍的nhid，输出是nhid
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))  # 师兄说relu和dropout重复几次都无所谓
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.log_softmax(self.lin3(x), dim=-1)  # 因为log softmax是在softmax的基础上加了个log。所以如果这个log_softmax的输入都<1，那么输出都是负的。

        return x  # （32,2）的tensor
