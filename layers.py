import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from sparse_softmax import Sparsemax
from torch.nn import Parameter
from torch_geometric.data import Data
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.pool.topk_pool import topk, filter_adj
from torch_geometric.utils import softmax, dense_to_sparse, add_remaining_self_loops
from torch_scatter import scatter_add
from torch_sparse import spspmm, coalesce

import numpy as np
import pandas as pd


class TwoHopNeighborhood(object):
    def __call__(self, data):
        edge_index, edge_attr = data.edge_index, data.edge_attr
        n = data.num_nodes

        fill = 1e16
        value = edge_index.new_full((edge_index.size(1),), fill, dtype=torch.float)
        index, value = spspmm(edge_index, value, edge_index, value, n, n, n, True)  # 两个稀疏矩阵相乘的结果，代表图上 u 到 v 恰好经过两条边的路径的条数的矩阵。https://www.cnblogs.com/BlairGrowing/p/15492526.html
        edge_index = torch.cat([edge_index, index], dim=1)  # edge_index 35584（之前的）与index 174976（稀疏矩阵相乘的结果）相加=210560
        if edge_attr is None:
            data.edge_index, _ = coalesce(edge_index, None, n, n)
        else:
            value = value.view(-1, *[1 for _ in range(edge_attr.dim() - 1)])  # 不知道这两行要干啥，反正没影响
            value = value.expand(-1, *list(edge_attr.size())[1:])
            edge_attr = torch.cat([edge_attr, value], dim=0)
            data.edge_index, edge_attr = coalesce(edge_index, edge_attr, n, n, op='min',)#去掉重复的，因为cat拼接的时候是愣把俩稀疏矩阵的index和value分别拼在一起，但是有重复的index，需要删了。https://www.wolai.com/chunxiaozhang/wSjfxLBdhK9MRNLYERnsBe#rPkYLx27B9247qVWZQ1rKb;https://blog.csdn.net/qq_40697172/article/details/120516369;https://zhuanlan.zhihu.com/p/76890808
            edge_attr[edge_attr >= fill] = 0
            data.edge_attr = edge_attr

        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


class GCN(MessagePassing):
    def __init__(self, in_channels, out_channels, cached=False, bias=True, **kwargs):
        super(GCN, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cached = cached
        self.cached_result = None
        self.cached_num_edges = None

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        nn.init.xavier_uniform_(self.weight.data)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
            nn.init.zeros_(self.bias.data)
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.cached_result = None
        self.cached_num_edges = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype, device=edge_index.device)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight=None):
        x = torch.matmul(x, self.weight)

        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}'.format(self.cached_num_edges, edge_index.size(1)))

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            edge_index, norm = self.norm(edge_index, x.size(0), edge_weight, x.dtype)
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels, self.out_channels)


class NodeInformationScore(MessagePassing):
    def __init__(self, improved=False, cached=False, **kwargs):
        super(NodeInformationScore, self).__init__(aggr='add', **kwargs)

        self.improved = improved
        self.cached = cached
        self.cached_result = None
        self.cached_num_edges = None

    @staticmethod  # 用于修饰类中的方法，使其可以在不创建类实例的时候就调用这个方法
    def norm(edge_index, num_nodes, edge_weight, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype, device=edge_index.device)

        row, col = edge_index  # 每个line的起和终点.edge_index就是邻接矩阵A
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)  # 我觉得是degree，deg=每个node有几个相关的线就=几
        deg_inv_sqrt = deg.pow(-0.5)  #
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, 0, num_nodes)  # 不用管怎么实现的，就是给每个没有自循环的节点加了自循环边。也就是加了200*32个edge（35584+200*32）。但是权重都给的0。

        row, col = edge_index  # 现在是加了自循环后的，每个line的source node 和 target node
        expand_deg = torch.zeros((edge_weight.size(0),), dtype=dtype, device=edge_index.device)
        expand_deg[-num_nodes:] = torch.ones((num_nodes,), dtype=dtype, device=edge_index.device)  # 把倒数的6400个置1，前边的全0

        return edge_index, expand_deg - deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]  # 就相当于I- Di的-1/2 * A * Dj的-1/2. i,j分别是行和列.结果就是把邻接矩阵归一化了。又因为是用的I-xxx，I除了自环其他都是0，所以这里的结果大都是负的。但是不代表不好，只是他写的是E-这种方式，就会得到一堆负的而已。这也是为啥后边要abs的原因。至于为啥前D是row后D是col：正常的norm就是D-1/2*A*D-1/2,这里的俩D，其实分别就是行和列。比如【1,3】有边，那么前D的第一行*A得到A_tmp的第一行，而前D的第一行只有1有值，其他都0，所以A_tmp的第一行也是只有1有值。那么A_tmp*后D的第3列得到最终A的第3列，而后D只有第3col有值，也就是最终A的第三列里，只有第3个空有值。

    def forward(self, x, edge_index, edge_weight):
        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}'.format(self.cached_num_edges, edge_index.size(1)))

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)#每个batch里的边的数量。比如=35584，就是1112*32.
            edge_index, norm = self.norm(edge_index, x.size(0), edge_weight, x.dtype)  # edge_index就是邻接矩阵A+I（也就是加了自环35584+200*32=41984），norm就是得分
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result

        return self.propagate(edge_index, x=x, norm=norm)  # 所有的逻辑代码都在forward()里面，当我们调用propagate()函数之后，它将会在内部调用message()和update()。

    def message(self, x_j, norm):  # 在message()函数中，需要通过norm对相邻的节点特征x_j进行归一化。x_j包含每条边的源节点特征，即每个节点的邻居。通过在变量名称中添加_i或_j来说明特征传递的来源。任何节点特征都可以这样转换，只要有源节点或目的节点特征。https://zhuanlan.zhihu.com/p/427083823
        return norm.view(-1, 1) * x_j  # norm.view(-1,1)相当于一个系数，是个（batch所有边,1）,x_j是个（batch所有边，128）对应相乘而已，不是矩阵乘法。至于为啥x_j要是128的，我猜是因为要获得非线性映射的能力。最终的结果，就是整个batch的所有边归一化之后的得分

    def update(self, aggr_out):
        return aggr_out


class HGPSLPool(torch.nn.Module):
    def __init__(self, in_channels, ratio=0.8, sample=False, sparse=False, sl=True, lamb=1.0, negative_slop=0.2, score_shape=None):
        super(HGPSLPool, self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.sample = sample
        self.sparse = sparse
        self.sl = sl
        self.negative_slop = negative_slop
        self.lamb = lamb

        self.att = Parameter(torch.Tensor(1, self.in_channels * 2))
        nn.init.xavier_uniform_(self.att.data)#反正就是一种初始化，保证每层传播的时候方差尽量不变
        self.sparse_attention = Sparsemax()
        self.neighbor_augment = TwoHopNeighborhood()
        self.calc_information_score = NodeInformationScore()
        self.score = Parameter(torch.ones([score_shape,])/score_shape, requires_grad=False)  # 给最开始每个总的批次（总batch）样本的得分赋值，这里是1000或503个，每个得分是1/1000或1/503。最后输出特征重要性的时候，（读取最优pth然后只做test不train），这个就是特征重要性。
        self.sdk = Parameter(torch.sqrt(torch.tensor(in_channels).float()), requires_grad=False)
    def forward(self, x, edge_index, edge_attr, batch=None, is_train=True, num_i=None):
        time1 = time.time()
        # if is_train:
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        x_information_score = self.calc_information_score(x, edge_index, edge_attr)  # 把每个节点的得分都embedding成128维的了，所以是3200*128
        score = torch.sum(torch.abs(x_information_score), dim=1)  # 3200*1，每个节点的得分是一个标量(tensor)
        time2 = time.time()
        # print('time2-time1={}'.format(time2 - time1))
        # my_score = []
        # 新建一个名为feature_importance的txt文档，然后并写入数组b
        with open('feature_importance.txt', 'a') as f:
            f.write(str(score.cpu().detach().numpy()))

        my_score_one_graph = score.view([torch.max(batch)+1, -1]).T.mean(-1)  # score.view([torch.max(batch)+1, -1]):32,1000.32个minibatch，每个minibatch1000个节点，每个节点的得分是一个标量(tensor)。.T:1000,32。.mean(-1):列取平均，变为1000的tensor。一句话，就是一个batch里每个患者的1000个节点的得分取平均，得到一个1000的tensor。然后再把32个batch的1000个tensor取平均，得到一个1000的tensor。
        if self.training:
            my_score_one_graph = torch.softmax(my_score_one_graph/self.sdk, dim=0) * 0.1 + self.score * 0.9  # 0.1部分就是当前批次batch的score，0.9部分就是之前累积的score。这里的softmax是为了让每个节点的得分都在0-1之间，然后再乘以0.1，这样就可以让每个节点的得分都在0-0.1之间，然后再加上之前累积的score，这样就可以让每个节点的得分都在0-1之间。
            self.score = torch.nn.Parameter(my_score_one_graph, requires_grad=False)  # 把self.score作为一个可训练参数，这样就可以在下一次训练的时候，把这一次的score作为下一次的0.9部分。
        else:  # 测试集直接用训练集挑选出来的节点。比如训练集挑选出来的节点是[1,2,3,4,5]，那么测试集就直接用这5个节点。
            my_score_one_graph = self.score
        # print('time3-time2={}'.format(time3 - time2))
        #my_topk = sorted(my_score_one_graph,reverse=True)[:int(len(batch)/len(set(np.array(batch.cpu())))*self.ratio)]# int(len(batch)/len(set(np.array(batch.cpu()))))=200
        my_score = my_score_one_graph.unsqueeze(dim=-1).repeat([1,batch.max()+1]).T.reshape(-1)

        # Graph Pooling
        original_x = x
        perm = topk(my_score, self.ratio, batch)
        x = x[perm]
        batch = batch[perm]
        induced_edge_index, induced_edge_attr = filter_adj(edge_index, edge_attr, perm, num_nodes=my_score.size(0))  # filter_adj就是把edge_index和edge_attr里的起点和终点都非-1的，也就是大于等于0的。就是起点和终点的排序，都要在6400个中排前3200个的edge
        time5 = time.time()
        # print('time5-time4={}'.format(time5 - time4))

        # original_x = x
        # x = x[my_perm]
        # batch = batch[my_perm]
        # induced_edge_index, induced_edge_attr = filter_adj(edge_index, edge_attr, my_perm, num_nodes=original_x.shape[0])  # 要了那些起点和终点都非-1，也就是大于等于0的。就是起点和终点的排序，都要在6400个中排前3200个的edge
        # time4 = time.time()
        # print('time4-time3={}'.format(time4 - time3))
        # # Graph Pooling
        # original_x = x
        # perm = topk(score, self.ratio, batch)  # perm为3200,的。是top3200的下标。至于为啥要输入batch。就是为了让每个图都能保存相同比例（比如50%）的点。对于整个batch的每个图保留比例相同。
        # x = x[perm]  # x从6400*128→3200*128
        # batch = batch[perm]  # batch从6400,→3200,。没用上。
        # induced_edge_index, induced_edge_attr = filter_adj(edge_index, edge_attr, perm, num_nodes=score.size(0))  # 要了那些起点和终点都非-1，也就是大于等于0的。就是起点和终点的排序，都要在6400个中排前3200个的edge


        # Discard structure learning layer, directly return
        if self.sl is False: #structure learning默认是true
            return x, induced_edge_index, induced_edge_attr, batch

        # Structure Learning
        if self.sample:
            # A fast mode for large graphs.
            # In large graphs, learning the possible edge weights between each pair of nodes is time consuming.
            # To accelerate this process, we sample it's K-Hop neighbors for each node and then learn the
            # edge weights between them.
            time6 = time.time()
            k_hop = 3  # 这里应该是指的a连着b，b连着c，c连着d。我以为是a到c，但下边hop_data的计算看起来是a到d
            if edge_attr is None:
                edge_attr = torch.ones((edge_index.size(1),), dtype=torch.float, device=edge_index.device)

            hop_data = Data(x=original_x, edge_index=edge_index, edge_attr=edge_attr)
            for _ in range(k_hop - 1):#这里应该是所有类似a-b-c-d-e，也就是a-e的可能的所有的边，一共有502336条。但是其实很多不会存在，比如边的节点就不是前50%的。这一步就加了自环我觉得。
                hop_data = self.neighbor_augment(hop_data)
            hop_edge_index = hop_data.edge_index
            hop_edge_attr = hop_data.edge_attr
            # new_edge_index, new_edge_attr = filter_adj(hop_edge_index, hop_edge_attr, my_perm, num_nodes=original_x.shape[0])  # todo perm和score改了！！直接从502336暴降到99840条。
            new_edge_index, new_edge_attr = filter_adj(hop_edge_index, hop_edge_attr, perm, num_nodes=my_score.shape[0]) # TODO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            new_edge_index, new_edge_attr = add_remaining_self_loops(new_edge_index, new_edge_attr, 0, x.size(0))  # 其实没增加,但是顺序变了。之前是source节点从小到大排，现在把所有自循环节点都放到了最后
            row, col = new_edge_index
            weights = (torch.cat([x[row], x[col]], dim=1) * self.att).sum(dim=-1)  # self.att就是那个a。只是点乘对应位置相乘
            weights = F.leaky_relu(weights, self.negative_slop) + new_edge_attr * self.lamb  # 就是文中的两个节点间（也就是每条边）的相似度得分E，我觉得是计算了Khop=4的
            adj = torch.zeros((x.size(0), x.size(0)), dtype=torch.float, device=x.device)
            adj[row, col] = weights  # 把每条边的权重赋值进邻接矩阵。但这时还是稠密的矩阵，下一步稀疏化
            new_edge_index, weights = dense_to_sparse(adj)  # 稀疏化,而且new_edge_index也变了，之前自环都在最后，现在全部是按照索引值从小到大的顺序
            row, col = new_edge_index  # 获取稀疏化矩阵的 row 和 col
            if self.sparse:
                new_edge_attr = self.sparse_attention(weights, row) #这里使得new_edge_index从99840到了37742（没法被32整除了）,这里输入的x就是weight，label就是row也就是source节点
            else:
                new_edge_attr = softmax(weights, row, x.size(0))
            # filter out zero weight edges
            adj[row, col] = new_edge_attr
            new_edge_index, new_edge_attr = dense_to_sparse(adj)
            # release gpu memory
            del adj
            torch.cuda.empty_cache()
            time7 = time.time()
            # print('time7-time6={}'.format(time7 - time6))
        else:
            # Learning the possible edge weights between each pair of nodes in the pooled subgraph, relative slower.
            if edge_attr is None:
                induced_edge_attr = torch.ones((induced_edge_index.size(1),), dtype=x.dtype,
                                               device=induced_edge_index.device)
            num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0)
            shift_cum_num_nodes = torch.cat([num_nodes.new_zeros(1), num_nodes.cumsum(dim=0)[:-1]], dim=0)
            cum_num_nodes = num_nodes.cumsum(dim=0)
            adj = torch.zeros((x.size(0), x.size(0)), dtype=torch.float, device=x.device)
            # Construct batch fully connected graph in block diagonal matirx format
            for idx_i, idx_j in zip(shift_cum_num_nodes, cum_num_nodes):
                adj[idx_i:idx_j, idx_i:idx_j] = 1.0
            new_edge_index, _ = dense_to_sparse(adj)
            row, col = new_edge_index

            weights = (torch.cat([x[row], x[col]], dim=1) * self.att).sum(dim=-1)
            weights = F.leaky_relu(weights, self.negative_slop)
            adj[row, col] = weights
            induced_row, induced_col = induced_edge_index

            adj[induced_row, induced_col] += induced_edge_attr * self.lamb
            weights = adj[row, col]
            if self.sparse:
                new_edge_attr = self.sparse_attention(weights, row)
            else:
                new_edge_attr = softmax(weights, row, x.size(0))
            # filter out zero weight edges
            adj[row, col] = new_edge_attr
            new_edge_index, new_edge_attr = dense_to_sparse(adj)
            # release gpu memory
            del adj
            torch.cuda.empty_cache()

        return x, new_edge_index, new_edge_attr, batch
