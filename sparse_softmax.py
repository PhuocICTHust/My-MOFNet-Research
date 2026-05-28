"""
An original implementation of sparsemax (Martins & Astudillo, 2016) is available at
https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/modules/sparse_activations.py.
See `From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification, ICML 2016`
for detailed description.

We make some modifications to make it work at scatter operation scenarios, e.g., calculate softmax according to batch
indicators.

Usage:
>> x = torch.tensor([ 1.7301,  0.6792, -1.0565,  1.6614, -0.3196, -0.7790, -0.3877, -0.4943,
         0.1831, -0.0061])
>> batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
>> sparse_attention = Sparsemax()
>> res = sparse_attention(x, batch)
>> print(res)
tensor([0.5343, 0.0000, 0.0000, 0.4657, 0.0612, 0.0000, 0.0000, 0.0000, 0.5640,
        0.3748])

"""
import torch
import torch.nn as nn
from torch.autograd import Function
from torch_scatter import scatter_add, scatter_max
import numpy as np


def scatter_sort(x, batch, fill_value=-1e16):
    num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0)
    batch_size, max_num_nodes = num_nodes.size(0), num_nodes.max().item() # batch_size=2,有两类， max_num_nodes=6拥有最多点的那类有6个点

    cum_num_nodes = torch.cat([num_nodes.new_zeros(1), num_nodes.cumsum(dim=0)[:-1]], dim=0)  # cum_num_nodes = tensor([0, 4])

    index = torch.arange(batch.size(0), dtype=torch.long, device=x.device)  # index = tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    index = (index - cum_num_nodes[batch]) + (batch * max_num_nodes)  # index = tensor([ 0,  1,  2,  3,  6,  7,  8,  9, 10, 11])其中：cum_num_nodes[batch]=tensor([0, 0, 0, 0, 4, 4, 4, 4, 4, 4])，index - cum_num_nodes[batch]=tensor([0, 1, 2, 3, 2, 3, 4, 5, 6, 7])

    dense_x = x.new_full((batch_size * max_num_nodes,), fill_value)  # dense_x = tensor([-1.0000e+16, -1.0000e+16, 一共12个）

    dense_x[index] = x  # tensor([ 0.0000e+00, -1.0509e+00, -2.7866e+00, -6.8700e-02, -1.0000e+16,-1.0000e+16, -5.0270e-01, -9.6210e-01, -5.7080e-01, -6.7740e-01,0.0000e+00, -1.8920e-01])
    dense_x = dense_x.view(batch_size, max_num_nodes)#tensor([[ 0.0000e+00, -1.0509e+00, -2.7866e+00, -6.8700e-02, -1.0000e+16,-1.0000e+16],[-5.0270e-01, -9.6210e-01, -5.7080e-01, -6.7740e-01,  0.0000e+00,-1.8920e-01]])

    sorted_x, _ = dense_x.sort(dim=-1, descending=True)
    cumsum_sorted_x = sorted_x.cumsum(dim=-1)
    cumsum_sorted_x = cumsum_sorted_x.view(-1)

    sorted_x = sorted_x.view(-1)
    filled_index = sorted_x != fill_value

    sorted_x = sorted_x[filled_index]
    cumsum_sorted_x = cumsum_sorted_x[filled_index]

    return sorted_x, cumsum_sorted_x  # sorted_x就是返回的每一组的降序排列，cumsum_sorted_x返回的是累计的降序排列。


def _make_ix_like(batch):
    num_nodes = scatter_add(batch.new_ones(batch.size(0)), batch, dim=0)
    idx = [torch.arange(1, i + 1, dtype=torch.long, device=batch.device) for i in num_nodes]
    idx = torch.cat(idx, dim=0)

    return idx


def _threshold_and_support(x, batch):
    """Sparsemax building block: compute the threshold
    Args:
        x: input tensor to apply the sparsemax
        batch: group indicators
    Returns:
        the threshold value
    """
    num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0)#把同一类的加起来=tensor([4, 6])
    cum_num_nodes = torch.cat([num_nodes.new_zeros(1), num_nodes.cumsum(dim=0)[:-1]], dim=0)#参数axis=0指的是按行累加，即本行=本行+上一行cum_num_nodes=tensor([0, 4])

    sorted_input, input_cumsum = scatter_sort(x, batch)  # sorted_input就是返回的每一组的降序排列；input_cumsum返回的是每一组累计的降序排列。
    input_cumsum = input_cumsum - 1.0
    rhos = _make_ix_like(batch).to(x.dtype)# =tensor([1., 2., 3., 4., 1., 2., 3., 4., 5., 6.])
    support = rhos * sorted_input > input_cumsum#support开始就不对了，有37742个true。 = tensor([ True,  True, False, False,  True,  True,  True, False, False, False])

    support_size = scatter_add(support.to(batch.dtype), batch)  # = tensor([2, 3])可能是有几个true
    # mask invalid index, for example, if batch is not start from 0 or not continuous, it may result in negative index
    idx = support_size + cum_num_nodes - 1 # = tensor([1, 6])好像是分组求每组最后一个true的位置,相当于ρ？
    mask = idx < 0  # = tensor([False, False])
    idx[mask] = 0  # tensor([1, 6])
    tau = input_cumsum.gather(0, idx)  # tensor([-1.0687, -1.6919])，idx=【1,6】。也就是第0维的第1个和第6个数，相当于input_comsum（【1】，）、（【6】，）
    tau /= support_size.to(x.dtype)  # tensor([-0.5344, -0.5640])，即[-1.0687, -1.6919]/[2,3].support_size求和就是37742

    return tau, support_size


class SparsemaxFunction(Function):

    @staticmethod
    def forward(ctx, x, batch):
        """sparsemax: normalizing sparse transform
        Parameters:
            ctx: context object
            x (Tensor): shape (N, )
            batch: group indicator
        Returns:
            output (Tensor): same shape as input
        """
        max_val, _ = scatter_max(x, batch)#分组求最大值，按照batch的分组，前四个为一组，后六个一组。max_val=tensor([1.7301, 0.1831])
        x -= max_val[batch]  # x = tensor([ 0.0000, -1.0509, -2.7866, -0.0687, -0.5027, -0.9621, -0.5708, -0.6774, 0.0000, -0.1892])
        tau, supp_size = _threshold_and_support(x, batch)
        output = torch.clamp(x - tau[batch], min=0)  # clamp将输入input张量每个元素的夹紧到区间 [min,max][min,max]，并返回结果到一个新张量。
        ctx.save_for_backward(supp_size, output, batch)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        supp_size, output, batch = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[output == 0] = 0

        v_hat = scatter_add(grad_input, batch) / supp_size.to(output.dtype)
        grad_input = torch.where(output != 0, grad_input - v_hat[batch], grad_input)

        return grad_input, None


sparsemax = SparsemaxFunction.apply


class Sparsemax(nn.Module):

    def __init__(self):
        super(Sparsemax, self).__init__()

    def forward(self, x, batch):
        return sparsemax(x, batch)


if __name__ == '__main__':
    sparse_attention = Sparsemax()
    input_x = torch.tensor([1.7301, 0.6792, -1.0565, 1.6614, -0.3196, -0.7790, -0.3877, -0.4943, 0.1831, -0.0061])
    input_batch = torch.cat([torch.zeros(4, dtype=torch.long), torch.ones(6, dtype=torch.long)], dim=0)
    res = sparse_attention(input_x, input_batch)
    print(res)


