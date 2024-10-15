from collections import OrderedDict
from copy import deepcopy
import itertools
import numpy as np
import os.path as osp
import pickle
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
import sklearn
from sklearn.manifold import TSNE
from torch_scatter import scatter_add

import torch
from torch.nn import Parameter, Linear
import torch.nn.functional as F
from torch.distributions.normal import Normal
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions.multivariate_normal import MultivariateNormal

import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
import torch_geometric.transforms as T
from torch_geometric.utils import add_remaining_self_loops, remove_self_loops, add_self_loops, softmax, degree, to_undirected
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
from numbers import Number

def get_reparam_num_neurons(out_channels, reparam_mode):
    if reparam_mode is None or reparam_mode == "None":
        return out_channels
    elif reparam_mode == "diag":
        return out_channels * 2
    elif reparam_mode == "full":
        return int((out_channels + 3) * out_channels / 2)
    else:
        raise "reparam_mode {} is not valid!".format(reparam_mode)
def set_cuda(tensor, is_cuda):
    if isinstance(is_cuda, str):
        return tensor.cuda(is_cuda)
    else:
        if is_cuda:
            return tensor.cuda()
        else:
            return tensor

def to_Variable(*arrays, **kwargs):
    """Transform numpy arrays into torch tensors/Variables"""
    is_cuda = kwargs["is_cuda"] if "is_cuda" in kwargs else False
    requires_grad = kwargs["requires_grad"] if "requires_grad" in kwargs else False
    array_list = []
    for array in arrays:
        is_int = False
        if isinstance(array, Number):
            is_int = True if isinstance(array, int) else False
            array = [array]
        if isinstance(array, np.ndarray) or isinstance(array, list) or isinstance(array, tuple):
            is_int = True if np.array(array).dtype.name == "int64" else False
            array = torch.tensor(array).float()
        if isinstance(array, torch.FloatTensor) or isinstance(array, torch.LongTensor) or isinstance(array, torch.ByteTensor):
            array = Variable(array, requires_grad=requires_grad)
        if "preserve_int" in kwargs and kwargs["preserve_int"] is True and is_int:
            array = array.long()
        array = set_cuda(array, is_cuda)
        array_list.append(array)
    if len(array_list) == 1:
        array_list = array_list[0]
    return array_list


def to_Variable_recur(item, type='float'):
    """Recursively transform numpy array into PyTorch tensor."""
    if isinstance(item, dict):
        return {key: to_Variable_recur(value, type=type) for key, value in item.items()}
    elif isinstance(item, tuple):
        return tuple(to_Variable_recur(element, type=type) for element in item)
    else:
        try:
            if type == "long":
                return torch.LongTensor(item)
            elif type == "float":
                return torch.FloatTensor(item)
            elif type == "bool":
                return torch.BoolTensor(item)
        except:
            return [to_Variable_recur(element, type=type) for element in item]


class Mixture_Gaussian_reparam(nn.Module):
    def __init__(
        self,
        # Use as reparamerization:
        mean_list=None,
        scale_list=None,
        weight_logits=None,
        # Use as prior:
        Z_size=None,
        n_components=None,
        mean_scale=0.1,
        scale_scale=0.1,
        # Mode:
        is_reparam=True,
        reparam_mode="diag",
        is_cuda=False,
    ):
        super(Mixture_Gaussian_reparam, self).__init__()
        self.is_reparam = is_reparam
        self.reparam_mode = reparam_mode
        self.is_cuda = is_cuda
        self.device = torch.device(self.is_cuda if isinstance(self.is_cuda, str) else "cuda" if self.is_cuda else "cpu")

        if self.is_reparam:
            self.mean_list = mean_list         # size: [B, Z, k]
            self.scale_list = scale_list       # size: [B, Z, k]
            self.weight_logits = weight_logits # size: [B, k]
            self.n_components = self.weight_logits.shape[-1]
            self.Z_size = self.mean_list.shape[-2]
        else:
            self.n_components = n_components
            self.Z_size = Z_size
            self.mean_list = nn.Parameter((torch.rand(1, Z_size, n_components) - 0.5) * mean_scale)
            self.scale_list = nn.Parameter(torch.log(torch.exp((torch.rand(1, Z_size, n_components) * 0.2 + 0.9) * scale_scale) - 1))
            self.weight_logits = nn.Parameter(torch.zeros(1, n_components))
            if mean_list is not None:
                self.mean_list.data = to_Variable(mean_list)
                self.scale_list.data = to_Variable(scale_list)
                self.weight_logits.data = to_Variable(weight_logits)

        self.to(self.device)


    def log_prob(self, input):
        """Obtain the log_prob of the input."""
        input = input.unsqueeze(-1)  # [S, B, Z, 1]
        if self.reparam_mode == "diag":
            if self.is_reparam:
                # logits: [S, B, Z, k]
                logits = - (input - self.mean_list) ** 2 / 2 / self.scale_list ** 2 - torch.log(self.scale_list * np.sqrt(2 * np.pi))
            else:
                scale_list = F.softplus(self.scale_list, beta=1)
                logits = - (input - self.mean_list) ** 2 / 2 / scale_list ** 2 - torch.log(scale_list * np.sqrt(2 * np.pi))
        else:
            raise
        # log_softmax(weight_logits): [B, k]
        # logits: [S, B, Z, k]
        # log_prob: [S, B, Z]
        log_prob = torch.logsumexp(logits + F.log_softmax(self.weight_logits, -1).unsqueeze(-2), axis=-1)  # F(...).unsqueeze(-2): [B, 1, k]
        return log_prob


    def prob(self, Z):
        return torch.exp(self.log_prob(Z))


    def sample(self, n=None):
        if n is None:
            n_core = 1
        else:
            assert isinstance(n, tuple)
            n_core = n[0]
        weight_probs = F.softmax(self.weight_logits, -1)  # size: [B, m]
        idx = torch.multinomial(weight_probs, n_core, replacement=True).unsqueeze(-2).expand(-1, self.mean_list.shape[-2], -1)  # multinomial result: [B, S]; result: [B, Z, S]
        mean_list  = torch.gather(self.mean_list,  dim=-1, index=idx)  # [B, Z, S]
        if self.is_reparam:
            scale_list = torch.gather(self.scale_list, dim=-1, index=idx)  # [B, Z, S]
        else:
            scale_list = F.softplus(torch.gather(self.scale_list, dim=-1, index=idx), beta=1)  # [B, Z, S]
        Z = torch.normal(mean_list, scale_list).permute(2, 0, 1)
        if n is None:
            Z = Z.squeeze(0)
        return Z


    def rsample(self, n=None):
        return self.sample(n=n)


    def __repr__(self):
        return "Mixture_Gaussian_reparam({}, Z_size={})".format(self.n_components, self.Z_size)


    @property
    def model_dict(self):
        model_dict = {"type": "Mixture_Gaussian_reparam"}
        model_dict["is_reparam"] = self.is_reparam
        model_dict["reparam_mode"] = self.reparam_mode
        model_dict["Z_size"] = self.Z_size
        model_dict["n_components"] = self.n_components
        model_dict["mean_list"] = to_np_array(self.mean_list)
        model_dict["scale_list"] = to_np_array(self.scale_list)
        model_dict["weight_logits"] = to_np_array(self.weight_logits)
        return model_dict




def reparameterize_diagonal(model, input, mode):
    if model is not None:
        mean_logit = model(input)
    else:
        mean_logit = input
    if mode.startswith("diagg"):
        if isinstance(mean_logit, tuple):
            mean = mean_logit[0]
        else:
            mean = mean_logit
        std = torch.ones(mean.shape).to(mean.device)
        dist = Normal(mean, std)
        return dist, (mean, std)
    elif mode.startswith("diag"):
        if isinstance(mean_logit, tuple):
            mean_logit = mean_logit[0]
        size = int(mean_logit.size(-1) / 2)
        mean = mean_logit[:, :size]
        std = F.softplus(mean_logit[:, size:], beta=1) + 1e-10
        dist = Normal(mean, std)
        return dist, (mean, std)
    else:
        raise Exception("mode {} is not valid!".format(mode))


def reparameterize_mixture_diagonal(model, input, mode):
    mean_logit, weight_logits = model(input)
    if mode.startswith("diagg"):
        mean_list = mean_logit
        scale_list = torch.ones(mean_list.shape).to(mean_list.device)
    else:
        size = int(mean_logit.size(-2) / 2)
        mean_list = mean_logit[:, :size]
        scale_list = F.softplus(mean_logit[:, size:], beta=1) + 0.01  # Avoid the std to go to 0
    dist = Mixture_Gaussian_reparam(mean_list=mean_list,
                                    scale_list=scale_list,
                                    weight_logits=weight_logits,
                                   )
    return dist, (mean_list, scale_list)

def fill_triangular(vec, dim, mode = "lower"):
    """Fill an lower or upper triangular matrices with given vectors"""
    num_examples, size = vec.shape
    assert size == dim * (dim + 1) // 2
    matrix = torch.zeros(num_examples, dim, dim).to(vec.device)
    idx = (torch.tril(torch.ones(dim, dim)) == 1).unsqueeze(0)
    idx = idx.repeat(num_examples,1,1)
    if mode == "lower":
        matrix[idx] = vec.contiguous().view(-1)
    elif mode == "upper":
        matrix[idx] = vec.contiguous().view(-1)
    else:
        raise Exception("mode {} not recognized!".format(mode))
    return matrix
def matrix_diag_transform(matrix, fun):
    """Return the matrices whose diagonal elements have been executed by the function 'fun'."""
    num_examples = len(matrix)
    idx = torch.eye(matrix.size(-1)).bool().unsqueeze(0)
    idx = idx.repeat(num_examples, 1, 1)
    new_matrix = matrix.clone()
    new_matrix[idx] = fun(matrix.diagonal(dim1 = 1, dim2 = 2).contiguous().view(-1))
    return new_matrix
def reparameterize_full(model, input, size=None):
    if model is not None:
        mean_logit = model(input)
    else:
        mean_logit = input
    if isinstance(mean_logit, tuple):
        mean_logit = mean_logit[0]
    if size is None:
        dim = mean_logit.size(-1)
        size = int((np.sqrt(9 + 8 * dim) - 3) / 2)
    mean = mean_logit[:, :size]
    scale_tril = fill_triangular(mean_logit[:, size:], size)
    scale_tril = matrix_diag_transform(scale_tril, F.softplus)
    dist = MultivariateNormal(mean, scale_tril = scale_tril)
    return dist, (mean, scale_tril)
def reparameterize(model, input, mode="full", size=None):
    if mode.startswith("diag"):
        if model is not None and model.__class__.__name__ == "Mixture_Model":
            return reparameterize_mixture_diagonal(model, input, mode=mode)
        else:
            return reparameterize_diagonal(model, input, mode=mode)
    elif mode == "full":
        return reparameterize_full(model, input, size=size)
    else:
        raise Exception("Mode {} is not valid!".format(mode))


def sample(dist, n=None):
    """Sample n instances from distribution dist"""
    if n is None:
        return dist.rsample()
    else:
        return dist.rsample((n,))


def to_np_array(*arrays, **kwargs):
    array_list = []
    for array in arrays:
        if isinstance(array, Variable):
            if array.is_cuda:
                array = array.cpu()
            array = array.data
        if isinstance(array, torch.Tensor) or isinstance(array, torch.FloatTensor) or isinstance(array, torch.LongTensor) or isinstance(array, torch.ByteTensor) or \
           isinstance(array, torch.cuda.FloatTensor) or isinstance(array, torch.cuda.LongTensor) or isinstance(array, torch.cuda.ByteTensor):
            if array.is_cuda:
                array = array.cpu()
            array = array.numpy()
        if isinstance(array, Number):
            pass
        elif isinstance(array, list) or isinstance(array, tuple):
            array = np.array(array)
        elif array.shape == (1,):
            if "full_reduce" in kwargs and kwargs["full_reduce"] is False:
                pass
            else:
                array = array[0]
        elif array.shape == ():
            array = array.tolist()
        array_list.append(array)
    if len(array_list) == 1:
        array_list = array_list[0]
    return array_list

class GCNConv(MessagePassing):
    r"""The graph convolutional operator from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        normalize (bool, optional): Whether to add self-loops and apply
            symmetric normalization. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, in_channels, out_channels, improved=False, cached=False,
                 bias=True, normalize=True,
                 reparam_mode=None, prior_mode=None, sample_size=1,
                 val_use_mean=True,
                 **kwargs):
        super(GCNConv, self).__init__(aggr='add', **kwargs)

        self.reparam_mode = None if reparam_mode == "None" else reparam_mode
        self.prior_mode = prior_mode
        self.val_use_mean = val_use_mean
        self.sample_size = sample_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out_neurons = get_reparam_num_neurons(out_channels,
                                                   self.reparam_mode)
        self.improved = improved
        self.cached = cached
        self.normalize = normalize

        self.weight = Parameter(torch.Tensor(in_channels, self.out_neurons))

        if bias:
            self.bias = Parameter(torch.Tensor(self.out_neurons))
        else:
            self.register_parameter('bias', None)

        if self.reparam_mode is not None:
            if self.prior_mode.startswith("mixGau"):
                n_components = eval(self.prior_mode.split("-")[1])
                self.feature_prior = Mixture_Gaussian_reparam(is_reparam=False,
                                                              Z_size=self.out_channels,
                                                              n_components=n_components)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self.cached_result = None
        self.cached_num_edges = None

    def set_cache(self, cached):
        self.cached = cached

    def to_device(self, device):
        self.to(device)
        if self.cached and self.cached_result is not None:
            edge_index, norm = self.cached_result
            self.cached_result = edge_index.to(device), norm.to(device)
        return self

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, improved=False,
             dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        x = torch.matmul(x, self.weight)

        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}. Please '
                    'disable the caching behavior of this layer by removing '
                    'the `cached=True` argument in its constructor.'.format(
                        self.cached_num_edges, edge_index.size(1)))

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            if self.normalize:
                edge_index, norm = self.norm(edge_index, x.size(
                    self.node_dim), edge_weight, self.improved, x.dtype)
            else:
                norm = edge_weight
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result
        out = self.propagate(edge_index, x=x, norm=norm)

        if self.reparam_mode is not None:
            # Reparameterize:
            self.dist, _ = reparameterize(model=None, input=out,
                                          mode=self.reparam_mode,
                                          size=self.out_channels
                                          )  # [B, Z]
            Z = sample(self.dist, self.sample_size)  # [S, B, Z]

            if self.prior_mode == "Gaussian":
                self.feature_prior = Normal(
                    loc=torch.zeros(x.size(0), self.out_channels).to(x.device),
                    scale=torch.ones(x.size(0), self.out_channels).to(
                        x.device),
                    )  # [B, Z]

            # Calculate prior loss:
            if self.reparam_mode == "diag" and self.prior_mode == "Gaussian":
                ixz = torch.distributions.kl.kl_divergence(self.dist,
                                                           self.feature_prior).sum(
                    -1)
            else:
                Z_logit = self.dist.log_prob(Z).sum(
                    -1) if self.reparam_mode.startswith(
                    "diag") else self.dist.log_prob(Z)  # [S, B]
                prior_logit = self.feature_prior.log_prob(Z).sum(-1)  # [S, B]
                # upper bound of I(X; Z):
                ixz = (Z_logit - prior_logit).mean(0)  # [B]

            self.Z_std = to_np_array(Z.std((0, 1)).mean())
            if self.val_use_mean is False or self.training:
                out = Z.mean(0)  # [B, Z]
            else:
                out = out[:, :self.out_channels]  # [B, Z]
        else:
            ixz = torch.zeros(x.size(0)).to(x.device)  # [B]

        structure_kl_loss = torch.zeros([]).to(x.device)
        return out, ixz, structure_kl_loss

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j if norm is not None else x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
def record_data(data_record_dict, data_list, key_list, nolist=False, ignore_duplicate=False):
    """Record data to the dictionary data_record_dict. It records each key: value pair in the corresponding location of
    key_list and data_list into the dictionary."""
    if not isinstance(data_list, list):
        data_list = [data_list]
    if not isinstance(key_list, list):
        key_list = [key_list]
    assert len(data_list) == len(key_list), "the data_list and key_list should have the same length!"
    for data, key in zip(data_list, key_list):
        if nolist:
            data_record_dict[key] = data
        else:
            if key not in data_record_dict:
                data_record_dict[key] = [data]
            else:
                if (not ignore_duplicate) or (data not in data_record_dict[key]):
                    data_record_dict[key].append(data)

class GIBGNN(torch.nn.Module):
    def __init__(
            self,
            model_type,
            num_features,
            num_classes,
            reparam_mode,
            prior_mode,
            latent_size,
            sample_size=1,
            num_layers=2,
            struct_dropout_mode=("standard", 0.6),
            dropout=True,
            with_relu=True,
            val_use_mean=True,
            reparam_all_layers=True,
            normalize=True,
            is_cuda=False,
    ):
        """Class implementing a general GNN, which can realize GAT, GIB-GAT, GCN.

        Args:
            model_type:   name of the base model. Choose from "GAT", "GCN".
            num_features: number of features of the data.x.
            num_classes:  number of classes for data.y.
            reparam_mode: reparameterization mode for XIB. Choose from "diag" and "full". Default "diag" that parameterizes the mean and diagonal element of the Gaussian
            prior_mode:   distribution type for the prior. Choose from "Gaussian" or "mixGau-{Number}", where {Number} is the number of components for mixture of Gaussian.
            latent_size:  latent size for each layer of GNN. If model_type="GAT", the true latent size is int(latent_size/2)
            sample_size=1:how many Z to sample for each feature X.
            num_layers=2: number of layers for the GNN
            struct_dropout_mode: Mode for how the structural representation is generated. Only effective for model_type=="GAT"
                          Choose from ("Nsampling", 'multi-categorical-sum', 0.1, 3) (here 0.1 is temperature, k=3 is the number of sampled edges with replacement),
                          ("DNsampling", 'multi-categorical-sum', 0.1, 3, 2) (similar as above, with the local dependence range T=2)
                          ("standard", 0.6) (standard dropout used on the attention weights in GAT)
            dropout:      whether to use dropout on features.
            with_relu:    whether to use nonlinearity for GCN.
            val_use_mean: Whether during evaluation use the parameter value instead of sampling. If True, during evaluation,
                          XIB will use mean for prediction, and AIB will use the parameter of the categorical distribution for prediction.
            reparam_all_layers: Which layers to use XIB, e.g. (1,2,4). Default (-2,), meaning the second last layer. If True, use XIB for all layers.
            normalize:    whether to normalize for GCN (only effective for GCN)
            is_cuda:      whether to use CUDA, and if so, which GPU to use. Choose from False, True, "CUDA:{GPU_ID}", where {GPU_ID} is the ID for the CUDA.
        """
        super(GIBGNN, self).__init__()
        self.model_type = model_type
        self.num_features = num_features
        self.num_classes = num_classes
        self.normalize = normalize
        self.reparam_mode = reparam_mode
        self.prior_mode = prior_mode
        self.struct_dropout_mode = struct_dropout_mode
        self.dropout = dropout
        self.latent_size = latent_size
        self.sample_size = sample_size
        self.num_layers = num_layers
        self.with_relu = with_relu
        self.val_use_mean = val_use_mean
        self.reparam_all_layers = reparam_all_layers
        self.is_cuda = is_cuda
        self.device = torch.device(self.is_cuda if isinstance(self.is_cuda,
                                                              str) else "cuda" if self.is_cuda else "cpu")

        self.init()

    def init(self):
        """Initialize the layers for the GNN."""
        self.reparam_layers = []
        if self.model_type == "GCN":
            for i in range(self.num_layers):
                if self.reparam_all_layers is True:
                    is_reparam = True
                elif isinstance(self.reparam_all_layers, tuple):
                    reparam_all_layers = tuple(
                        [kk + self.num_layers if kk < 0 else kk for kk in
                         self.reparam_all_layers])
                    is_reparam = i in reparam_all_layers
                else:
                    raise
                if is_reparam:
                    self.reparam_layers.append(i)
                setattr(self, "conv{}".format(i + 1),
                        GCNConv(
                            self.num_features if i == 0 else self.latent_size,
                            self.latent_size if i != self.num_layers - 1 else self.num_classes,
                            cached=True,
                            reparam_mode=self.reparam_mode if is_reparam else None,
                            prior_mode=self.prior_mode if is_reparam else None,
                            sample_size=self.sample_size,
                            bias=True if self.with_relu else False,
                            val_use_mean=self.val_use_mean,
                            normalize=self.normalize,
                            ))
            # self.conv1 = ChebConv(self.num_features, 16, K=2)
            # self.conv2 = ChebConv(16, self.num_features, K=2)


        else:
            raise Exception(
                "Model_type {} is not valid!".format(self.model_type))

        self.reparam_layers = sorted(self.reparam_layers)

        if self.model_type == "GCN":
            if self.with_relu:
                reg_params = [
                    getattr(self, "conv{}".format(i + 1)).parameters() for i in
                    range(self.num_layers - 1)]
                self.reg_params = itertools.chain(*reg_params)
                self.non_reg_params = getattr(self, "conv{}".format(
                    self.num_layers)).parameters()
            else:
                self.reg_params = OrderedDict()
                self.non_reg_params = self.parameters()
        else:
            self.reg_params = self.parameters()
            self.non_reg_params = OrderedDict()
        self.to(self.device)

    def set_cache(self, cached):
        """Set cache for GCN."""
        for i in range(self.num_layers):
            if hasattr(getattr(self, "conv{}".format(i + 1)), "set_cache"):
                getattr(self, "conv{}".format(i + 1)).set_cache(cached)

    def to_device(self, device):
        """Send all the layers to the specified device."""
        for i in range(self.num_layers):
            getattr(self, "conv{}".format(i + 1)).to_device(device)
        self.to(device)
        return self

    def forward(self, feature, edge_index, edge_weight, record_Z=False, isplot=False):
        """Main forward function.

        Args:
            data: the pytorch-geometric data class.
            record_Z: whether to record the standard deviation for the representation Z.
            isplot:   whether to plot.

        Returns:
            x: output
            reg_info: other information or metrics.
        """
        reg_info = {}
        if self.model_type == "GCN":
            x, edge_index, edge_weight = feature, edge_index, edge_weight
            for i in range(self.num_layers - 1):
                layer = getattr(self, "conv{}".format(i + 1))
                x, ixz, structure_kl_loss = layer(x, edge_index, edge_weight)
                # Record:
                record_data(reg_info, [ixz, structure_kl_loss],
                            ["ixz_list", "structure_kl_list"])
                if layer.reparam_mode is not None:
                    record_data(reg_info, [layer.Z_std], ["Z_std"])
                if record_Z:
                    record_data(reg_info, [to_np_array(x)], ["Z_{}".format(i)],
                                nolist=True)
                if self.with_relu:
                    x = F.relu(x)
                    # self.plot(x, data.y, titles="Layer{}".format(i + 1),
                    #           isplot=isplot)
                    if self.dropout is True:
                        x = F.dropout(x, training=self.training)
            layer = getattr(self, "conv{}".format(self.num_layers))
            x, ixz, structure_kl_loss = layer(x, edge_index, edge_weight)
            # Record:
            record_data(reg_info, [ixz, structure_kl_loss],
                        ["ixz_list", "structure_kl_list"])
            if layer.reparam_mode is not None:
                record_data(reg_info, [layer.Z_std], ["Z_std"])
            if record_Z:
                record_data(reg_info, [to_np_array(x)],
                            ["Z_{}".format(self.num_layers - 1)], nolist=True)
            # self.plot(x, data.y, titles="Layer{}".format(self.num_layers),
            #           isplot=isplot)


        return x, reg_info

    def compute_metrics_fun(self, data, metrics, mask=None, mask_id=None):
        """Compute metrics for measuring clustering performance.
        Choices: "Silu", "CH", "DB".
        """
        _, info_dict = self(data, record_Z=True)
        y = to_np_array(data.y)
        info_metrics = {}
        if mask is not None:
            mask = to_np_array(mask)
            mask_id += "_"
        else:
            mask_id = ""
        for k in range(self.num_layers):
            if mask is not None:
                Z_i = info_dict["Z_{}".format(k)][mask]
                y_i = y[mask]
            else:
                Z_i = info_dict["Z_{}".format(k)]
                y_i = y
            for metric in metrics:
                if metric == "Silu":
                    score = sklearn.metrics.silhouette_score(Z_i, y_i,
                                                             metric='euclidean')
                elif metric == "DB":
                    score = sklearn.metrics.davies_bouldin_score(Z_i, y_i)
                elif metric == "CH":
                    score = sklearn.metrics.calinski_harabasz_score(Z_i, y_i)
                info_metrics["{}{}_{}".format(mask_id, metric, k)] = score
        return info_metrics



    @property
    def model_dict(self):
        """Record model_dict for saving."""
        model_dict = {}
        model_dict["model_type"] = self.model_type
        model_dict["num_features"] = self.num_features
        model_dict["num_classes"] = self.num_classes
        model_dict["normalize"] = self.normalize
        model_dict["reparam_mode"] = self.reparam_mode
        model_dict["prior_mode"] = self.prior_mode
        model_dict["struct_dropout_mode"] = self.struct_dropout_mode
        model_dict["dropout"] = self.dropout
        model_dict["latent_size"] = self.latent_size
        model_dict["sample_size"] = self.sample_size
        model_dict["num_layers"] = self.num_layers
        model_dict["with_relu"] = self.with_relu
        model_dict["val_use_mean"] = self.val_use_mean
        model_dict["reparam_all_layers"] = self.reparam_all_layers
        # model_dict["state_dict"] = to_cpu_recur(self.state_dict())
        return model_dict


def load_model_dict_GNN(model_dict, is_cuda=False):
    """Load the GNN model."""
    model = GIBGNN(
        model_type=model_dict["model_type"],
        num_features=model_dict["num_features"],
        num_classes=model_dict["num_classes"],
        normalize=model_dict["normalize"],
        reparam_mode=model_dict["reparam_mode"],
        prior_mode=model_dict["prior_mode"],
        struct_dropout_mode=model_dict["struct_dropout_mode"],
        dropout=model_dict["dropout"],
        latent_size=model_dict["latent_size"],
        sample_size=model_dict["sample_size"],
        num_layers=model_dict["num_layers"],
        with_relu=model_dict["with_relu"],
        val_use_mean=model_dict["val_use_mean"],
        reparam_all_layers=model_dict["reparam_all_layers"],
        is_cuda=is_cuda,
    )
    if "state_dict" in model_dict:
        model.load_state_dict(model_dict["state_dict"])
    return model

# Train and test functions:
def train_model(model, data, optimizer, loss_type, beta1=None, beta2=None):
    """Train the model for one epoch."""
    model.train()
    optimizer.zero_grad()
    logits, reg_info = model(data)
    if loss_type == 'sigmoid':
        loss = torch.nn.BCEWithLogitsLoss(reduction='mean')(logits[data.train_mask], data.y[data.train_mask])
    elif loss_type == 'softmax':
        loss = torch.nn.CrossEntropyLoss(reduction='mean')(logits[data.train_mask], data.y[data.train_mask])
    else:
        raise
    # Add IB loss:
    if beta1 is not None and beta1 != 0:
        ixz = torch.stack(reg_info["ixz_list"], 1).mean(0).sum()
        if model.struct_dropout_mode[0] == 'DNsampling' or (model.struct_dropout_mode[0] == 'standard' and len(model.struct_dropout_mode) == 3):
            ixz = ixz + torch.stack(reg_info["ixz_DN_list"], 1).mean(0).sum()
        loss = loss + ixz * beta1
    if beta2 is not None and beta2 != 0:
        structure_kl_loss = torch.stack(reg_info["structure_kl_list"]).mean()
        if model.struct_dropout_mode[0] == 'DNsampling' or (model.struct_dropout_mode[0] == 'standard' and len(model.struct_dropout_mode) == 3):
            structure_kl_loss = structure_kl_loss + torch.stack(reg_info["structure_kl_DN_list"]).mean()
        loss = loss + structure_kl_loss * beta2
    loss.backward()
    optimizer.step()