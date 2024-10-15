import os
import sys
# Solving the problem of not being able to import your own packages under linux
cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, cur_path + "/..")
print(cur_path)
import torch
import os.path as osp
from torch_geometric.datasets import Planetoid, Amazon, Coauthor,WikipediaNetwork,WebKB
import torch_geometric.transforms as T
from torch_geometric.utils import dropout_adj, index_to_mask, dropout_edge
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.utils import remove_self_loops, segregate_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils import to_scipy_sparse_matrix,add_random_edge
from torch_geometric.utils import to_dense_adj,to_edge_index
import random

import scipy.sparse as sp
from torch_geometric.utils import degree,homophily,scatter,to_dense_adj
from torch_geometric.typing import Adj, OptTensor, SparseTensor
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils.convert import from_scipy_sparse_matrix


def statistic_num_isolated_nodes(data):
    num_nodes = maybe_num_nodes(data.edge_index, data.num_nodes)
    edge_index, _ = remove_self_loops(data.edge_index)
    return num_nodes - torch.unique(edge_index.view(-1)).numel()

def statistic_num_components(data):
    adj = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes)
    num_components, component = sp.csgraph.connected_components(
            adj, connection='weak')

    output = torch.unique(torch.tensor(component), return_counts =True)

    return num_components, output[1]
def distribution_node_homophily(edge_index,y, batch= None,method='edge'):
    y = y.squeeze(-1) if y.dim() > 1 else y
    if isinstance(edge_index, SparseTensor):
        row, col, _ = edge_index.coo()
    else:
        row, col = edge_index
    if method == 'edge':
        out = torch.zeros(row.size(0), device=row.device)
        out[y[row] == y[col]] = 1.
        
        return out
        

    elif method == 'node':
        out = torch.zeros(row.size(0), device=row.device)
        out[y[row] == y[col]] = 1.
        out = scatter(out, col, 0, dim_size=y.size(0), reduce='mean')
        return out
    
        
def find_commin_neighbors_matrix(batch_data, 
                               num_nodes):
    adj = to_dense_adj(batch_data.edge_index, max_num_nodes=num_nodes)[0]
    adj1 = torch.index_select(adj, 0, batch_data.n_id[:batch_data.batch_size])
    adj2 = torch.index_select(adj1.T, 0, batch_data.n_id[:batch_data.batch_size])
    common_neighbors_tensor = []
    for i in range(adj1.shape[0]):
        common_neighbors_tensor.append(torch.sum(torch.logical_and(adj2[i], adj2), dim=1))

    common_neighbors_tensor = torch.stack(common_neighbors_tensor)
    mask = torch.eye(batch.batch_size, batch.batch_size).byte()
    common_neighbors_tensor.masked_fill_(mask, 0)
    
#     print(torch.topk(common_neighbors_tensor, 3))
    
    return common_neighbors_tensor



def split(y, num_classes, train_per_class=20, val_per_class=30):

    indices = []

    for i in range(num_classes):
        index = (y == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:train_per_class] for i in indices], dim=0)
    val_index = torch.cat([i[train_per_class:train_per_class+val_per_class] for i in indices], dim=0)
    test_index = torch.cat([i[train_per_class+val_per_class:] for i in indices], dim=0)

    return train_index, val_index, test_index

def get_feature_mask(rate, n_nodes, n_features):
    return torch.bernoulli(torch.Tensor([1 - rate]).repeat(n_nodes, n_features)).bool()

# def feature_mask(features, missing_rate):
#     mask = torch.rand(size=features.size())
#     mask = mask <= missing_rate
#     return mask
#
# def apply_feature_mask(features, mask):
#     # features[mask] = float('nan')
#     features[mask] = 0.0
#
def edge_delete(prob_del, adj, enforce_connected=False):
    rnd = np.random.RandomState(1234)
    adj= adj.toarray()
    del_adj = np.array(adj, dtype=np.float32)
    smpl = rnd.choice([0., 1.], p=[prob_del, 1. - prob_del], size=adj.shape) * np.triu(np.ones_like(adj), 1)
    smpl += smpl.transpose()
    del_adj *= smpl
    if enforce_connected:
        add_edges = 0
        for k, a in enumerate(del_adj):
            if not list(np.nonzero(a)[0]):
                prev_connected = list(np.nonzero(adj[k, :])[0])
                other_node = rnd.choice(prev_connected)
                del_adj[k, other_node] = 1
                del_adj[other_node, k] = 1
                add_edges += 1
    del_adj= sp.csr_matrix(del_adj)

    return del_adj


# def edge_add(prob_add, adj, enforce_connected=False):
#     rnd = np.random.RandomState(1234)
#     adj= adj.toarray()
#     add_adj = np.array(adj, dtype=np.float32)
#     smpl = rnd.choice([1., 0.], p=[prob_add, 1. - prob_add], size=adj.shape) * np.triu(np.ones_like(adj), 1)
#     smpl += smpl.transpose()
#     add_adj *= smpl
#     if enforce_connected:
#         add_edges = 0
#         for k, a in enumerate(add_adj):
#             if not list(np.nonzero(a)[0]):
#                 prev_connected = list(np.nonzero(adj[k, :])[0])
#                 other_node = rnd.choice(prev_connected)
#                 add_adj[k, other_node] = 1
#                 add_adj[other_node, k] = 1
#                 add_edges += 1
#     add_adj= sp.csr_matrix(add_adj)
#
#     return add_adj


def load_data(dset, missing_link=0.0, missing_feature=0.0, trial=1,
              split_datasets_type='geom-gcn', normalize_features=True):
    path = osp.join('/home/LAB/liudezhi/D2PT/', 'Data', dset)

    f = np.loadtxt(path + '/{}.feature'.format(dset), dtype=float)
    l = np.loadtxt(path + '/{}.label'.format(dset), dtype=int)
    test = np.loadtxt(path + '/{}test.txt'.format(trial - 1), dtype=int)
    train = np.loadtxt(path + '/{}train.txt'.format(trial - 1), dtype=int)
    val = np.loadtxt(path + '/{}val.txt'.format(trial - 1), dtype=int)
    # all_label = np.loadtxt(path + '/{}.label'.format(dset), dtype=int)
    features = sp.csr_matrix(f, dtype=np.float32)
    features = torch.FloatTensor(np.array(features.todense()))
    groundtruth_features = features.clone()
    new_features = features.clone()
    # mask = feature_mask(features, rate)
    # apply_feature_mask(features, mask)

    idx_test = test.tolist()
    idx_train = train.tolist()
    idx_val = val.tolist()

    idx_train = torch.LongTensor(idx_train)
    idx_test = torch.LongTensor(idx_test)
    idx_val = torch.LongTensor(idx_val)
    label = torch.LongTensor(np.array(l))
    # all_label = torch.LongTensor(np.array(all_label))

    # label_oneHot = torch.FloatTensor(to_categorical(l))

    struct_edges = np.genfromtxt(path + '/{}.edge'.format(dset),
                                 dtype=np.int32)
    sedges = np.array(list(struct_edges), dtype=np.int32).reshape(
        struct_edges.shape)
    sadj = sp.coo_matrix(
        (np.ones(sedges.shape[0]), (sedges[:, 0], sedges[:, 1])),
        shape=(features.shape[0], features.shape[0]), dtype=np.float32)
    sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)
    # groundtruth_adj = None
    groundtruth_adj = torch.from_numpy(sadj.todense())

    if missing_link > 0.0:
        print("raw edge num:", np.sum(sadj))

        sadj = edge_delete(missing_link, sadj)
        if missing_link * 2 >= 1.0:
            new_sadj = edge_delete(missing_link, sadj)

        else:
            new_sadj = edge_delete(missing_link * 2, sadj)

        # data.edge_index, _ = dropout_adj(data.edge_index, p=missing_link, force_undirected=True, num_nodes=data.num_nodes)
        print("masked edge num:", np.sum(sadj))
        print("masked new edge num:", np.sum(new_sadj))
    else:
        sadj = sadj
        new_sadj = sadj
        print("masked edge num:", np.sum(sadj))
        print("masked new edge num:", np.sum(new_sadj))
    if missing_feature > 0.0:
        print("raw feature value:", torch.sum(features))
        features_mask = torch.rand(size=features.size())
        features_mask = features_mask <= missing_feature

        new_features_mask = torch.rand(size=new_features.size())
        if missing_feature * 2 >= 1.0:
            new_features_mask = new_features_mask <= missing_feature

        else:
            new_features_mask = new_features_mask <= missing_feature * 2

        # feature_mask = get_feature_mask(rate=missing_feature, n_nodes=data.num_nodes,
        #                                 n_features=data.num_features)
        # data.x[~feature_mask] = 0.0 # float('nan')
        # torch.unique(torch.bernoulli(a), return_counts=True)
        features[features_mask] = 0.0
        new_features[new_features_mask] = 0.0
        # features[features_mask] = 0.0
        print("masked  feature value:", torch.sum(features))
        print("masked  new feature value:", torch.sum(new_features))
        features[features_mask] = float('nan')
        new_features[new_features_mask] = float('nan')

    else:
        features_mask = torch.zeros(size=features.size(), dtype=torch.bool)
        new_features_mask = torch.zeros(size=features.size(), dtype=torch.bool)
        features[features_mask] = 0.0
        new_features[new_features_mask] = 0.0
        print("masked  feature value:", torch.sum(features))
        print("masked  new feature value:", torch.sum(new_features))
        features[features_mask] = float('nan')
        new_features[new_features_mask] = float('nan')

        # features_mask = features_mask <= missing_feature
    features_t = ~features_mask
    features_unmask_train = features_t.clone()
    # 计算True的总数
    # Subproblem 2: Identify the values in matrix A that are equal to True
    true_indices = torch.nonzero(features_unmask_train)
    features_unmask_eval = torch.zeros_like(features_unmask_train,
                                            dtype=torch.bool)
    # Subproblem 3: Randomly select 30% of the identified values (True) and set them to False
    num_false = int(0.3 * len(true_indices))
    false_indices = random.sample(range(len(true_indices)), num_false)
    true_indices = true_indices[false_indices]
    features_unmask_train[true_indices[:, 0], true_indices[:, 1]] = False
    features_unmask_eval[true_indices[:, 0], true_indices[:, 1]] = True

    print(torch.sum(features_unmask_train), torch.sum(features_unmask_eval))

    edge_index, edge_weight = from_scipy_sparse_matrix(sadj)
    # if missing_link > 0.0:
    #     print("raw edge num:", np.sum(sadj))

    # edge_index, added_edges = add_random_edge(edge_index, p=missing_link,
    #                                           force_undirected=True,
    #                                           num_nodes=features.shape[1])
    #
    # edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float32)

    new_edge_index, new_edge_weight = from_scipy_sparse_matrix(new_sadj)

    # todo 可以考虑更换prefill的方法，如均值补全、KNN补全等
    features = torch.where(torch.isnan(features),
                            0.5 * torch.ones_like(features),
                           features)
    new_features = torch.where(torch.isnan(new_features),
                               0.5 * torch.ones_like(new_features),
                               new_features)
    data = Data(x=features, edge_index=edge_index, edge_weight=edge_weight,
                y=label)

    data.new_feature = new_features
    data.new_edge_index = new_edge_index
    data.new_edge_weight = new_edge_weight
    node_degree = degree(data.edge_index[0], data.num_nodes)
    data.node_degree = node_degree
    data.degree_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.degree_mask[node_degree < 2] = True
    # data.degree_mask = degree_mask
    data.train_mask = index_to_mask(idx_train, size=data.num_nodes)
    data.val_mask = index_to_mask(idx_val, size=data.num_nodes)
    data.test_mask = index_to_mask(idx_test, size=data.num_nodes)

    data.feature_mask = features_mask
    data.features_unmask_train = features_unmask_train
    data.features_unmask_eval = features_unmask_eval
    data.groundtruth_x = groundtruth_features
    data.groundtruth_adj = groundtruth_adj

    data.features_unmask_train_sample = data.features_unmask_train.clone()
    print("feature sample before: total num :{}, 1 num:{}".format(
        torch.sum(data.features_unmask_train).item(),
        torch.nonzero(data.x[data.features_unmask_train]).shape[0]))
    # tt = data.x[data.features_unmask_train].clone()
    # print(tt.shape)
    # zeros_indices = torch.nonzero(tt == 0.0)
    zeros_indices = torch.nonzero(data.x == 0)
    ones_indices = torch.nonzero(data.x == 1)
    # ones_indices = torch.nonzero(tt  == 1.0)
    # print()
    # Subproblem 3: Randomly select 30% of the identified values (1s) and set them to 0
    num_zeros = int(len(zeros_indices)) - 4 * int(len(ones_indices))
    zero_indices = random.sample(range(len(zeros_indices)), num_zeros)
    zeros_indices_t = zeros_indices[zero_indices]

    data.features_unmask_train_sample[
        zeros_indices_t[:, 0], zeros_indices_t[:, 1]] = False

    print("feature sample after: total num:{}, 1 num:{}".format(
        torch.sum(data.features_unmask_train_sample).item(),
        torch.nonzero(data.x[data.features_unmask_train_sample]).shape[0]))
    print(torch.unique(data.x[data.features_unmask_train_sample],
                       return_counts=True))

    # raw data
    data.raw_x = data.x.clone()
    data.raw_edge_index = data.edge_index.clone()
    data.raw_edge_weight = data.edge_weight.clone()

    data.raw_adj = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes,
                                edge_attr=data.edge_weight).squeeze()
    data.adj_mask = ~(data.groundtruth_adj == data.raw_adj)

    # data.adj_is_one = data.raw_adj ==1.0
    # data.raw_edge_weight = data.edge_weight.clone()
    # to_scipy_sparse_matrix
    # data.adj = to_torch_sparse_tensor(data.edge_index).to(device)
    # x = self.GCN_encoder(features_0, adj)
    # features_1 = torch.where(features == float('nan'), x, features)

    # Subproblem 2: Identify the values in matrix A that are equal to 1
    train_edge_index, edge_id = dropout_edge(data.edge_index, p=0.3,
                                             force_undirected=True)
    train_adj = to_dense_adj(train_edge_index,
                             max_num_nodes=data.num_nodes).squeeze()

    eval_adj = data.raw_adj - train_adj

    data.adj_unmask_train = data.raw_adj == train_adj
    data.adj_unmask_eval = data.raw_adj == eval_adj
    data.adj_mask = data.raw_adj != data.groundtruth_adj
    #
    # # Subproblem 2: Identify the values in matrix A that are equal to 1
    #
    # zeros_indices = torch.nonzero(train_adj == 0)
    # ones_indices = torch.nonzero(train_adj == 1)
    #
    # # Subproblem 3: Randomly select 30% of the identified values (1s) and set them to 0
    #
    # num_zeros = int(len(zeros_indices)) - int(len(ones_indices))
    # zero_indices = random.sample(range(len(zeros_indices)), num_zeros)
    # zeros_indices_t = zeros_indices[zero_indices]
    # data.adj_unmask_train_sample[zeros_indices_t[:, 0], zeros_indices_t[:, 1]] = False

    # print("edge sample after: total num in adj_unmask_train {},  1 num {} ".format(torch.sum(data.adj_unmask_train).item(),
    #       torch.nonzero(data.raw_adj[data.adj_unmask_train]).shape[0]))

    print("node: ",
          torch.sum(data.train_mask + data.val_mask + data.test_mask))
    print("train node/ val node/ test node:",
          torch.sum(data.train_mask), torch.sum(data.val_mask),
          torch.sum(data.test_mask))
    if normalize_features:
        data.transform = T.NormalizeFeatures()

    # meta = {'num_classes': dataset.num_classes}

    # return features,adj, label, idx_train, idx_val, idx_test

    return data


def load_data_0927(dset,missing_link=0.0, missing_feature=0.0, trial=1, split_datasets_type = 'geom-gcn', normalize_features=True):
    path = osp.join('/home/LAB/liudezhi/D2PT/', 'Data', dset)


    f = np.loadtxt(path + '/{}.feature'.format(dset), dtype=float)
    l = np.loadtxt(path + '/{}.label'.format(dset), dtype=int)
    test = np.loadtxt(path + '/{}test.txt'.format(trial-1), dtype=int)
    train = np.loadtxt(path + '/{}train.txt'.format(trial-1), dtype=int)
    val = np.loadtxt(path + '/{}val.txt'.format(trial-1), dtype=int)
    features = sp.csr_matrix(f, dtype=np.float32)
    features = torch.FloatTensor(np.array(features.todense()))
    groundtruth_features = features.clone()
    new_features = features.clone()
    # mask = feature_mask(features, rate)
    # apply_feature_mask(features, mask)

    idx_test = test.tolist()
    idx_train = train.tolist()
    idx_val = val.tolist()

    idx_train = torch.LongTensor(idx_train)
    idx_test = torch.LongTensor(idx_test)
    idx_val = torch.LongTensor(idx_val)
    label = torch.LongTensor(np.array(l))

    # label_oneHot = torch.FloatTensor(to_categorical(l))

    struct_edges = np.genfromtxt(path + '/{}.edge'.format(dset),
                                 dtype=np.int32)
    sedges = np.array(list(struct_edges), dtype=np.int32).reshape(
        struct_edges.shape)
    sadj = sp.coo_matrix(
        (np.ones(sedges.shape[0]), (sedges[:, 0], sedges[:, 1])),
        shape=(features.shape[0], features.shape[0]), dtype=np.float32)
    sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)
    # groundtruth_adj = None
    groundtruth_adj = torch.from_numpy(sadj.todense())

    groundtruth_edge_index,groundtruth_edge_weight = from_scipy_sparse_matrix(sadj)



    if missing_link > 0.0:
        print("raw edge num:",np.sum(sadj))

        # sadj = edge_delete(missing_link/2.0, sadj)
        sadj = edge_delete(missing_link, sadj)
        # sadj = edge_add(missing_link, sadj)
        if missing_link*2 >= 1.0:
            new_sadj = edge_delete(missing_link, sadj)

        else:
            new_sadj = edge_delete(missing_link*2, sadj)

        # data.edge_index, _ = dropout_adj(data.edge_index, p=missing_link, force_undirected=True, num_nodes=data.num_nodes)
        print("masked edge num:",np.sum(sadj))
        print("masked new edge num:",np.sum(new_sadj))
    else:
        sadj = sadj
        new_sadj= sadj
        print("masked edge num:", np.sum(sadj))
        print("masked new edge num:", np.sum(new_sadj))
    if missing_feature > 0.0:
        print("raw feature value:",torch.sum(features))
        features_mask = torch.rand(size=features.size())
        features_mask = features_mask <= missing_feature

        new_features_mask = torch.rand(size=new_features.size())
        if missing_feature*2 >= 1.0:
            new_features_mask = new_features_mask <= missing_feature

        else:
            new_features_mask = new_features_mask <= missing_feature*2

        # feature_mask = get_feature_mask(rate=missing_feature, n_nodes=data.num_nodes,
        #                                 n_features=data.num_features)
        # data.x[~feature_mask] = 0.0 # float('nan')
        # torch.unique(torch.bernoulli(a), return_counts=True)
        features[features_mask] = 0.0
        new_features[new_features_mask] = 0.0
        # features[features_mask] = 0.0
        print("masked  feature value:",torch.sum(features) )
        print("masked  new feature value:",torch.sum(new_features) )
        features[features_mask] = float('nan')
        new_features[new_features_mask] = float('nan')

    else:
        features_mask = torch.zeros(size=features.size(), dtype=torch.bool)
        new_features_mask = torch.zeros(size=features.size(), dtype=torch.bool)
        features[features_mask] = 0.0
        new_features[new_features_mask] = 0.0
        print("masked  feature value:", torch.sum(features))
        print("masked  new feature value:", torch.sum(new_features))
        features[features_mask] = float('nan')
        new_features[new_features_mask] = float('nan')

        # features_mask = features_mask <= missing_feature
    features_t = ~features_mask
    features_unmask_train = features_t.clone()
    # 计算True的总数
    # Subproblem 2: Identify the values in matrix A that are equal to True
    true_indices = torch.nonzero(features_unmask_train)
    features_unmask_eval = torch.zeros_like(features_unmask_train,dtype=torch.bool)
    # Subproblem 3: Randomly select 30% of the identified values (True) and set them to False
    num_false = int(0.3 * len(true_indices))
    false_indices = random.sample(range(len(true_indices)), num_false)
    true_indices = true_indices[false_indices]
    features_unmask_train[true_indices[:, 0], true_indices[:, 1]] = False
    features_unmask_eval[true_indices[:, 0], true_indices[:, 1]] = True


    print(torch.sum(features_unmask_train),torch.sum(features_unmask_eval) )


    edge_index,edge_weight = from_scipy_sparse_matrix(sadj)
    # if missing_link > 0.0:
    #     print("raw edge num:", np.sum(sadj))

    # edge_index, added_edges = add_random_edge(edge_index, p=missing_link/2.0,force_undirected=True,num_nodes=features.shape[0])
    _, added_edges = add_random_edge(groundtruth_edge_index, p=missing_link,force_undirected=True,num_nodes=features.shape[0])
    edge_index = torch.cat([edge_index,added_edges],dim=1)
    edge_weight = torch.ones(edge_index.shape[1],dtype=torch.float32)

    new_edge_index,new_edge_weight = from_scipy_sparse_matrix(new_sadj)

    # todo 可以考虑更换prefill的方法，如均值补全、KNN补全等
    # feature_means = torch.mean(features,1,True) * torch.ones_like(features)
    # features = torch.where(torch.isnan(features),0.5 * torch.ones_like(features),features)
    features = torch.where(torch.isnan(features), 0.5*torch.ones_like(features),
                         features)
    new_features = torch.where(torch.isnan(new_features), 0.5*torch.ones_like(new_features),
                           new_features)
    data = Data(x=features, edge_index=edge_index,edge_weight =edge_weight, y=label)

    data.new_feature = new_features
    data.new_edge_index = new_edge_index
    data.new_edge_weight = new_edge_weight
    node_degree = degree(data.edge_index[0], data.num_nodes)
    data.degree_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.degree_mask[node_degree < 2] = True
    # data.degree_mask = degree_mask
    data.train_mask = index_to_mask(idx_train, size=data.num_nodes)
    data.val_mask = index_to_mask(idx_val, size=data.num_nodes)
    data.test_mask = index_to_mask(idx_test, size=data.num_nodes)

    data.feature_mask = features_mask
    data.features_unmask_train = features_unmask_train
    data.features_unmask_eval = features_unmask_eval
    data.groundtruth_x = groundtruth_features
    data.groundtruth_adj = groundtruth_adj

    data.features_unmask_train_sample = data.features_unmask_train.clone()
    print("feature sample before: total num :{}, 1 num:{}".format(torch.sum(data.features_unmask_train).item(),torch.nonzero(data.x[data.features_unmask_train]).shape[0]))
    # tt = data.x[data.features_unmask_train].clone()
    # print(tt.shape)
    # zeros_indices = torch.nonzero(tt == 0.0)
    zeros_indices = torch.nonzero(data.x == 0)
    ones_indices = torch.nonzero(data.x == 1)
    # ones_indices = torch.nonzero(tt  == 1.0)
    # print()
    # Subproblem 3: Randomly select 30% of the identified values (1s) and set them to 0
    num_zeros = int(len(zeros_indices)) - 4*int(len(ones_indices))
    zero_indices = random.sample(range(len(zeros_indices)), num_zeros)
    zeros_indices_t = zeros_indices[zero_indices]

    data.features_unmask_train_sample[zeros_indices_t[:, 0], zeros_indices_t[:, 1]] = False

    print("feature sample after: total num:{}, 1 num:{}".format( torch.sum(data.features_unmask_train_sample).item(),
          torch.nonzero(data.x[data.features_unmask_train_sample]).shape[0]))
    print(torch.unique(data.x[data.features_unmask_train_sample], return_counts=True))


    # raw data
    data.raw_x =  data.x.clone()
    data.raw_edge_index = data.edge_index.clone()
    data.raw_edge_weight = data.edge_weight.clone()

    data.raw_adj = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes,edge_attr=data.edge_weight).squeeze()
    data.adj_mask = ~(data.groundtruth_adj == data.raw_adj)

    # data.adj_is_one = data.raw_adj ==1.0
    # data.raw_edge_weight = data.edge_weight.clone()
    # to_scipy_sparse_matrix
    # data.adj = to_torch_sparse_tensor(data.edge_index).to(device)
    # x = self.GCN_encoder(features_0, adj)
    # features_1 = torch.where(features == float('nan'), x, features)

    # Subproblem 2: Identify the values in matrix A that are equal to 1
    train_edge_index, edge_id = dropout_edge(data.edge_index, p=0.3,
                                      force_undirected=True)
    train_adj = to_dense_adj(train_edge_index, max_num_nodes=data.num_nodes).squeeze()

    eval_adj =  data.raw_adj - train_adj

    data.adj_unmask_train_bak =  data.raw_adj == train_adj
    data.adj_unmask_eval_bak = data.raw_adj == eval_adj


    adj_mask_t = ~data.adj_mask
    adj_unmask_train = adj_mask_t.clone()
    # 计算True的总数
    # Subproblem 2: Identify the values in matrix A that are equal to True
    adj_true_indices = torch.nonzero(adj_unmask_train)
    adj_unmask_eval = torch.zeros_like(adj_unmask_train,
                                            dtype=torch.bool)
    # Subproblem 3: Randomly select 30% of the identified values (True) and set them to False
    adj_num_false = int(0.3 * len(adj_true_indices))
    adj_false_indices = random.sample(range(len(adj_true_indices)), adj_num_false)
    adj_true_indices = adj_true_indices[adj_false_indices]
    adj_unmask_train[adj_true_indices[:, 0], adj_true_indices[:, 1]] = False
    adj_unmask_eval[adj_true_indices[:, 0], adj_true_indices[:, 1]] = True

    data.adj_unmask_train = adj_unmask_train
    data.adj_unmask_eval = adj_unmask_eval





    #
    # # Subproblem 2: Identify the values in matrix A that are equal to 1
    #
    # zeros_indices = torch.nonzero(train_adj == 0)
    # ones_indices = torch.nonzero(train_adj == 1)
    #
    # # Subproblem 3: Randomly select 30% of the identified values (1s) and set them to 0
    #
    # num_zeros = int(len(zeros_indices)) - int(len(ones_indices))
    # zero_indices = random.sample(range(len(zeros_indices)), num_zeros)
    # zeros_indices_t = zeros_indices[zero_indices]
    # data.adj_unmask_train_sample[zeros_indices_t[:, 0], zeros_indices_t[:, 1]] = False


    # print("edge sample after: total num in adj_unmask_train {},  1 num {} ".format(torch.sum(data.adj_unmask_train).item(),
    #       torch.nonzero(data.raw_adj[data.adj_unmask_train]).shape[0]))



    print("node: ",
          torch.sum(data.train_mask + data.val_mask + data.test_mask))
    print("train node/ val node/ test node:",
          torch.sum(data.train_mask), torch.sum(data.val_mask),
          torch.sum(data.test_mask))
    if normalize_features:
        data.transform = T.NormalizeFeatures()

    # meta = {'num_classes': dataset.num_classes}


    # return features,adj, label, idx_train, idx_val, idx_test

    return data



def load_data_bak(dset, train_per_class=20, val_per_class=30,
              missing_link=0.0, missing_feature=0.0, trial=1, split_datasets_type = 'geom-gcn', ogb_train_ratio=0.01,
              normalize_features=True, use_public_split=True):
    if split_datasets_type == "geom-gcn":
        path = osp.join('.', 'Data', dset)
        if dset in ['cora', 'citeseer', 'pubmed']:
            dataset = Planetoid(path, dset, split=split_datasets_type)
        elif dset in ['cornell','texas','wisconsin']:
            dataset = WebKB(path, dset)
        elif dset in["chameleon", "squirrel"]:
            dataset = WikipediaNetwork(path, dset)
        # elif dset in ['photo', 'computers']:
        #     dataset = Amazon(path, dset)
        # elif dset in ['physics', 'cs']:
        #     dataset = Coauthor(path, dset)
        # elif dset in ['arxiv']:
        #     dataset = PygNodePropPredDataset('ogbn-'+dset, path)
        else:
            assert Exception
        data = dataset[0]
        # if dset in ['cora', 'citeseer', 'pubmed'] and use_public_split:
            # print('Using public split of {}! 20 per class/30 per class/1000 for train/val/test.'.format(dset))
        print('Using {} split of {}.'.format(split_datasets_type, dset))
        data.train_mask = data.train_mask[:, trial - 1]
        data.val_mask = data.val_mask[:, trial - 1]
        data.test_mask = data.test_mask[:, trial - 1]
        # data.train_mask =  data.train_mask[:, trial]
        # data.val_mask = data.val_mask[:, trial]
        # data.test_mask =data.test_mask[:, trial]
        print("mask_sum: ",
              torch.sum(data.train_mask + data.val_mask + data.test_mask))
        print("train mask/ val mask/ test mask:",
              torch.sum(data.train_mask), torch.sum(data.val_mask),
              torch.sum(data.test_mask))

            # assert  == data.num_nodes

        # elif dset not in ['arxiv']:
        #     train_index, val_index, test_index = split(data.y,
        #                                                dataset.num_classes,
        #                                                train_per_class,
        #                                                val_per_class)
        #     data.train_mask = index_to_mask(train_index, size=data.num_nodes)
        #     data.val_mask = index_to_mask(val_index, size=data.num_nodes)
        #     data.test_mask = index_to_mask(test_index, size=data.num_nodes)
        # else:
        #     ogb_split = dataset.get_idx_split()
        #     train_index, val_index, test_index = ogb_split['train'], ogb_split[
        #         'valid'], ogb_split['test']
        #     train_index = train_index[torch.randperm(train_index.size(0))]
        #     train_index = train_index[:int(data.num_nodes * ogb_train_ratio)]
        #     data.train_mask = index_to_mask(train_index, size=data.num_nodes)
        #     data.val_mask = index_to_mask(val_index, size=data.num_nodes)
        #     data.test_mask = index_to_mask(test_index, size=data.num_nodes)
        #     data.y = data.y.squeeze(1)
    elif split_datasets_type == 't2-gnn':

        path = osp.join('.', 'Data', dset)
        if dset in ['cora', 'citeseer', 'pubmed']:
            dataset = Planetoid(path, dset)
        elif dset in ['cornell', 'texas', 'wisconsin']:
            dataset = WebKB(path, dset)
        elif dset in ["chameleon", "squirrel"]:
            dataset = WikipediaNetwork(path, dset)
        # elif dset in ['photo', 'computers']:
        #     dataset = Amazon(path, dset)
        # elif dset in ['physics', 'cs']:
        #     dataset = Coauthor(path, dset)
        # elif dset in ['arxiv']:
        #     dataset = PygNodePropPredDataset('ogbn-' + dset, path)
        else:
            assert Exception

        data = dataset[0]
        # path1 = osp.join('.', 'Data', dset)
        test = np.loadtxt(path + '/{}test.txt'.format(trial-1), dtype=int)
        train = np.loadtxt(path + '/{}train.txt'.format(trial-1), dtype=int)
        val = np.loadtxt(path + '/{}val.txt'.format(trial-1), dtype=int)
        l = np.loadtxt(path + '/{}.label'.format(dset), dtype=int)
        label = torch.LongTensor(np.array(l))
        assert label.equal(data.y)
        # print("datset================",label.eq(data.y))
        # label_oneHot = torch.FloatTensor(to_categorical(l)).to(device)


        idx_test = test.tolist()
        idx_train = train.tolist()
        idx_val = val.tolist()

        idx_train = torch.LongTensor(idx_train)
        idx_test = torch.LongTensor(idx_test)
        idx_val = torch.LongTensor(idx_val)
        data.train_mask = index_to_mask(idx_train, size=data.num_nodes)
        data.val_mask = index_to_mask(idx_val, size=data.num_nodes)
        data.test_mask = index_to_mask(idx_test, size=data.num_nodes)

        print("mask_sum: ",
              torch.sum(data.train_mask + data.val_mask + data.test_mask))
        print("train mask/ val mask/ test mask:", torch.sum(data.train_mask),
              torch.sum(data.val_mask), torch.sum(data.test_mask))
    else:
        path = osp.join('.', 'Data', dset)
        if dset in ['cora', 'citeseer', 'pubmed']:
            dataset = Planetoid(path, dset, split=split_datasets_type)
        elif dset in ['cornell', 'texas', 'wisconsin']:
            dataset = WebKB(path, dset)
        elif dset in ["chameleon", "squirrel"]:
            dataset = WikipediaNetwork(path, dset)

        data = dataset[0]


    if normalize_features:
        data.transform = T.NormalizeFeatures()


    if missing_link > 0.0:
        print("raw edge num:",data.num_edges)
        data.edge_index, _ = dropout_adj(data.edge_index, p=missing_link, force_undirected=True, num_nodes=data.num_nodes)
        print("masked edge num:",data.num_edges)

    if missing_feature > 0.0:
        print("raw feature value:",torch.sum(data.x))
        feature_mask = get_feature_mask(rate=missing_feature, n_nodes=data.num_nodes,
                                        n_features=data.num_features)
        data.x[~feature_mask] = 0.0
        print("masked  feature value:",torch.sum(data.x))

    meta = {'num_classes': dataset.num_classes}

    return data, meta


def load_and_sta_data(dset, train_per_class=20, val_per_class=30,
              missing_link=0.0, missing_feature=0.0, ogb_train_ratio=0.01,
              normalize_features=True, use_public_split=False):
    sta_dic = {}
    sta_dic['dataset'] = dset

    path = osp.join('.', 'Data', dset)
    if dset in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(path, dset)
    if dset in ['photo', 'computers']:
        dataset = Amazon(path, dset)
    elif dset in ['physics', 'cs']:
        dataset = Coauthor(path, dset)
    elif dset in ['arxiv']:
        dataset = PygNodePropPredDataset('ogbn-'+dset, path)
    else:
        assert Exception

    data = dataset[0]
    sta_dic['num_nodes'] = data.num_nodes
    sta_dic['num_edges'] = data.num_edges / 2 
    sta_dic['num_isolated_nodes'] = statistic_num_isolated_nodes(data)
    sta_dic['num_connected_components'], sta_dic['dis_connected_components']= statistic_num_components(data)
    sta_dic['num_LCC_nodes'] = torch.max(sta_dic['dis_connected_components']).item()

    sta_dic['dis_degree']= degree(data.edge_index[0], data.num_nodes)
    sta_dic['X'] = data.x.clone()
    sta_dic['y'] = data.y.clone()
    sta_dic['node_homophily'] = homophily(data.edge_index, data.y, method='node')
    sta_dic['edge_homophily'] = homophily(data.edge_index, data.y, method='edge')
    sta_dic['class_homophily'] = homophily(data.edge_index, data.y, method='edge_insensitive')

    sta_dic['dis_node_homophily'] = distribution_node_homophily(data.edge_index,data.y, method = 'node')
    # sta_dic['common_neighbors'] = find_commin_neighbors_matrix(data, data.num_nodes)


    # data.x
    # data.edge_index
    # data.y
    # data.num_nodes
    # data.num_edges
    # data.is_directed()
    # data.has_isolated_nodes()


    if normalize_features:
        data.transform = T.NormalizeFeatures()

    if dset in ['cora', 'citeseer', 'pubmed'] and use_public_split:
        print('Using public split of {}! 20 per class/30 per class/1000 for train/val/test.'.format(dset))
    elif dset not in ['arxiv']:
        train_index, val_index, test_index = split(data.y, dataset.num_classes, train_per_class, val_per_class)
        data.train_mask = index_to_mask(train_index, size=data.num_nodes)
        data.val_mask = index_to_mask(val_index, size=data.num_nodes)
        data.test_mask = index_to_mask(test_index, size=data.num_nodes)
    else:
        ogb_split = dataset.get_idx_split()
        train_index, val_index, test_index = ogb_split['train'], ogb_split['valid'], ogb_split['test']
        train_index = train_index[torch.randperm(train_index.size(0))]
        train_index = train_index[:int(data.num_nodes * ogb_train_ratio)]
        data.train_mask = index_to_mask(train_index, size=data.num_nodes)
        data.val_mask = index_to_mask(val_index, size=data.num_nodes)
        data.test_mask = index_to_mask(test_index, size=data.num_nodes)
        data.y = data.y.squeeze(1)

    if missing_link > 0.0:
        print(data.num_edges)
        data.edge_index, _ = dropout_adj(data.edge_index, p=missing_link, force_undirected=True, num_nodes=data.num_nodes)
        print(data.num_edges)

    if missing_feature > 0.0:
        print(torch.sum(data.x))
        feature_mask = get_feature_mask(rate=missing_feature, n_nodes=data.num_nodes,
                                        n_features=data.num_features)
        data.x[~feature_mask] = 0.0
        print(torch.sum(data.x))

    meta = {'num_classes': dataset.num_classes}
    miss_rate = int(missing_link*100)

    sta_dic['{}_num_nodes'.format(miss_rate)] = data.num_nodes
    sta_dic['{}_num_edges'.format(miss_rate)] = data.num_edges / 2 
    sta_dic['{}_num_isolated_nodes'.format(miss_rate)] = statistic_num_isolated_nodes(data)
    sta_dic['{}_num_connected_components'.format(miss_rate)], sta_dic['{}_dis_connected_components'.format(miss_rate)] = statistic_num_components(data)
    sta_dic['{}_mum_LCC_nodes'.format(miss_rate)] = torch.max(sta_dic['{}_dis_connected_components'.format(miss_rate)]).item()

    sta_dic['{}_dis_degree'.format(miss_rate)]= degree(data.edge_index[0], data.num_nodes)
    sta_dic['{}_X'.format(miss_rate)] = data.x
    sta_dic['{}_y'.format(miss_rate)] = data.y
    sta_dic['{}_node_homophily'.format(miss_rate)] = homophily(data.edge_index, data.y, method ='node')
    sta_dic['{}_edge_homophily'.format(miss_rate)] = homophily(data.edge_index, data.y, method ='edge')
    sta_dic['{}_class_homophily'.format(miss_rate)] = homophily(data.edge_index, data.y, method ='edge_insensitive')

    sta_dic['{}_dis_node_homophily'.format(miss_rate)] = distribution_node_homophily(data.edge_index,data.y, method = 'node')

    return sta_dic







def load_and_sta_data_v2(dset, train_per_class=20, val_per_class=30,
              missing_link=0.0, missing_feature=0.0, ogb_train_ratio=0.01,
              normalize_features=True, use_public_split=False):
    path = osp.join('.', 'Data', dset)

    f = np.loadtxt(path + '/{}.feature'.format(dset), dtype=float)
    l = np.loadtxt(path + '/{}.label'.format(dset), dtype=int)
    test = np.loadtxt(path + '/{}test.txt'.format(1 - 1), dtype=int)
    train = np.loadtxt(path + '/{}train.txt'.format(1 - 1), dtype=int)
    val = np.loadtxt(path + '/{}val.txt'.format(1 - 1), dtype=int)
    features = sp.csr_matrix(f, dtype=np.float32)
    features = torch.FloatTensor(np.array(features.todense()))
    groundtruth_features = features.clone()

    # mask = feature_mask(features, rate)
    # apply_feature_mask(features, mask)

    idx_test = test.tolist()
    idx_train = train.tolist()
    idx_val = val.tolist()

    idx_train = torch.LongTensor(idx_train)
    idx_test = torch.LongTensor(idx_test)
    idx_val = torch.LongTensor(idx_val)
    label = torch.LongTensor(np.array(l))

    # label_oneHot = torch.FloatTensor(to_categorical(l))

    struct_edges = np.genfromtxt(path + '/{}.edge'.format(dset),
                                 dtype=np.int32)
    sedges = np.array(list(struct_edges), dtype=np.int32).reshape(
        struct_edges.shape)
    sadj = sp.coo_matrix(
        (np.ones(sedges.shape[0]), (sedges[:, 0], sedges[:, 1])),
        shape=(features.shape[0], features.shape[0]), dtype=np.float32)
    sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)
    # groundtruth_adj = None
    groundtruth_adj = torch.from_numpy(sadj.todense())
    edge_index, edge_weight = from_scipy_sparse_matrix(sp.csr_matrix(sadj))

    raw_data = Data(x=features, edge_index=edge_index, edge_weight=edge_weight,
                y=label)

    raw_data.train_mask = index_to_mask(idx_train, size=raw_data.num_nodes)
    raw_data.val_mask = index_to_mask(idx_val, size=raw_data.num_nodes)
    raw_data.test_mask = index_to_mask(idx_test, size=raw_data.num_nodes)

    sta_dic = {}

    sta_dic['num_nodes'] = raw_data.num_nodes
    sta_dic['feature_sum'] = torch.sum(raw_data.x).item()
    sta_dic['num_edges'] = raw_data.num_edges / 2
    sta_dic['num_isolated_nodes'] = statistic_num_isolated_nodes(raw_data)
    sta_dic['num_connected_components'], sta_dic[
        'dis_connected_components'] = statistic_num_components(raw_data)
    sta_dic['num_LCC_nodes'] = torch.max(
        sta_dic['dis_connected_components']).item()

    sta_dic['dis_degree'] = degree(raw_data.edge_index[0], raw_data.num_nodes)
    sta_dic['X'] = raw_data.x.clone()
    sta_dic['y'] = raw_data.y.clone()
    sta_dic['node_homophily'] = homophily(raw_data.edge_index, raw_data.y,
                                          method='node')
    sta_dic['edge_homophily'] = homophily(raw_data.edge_index, raw_data.y,
                                          method='edge')
    sta_dic['class_homophily'] = homophily(raw_data.edge_index, raw_data.y,
                                           method='edge_insensitive')

    sta_dic['dis_node_homophily'] = distribution_node_homophily(
        raw_data.edge_index, raw_data.y, method='node')



    if missing_link > 0.0:
        print("raw edge num:", np.sum(sadj))

        sadj = edge_delete(missing_link, sadj)

        # data.edge_index, _ = dropout_adj(data.edge_index, p=missing_link, force_undirected=True, num_nodes=data.num_nodes)
        print("masked edge num:", np.sum(sadj))
    features_mask = torch.zeros(size=features.size(), dtype=torch.bool)
    if missing_feature > 0.0:
        print("raw feature value:", torch.sum(features))
        features_mask = torch.rand(size=features.size())
        features_mask = features_mask <= missing_feature

        # feature_mask = get_feature_mask(rate=missing_feature, n_nodes=data.num_nodes,
        #                                 n_features=data.num_features)
        # data.x[~feature_mask] = 0.0 # float('nan')
        # torch.unique(torch.bernoulli(a), return_counts=True)
        features[features_mask] = 0.0
        print("masked  feature value:", torch.sum(features))
        features[features_mask] = float('nan')

    edge_index, edge_weight = from_scipy_sparse_matrix(sadj)
    features = torch.where(torch.isnan(features), torch.zeros_like(features),
                           features)

    data = Data(x=features, edge_index=edge_index, edge_weight=edge_weight,
                y=label)

    node_degree = degree(data.edge_index[0], data.num_nodes)
    data.degree_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.degree_mask[node_degree < 2] = True
    # data.degree_mask = degree_mask
    data.train_mask = index_to_mask(idx_train, size=data.num_nodes)
    data.val_mask = index_to_mask(idx_val, size=data.num_nodes)
    data.test_mask = index_to_mask(idx_test, size=data.num_nodes)

    data.feature_mask = features_mask
    data.groundtruth_x = groundtruth_features
    data.groundtruth_adj = groundtruth_adj
    # data.groundtruth_edge_weight = groundtruth_edge_weight

    # raw data
    data.raw_x = data.x.clone()
    data.raw_edge_index = data.edge_index.clone()
    data.raw_edge_weight = data.edge_weight.clone()

    # data.raw_adj = to_dense_adj(edge_index).squeeze()
    # data.raw_edge_weight = data.edge_weight.clone()
    # to_scipy_sparse_matrix
    # data.adj = to_torch_sparse_tensor(data.edge_index).to(device)
    # x = self.GCN_encoder(features_0, adj)
    # features_1 = torch.where(features == float('nan'), x, features)




    # sta_dic['common_neighbors'] = find_commin_neighbors_matrix(data, data.num_nodes)


    # data.x
    # data.edge_index
    # data.y
    # data.num_nodes
    # data.num_edges
    # data.is_directed()
    # data.has_isolated_nodes()



    miss_rate = int(missing_link*100)

    sta_dic['{}_num_nodes'.format(miss_rate)] = data.num_nodes
    sta_dic['{}_feature_sum'.format(miss_rate)] = torch.sum(data.x).item()
    sta_dic['{}_num_edges'.format(miss_rate)] = data.num_edges / 2

    sta_dic['{}_num_isolated_nodes'.format(miss_rate)] = statistic_num_isolated_nodes(data)
    sta_dic['{}_num_connected_components'.format(miss_rate)], sta_dic['{}_dis_connected_components'.format(miss_rate)] = statistic_num_components(data)
    sta_dic['{}_mum_LCC_nodes'.format(miss_rate)] = torch.max(sta_dic['{}_dis_connected_components'.format(miss_rate)]).item()

    sta_dic['{}_dis_degree'.format(miss_rate)]= degree(data.edge_index[0], data.num_nodes)
    sta_dic['{}_X'.format(miss_rate)] = data.x
    sta_dic['{}_y'.format(miss_rate)] = data.y
    sta_dic['{}_node_homophily'.format(miss_rate)] = homophily(data.edge_index, data.y, method ='node')
    sta_dic['{}_edge_homophily'.format(miss_rate)] = homophily(data.edge_index, data.y, method ='edge')
    sta_dic['{}_class_homophily'.format(miss_rate)] = homophily(data.edge_index, data.y, method ='edge_insensitive')

    sta_dic['{}_dis_node_homophily'.format(miss_rate)] = distribution_node_homophily(data.edge_index,data.y, method = 'node')




    return sta_dic


if __name__ == '__main__':
    for dset in ['cora', 'citeseer', 'pubmed', 'wisconsin', 'texas', 'cornell',
     'chameleon', 'squirrel']:
        stat=load_and_sta_data_v2(dset,missing_link=0.3,missing_feature=0.3)
    pass