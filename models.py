import torch.nn as nn
import math
import torch
import torch.nn.functional as F
from torch_geometric.nn.models import GCN,GAT,GraphSAGE
from torch_geometric.nn.conv import APPNP
from utils import get_knn_graph,show_gpu_allocated

from torch_geometric.utils.convert import from_scipy_sparse_matrix
from torch_geometric.nn import knn_graph
from torch_geometric.utils import to_dense_adj,dense_to_sparse,add_self_loops,\
    remove_self_loops,is_undirected,to_undirected
# to_torch_sparse_tensor
# from torch_geometric.utils import to_dense_adj,to_edge_index,add_self_loops
from GCL.augmentors.ppr_diffusion import PPRDiffusion
from GCL.augmentors.augmentor import Graph
from torch_geometric.nn import knn_graph
import GCL.augmentors as A
from torch_geometric.nn import GCNConv
from torch.autograd import Variable
import torch.nn.init as init
# from autoencoder import GAE
from numbers import Number
from GIB import GIBGNN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels,dropout):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True) # cached only for transductive learning
        self.conv2 = GCNConv(2 * out_channels, out_channels, cached=True) # cached only for transductive learning
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, edge_index):
        x = self.dropout(self.conv1(x, edge_index).relu())
        return self.conv2(x, edge_index)


class GCNEncoder_IB(torch.nn.Module):
    def __init__(self, in_channels, out_channels,dropout):
        super(GCNEncoder_IB, self).__init__()
        self.K = out_channels
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True) # cached only for transductive learning
        self.conv2 = GCNConv(2 * out_channels, 2 * self.K, cached=True) # cached only for transductive learning
        self.dropout = nn.Dropout(dropout)

    def cuda(self, tensor, is_cuda):
        if is_cuda:
            return tensor.cuda()
        else:
            return tensor

    def reparametrize_n(self, mu, std, n=1):
        # reference :
        # http://pytorch.org/docs/0.3.1/_modules/torch/distributions.html#Distribution.sample_n
        def expand(v):
            if isinstance(v, Number):
                return torch.Tensor([v]).expand(n, 1)
            else:
                return v.expand(n, *v.size())

        if n != 1:
            mu = expand(mu)
            std = expand(std)

        eps = Variable(
            self.cuda(std.data.new(std.size()).normal_(), std.is_cuda))

        return mu + eps * std

    def forward(self, x, edge_index,edge_weight):
        x = self.dropout(self.conv1(x, edge_index,edge_weight).relu())
        h = self.conv2(x, edge_index,edge_weight)

        mu = h[:, :self.K]
        std = F.softplus(h[:, self.K:] - 5, beta=1)
        h = self.reparametrize_n(mu,std,1)

        return (mu, std), h


class Linear(nn.Module):
    def __init__(self, in_features, out_features, dropout, bias=False):
        super(Linear, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, mode='fan_out', a=math.sqrt(5))
        if self.bias is not None:
            stdv = 1. / math.sqrt(self.weight.size(1))
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        input = F.dropout(input, self.dropout, training=self.training)
        output = torch.matmul(input, self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class MLP_encoder(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(MLP_encoder, self).__init__()
        self.Linear1 = Linear(nfeat, nhid, dropout, bias=True)

    def forward(self, x):
        x = torch.relu(self.Linear1(x))
        return x

class MLP_classifier(nn.Module):#
    def __init__(self, nfeat, nclass, dropout):
        super(MLP_classifier, self).__init__()
        self.Linear1 = Linear(nfeat, nfeat, dropout, bias=True)
        self.Linear2 = Linear(nfeat, nclass, dropout, bias=True)

    def forward(self, x):
        x_t = torch.relu(self.Linear1(x))
        # torch.relu(self.Linear1(x))
        out = self.Linear2(x_t)
        return torch.log_softmax(out, dim=1), x_t


class Two_MLP_Layer(nn.Module):#
    def __init__(self, in_size, hidden_size,out_size, dropout):
        super(Two_MLP_Layer, self).__init__()
        self.Linear1 = nn.Linear(in_size, hidden_size, bias=True)
        self.Linear2 = nn.Linear(hidden_size, out_size, bias=True)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x_t = self.dropout(torch.relu(self.Linear1(x)))
        # torch.relu(self.Linear1(x))
        # out = F.sigmoid(self.Linear2(x_t))
        return self.Linear2(x_t)


class S2_Decoupled_GCN_3SCL_1CE_SUM_V3(nn.Module):
    def __init__(self, num_nodes, in_size, hidden_size, out_size, num_layer,
                 dropout, device,lxw,law,lzw,GNN_type):
        super(S2_Decoupled_GCN_3SCL_1CE_SUM_V3, self).__init__()

        # self.tgc1 = GraphConvolution(in_size, hidden_size)
        # self.tgc2 = GraphConvolution(hidden_size, out_size)
        # self.dropout = dropout
        # self.pe_feat = torch.FloatTensor(torch.eye(num_nodes)).to(device)
        # self.linear = nn.Linear(num_nodes, in_size, bias=True)
        #
        # self.GCN_classifier=  GCN(in_channels=in_size,hidden_channels=hidden_size,num_layers=num_layer,out_channels=out_size,dropout=dropout)
        # self.num_nodes = num_nodes
        # self.imp_feat = nn.Parameter(torch.empty(size=(num_nodes, in_size)))
        # nn.init.xavier_normal_(self.imp_feat.data, gain=1.414)
        self.mlp_x = MLP_X_Layer_NoIMP(num_nodes, in_size, hidden_size,
                                       hidden_size, num_layer, dropout)
        self.mlp_a = MLP_A_Layer(num_nodes, in_size, hidden_size, hidden_size,
                                 num_layer, dropout, device)
        self.lxw = lxw
        self.law = law
        self.lzw = lzw

        #
        # self.X_fm1 = nn.Linear(in_size, hidden_size, bias=True)
        # self.X_fm2 = nn.Linear(hidden_size, hidden_size, bias=True)
        #
        # self.X_fm3 = nn.Linear(num_nodes, hidden_size, bias=True)
        # self.X_fm4 = nn.Linear(hidden_size, hidden_size, bias=True)

        # self.A_fm1 = nn.Linear(num_nodes, hidden_size, bias=True)
        # self.A_fm2 = nn.Linear(hidden_size, hidden_size, bias=True)
        #
        # self.A_fm3 = nn.Linear(num_nodes, hidden_size, bias=True)
        # self.A_fm4 = nn.Linear(hidden_size, hidden_size, bias=True)
        if GNN_type == 'GAT':
            self.GCN_classifier = GAT(in_channels=in_size,
                                      hidden_channels=hidden_size,
                                      num_layers=num_layer,
                                      out_channels=hidden_size,
                                      dropout=dropout)
        elif GNN_type == 'GraphSAGE':
            self.GCN_classifier = GraphSAGE(in_channels=in_size,
                                      hidden_channels=hidden_size,
                                      num_layers=num_layer,
                                      out_channels=hidden_size,
                                      dropout=dropout)
        # elif GNN_type == 'APPNP':
        #     self.GCN_classifier = APPNP(in_channels=in_size,
        #                               hidden_channels=hidden_size,
        #                               num_layers=num_layer,
        #                               out_channels=hidden_size,
        #                               dropout=dropout)

        else:
            self.GCN_classifier = GCN(in_channels=in_size,
                                      hidden_channels=hidden_size,
                                      num_layers=num_layer, out_channels=hidden_size,
                                      dropout=dropout)

        self.Z_fm1 = nn.Linear(hidden_size, out_size, bias=False)
        # self.Z_fm2 = nn.Linear(hidden_size, out_size, bias=True)
        self.dropout = nn.Dropout(p=dropout)

        # self.stru_rec1 = GCN(in_channels=in_size,hidden_channels=hidden_size,num_layers=num_layer,out_channels=hidden_size,dropout=dropout)
        # self.GCN_classifier=  GCN(in_channels=in_size,hidden_channels=hidden_size,num_layers=num_layer,out_channels=hidden_size,dropout=dropout)

        # self.stru_rec2 = GCN(in_channels=in_size,hidden_channels=hidden_size,num_layers=num_layer,out_channels=hidden_size,dropout=dropout)
        # self.pe_feat = torch.FloatTensor(torch.eye(num_nodes)).to(device)
        # self.linear = nn.Linear(num_nodes, in_size, bias=True)
        #
        # # self.dropout = dropout
        # self.weights_init()

    # def weights_init(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Linear):
    #             torch.nn.init.xavier_uniform_(m.weight.data)
    #             if m.bias is not None:
    #                 m.bias.data.fill_(0.0)

    def forward(self, feature, edge_index, edge_weight,feature2, edge_index2, edge_weight2,
                num_neighbor=5):
        # feature = torch.where(feature_mask, self.imp_feat, feature)
        # h = self.X_fm1(feature)
        # h = F.relu(h)
        # h = self.dropout(h)  # 在激活之后应用 Dropout
        # node_emb = self.X_fm2(h)
        #
        # h = self.X_fm3(feature.T)
        # h = F.relu(h)
        # h = self.dropout(h)  # 在激活之后应用 Dropout
        # attr_emb = self.X_fm4(h)
        #
        # new_feature = node_emb @ attr_emb.T
        # # new_feature = torch.where(F.sigmoid(new_feature) > 0.5, 1.0, 0.0)
        #
        # # print(torch.sum(new_feature),new_feature.nonzero().shape[0])
        # new_feature2 = torch.where(feature_mask, new_feature, feature)

        # new_feature = F.relu(new_feature)
        # self.stru_rec()

        # pe = self.linear(self.pe_feat)
        # pe = F.dropout(pe, self.dropout, training=self.training)

        # h1 = self.stru_rec1(pe, edge_index, edge_weight = edge_weight)
        # h2 = self.stru_rec2(pe, edge_index, edge_weight = edge_weight)

        # adj =to_dense_adj(add_self_loops(edge_index)[0],max_num_nodes=self.num_nodes, edge_attr=edge_weight).squeeze()
        # adj = to_dense_adj(edge_index, max_num_nodes=self.num_nodes,
        #                    edge_attr=edge_weight).squeeze()
        # h = self.A_fm1(adj)
        # h = F.relu(h)
        # h = self.dropout(h)  # 在激活之后应用 Dropout
        # A1_emb = self.A_fm2(h)
        #
        # h = self.X_fm3(adj.T)
        # h = F.relu(h)
        # h = self.dropout(h)  # 在激活之后应用 Dropout
        # A2_emb = self.X_fm4(h)
        #
        # new_structure = A1_emb @ A2_emb.T
        # # new_structure = h1 @ h2.T
        # new_structure = F.relu(new_structure)
        # #
        # # adj_2 = self.build_knn_neighbourhood(new_structure, num_neighbor) # 保留前tok
        # # new_adj = torch.where(adj == 0.0, adj, new_structure)
        #
        # egde_index_1, edge_weight_1 = dense_to_sparse(new_structure)

        # egde_index_2, edge_weight_2 = remove_self_loops(egde_index_1, edge_weight_1)

        # new_adj =  to_dense_adj(egde_index_2, max_num_nodes=self.num_nodes, edge_attr=edge_weight_2).squeeze()
        x_h = self.mlp_x(feature)
        a_h = self.mlp_a(edge_index,edge_weight)


        output1 = self.GCN_classifier(x=feature2, edge_index=edge_index2,
                                     edge_weight=edge_weight2)

        # output2 = self.GCN_classifier(x=new_feature, edge_index=edge_index,
        #                              edge_weight=edge_weight)
        # output3 = self.GCN_classifier(x=feature, edge_index=new_edge_index,
        #                               edge_weight=new_edge_weight)
        # z = torch.cat((output1,x_h,a_h),1)
        z = self.lxw * x_h + self.law * a_h + self.lzw * output1

        # z = self.Z_fm2(self.Z_fm1(self.dropout(F.relu(z))))
        z = self.Z_fm1(z)
        return  x_h,a_h,output1,F.log_softmax(z, dim=1)

