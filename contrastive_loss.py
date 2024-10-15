import torch
# torch.cuda.set_device(3)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GraphLoss(nn.Module):
    """
    smoothness_ratio: 0.2 # 0.2, IL: 0.2
    degree_ratio: 0 # 0
    sparsity_ratio: 0 # 0, IL: 0
    """

    def __init__(self, smoothness_ratio =0.2,degree_ratio=0,sparsity_ratio=0 ):
        super(GraphLoss,self).__init__()
        self.smoothness_ratio = smoothness_ratio
        self.degree_ratio = degree_ratio
        self.sparsity_ratio = sparsity_ratio

    def forward(self, out_adj, features):
        """

        :param out_adj: refine_graph
        :param features:  init_node_embeddings
        :return:
        """
        # Graph regularization
        graph_loss = 0
        L = torch.diagflat(torch.sum(out_adj, -1)) - out_adj
        graph_loss += self.smoothness_ratio * torch.trace(
            torch.mm(features.transpose(-1, -2), torch.mm(L, features))) / int(np.prod(out_adj.shape))
        ones_vec = torch.ones(out_adj.size(-1)).cuda()
        graph_loss += -self.degree_ratio * torch.mm(ones_vec.unsqueeze(0), torch.log(
            torch.mm(out_adj, ones_vec.unsqueeze(-1)) + 1e-12)).squeeze() / out_adj.shape[-1]
        graph_loss += self.sparsity_ratio * torch.sum(torch.pow(out_adj, 2)) / int(np.prod(out_adj.shape))
        return graph_loss

#
# class ProtoSupConLoss(nn.Module):
#
#     def __init__(self, temperature=0.07):
#         super(ProtoSupConLoss, self).__init__()
#         self.temperature = temperature
#         self.loss_fct = nn.CrossEntropyLoss()
#     def forward(self, prototypes, proto_labels, features, labels):
#         """
#             计算将各实例与原型的的对比损失
#
#             Args:
#                 prototypes (torch.Tensor): 原型向量矩阵，大小为(num_prototypes, feature_dim)
#                 proto_labels (torch.Tensor): 原型向量的标签向量，大小为(num_prototypes,)
#                 features (torch.Tensor): 实例特征向量矩阵，大小为(num_instances, feature_dim)
#                 labels (torch.Tensor): 实例标签向量，大小为(num_instances,)
#                 temperature (float): 温度参数，用于平滑化互信息
#
#             Returns:
#                 torch.Tensor: 有监督对比学习损失
#             """
#         features = F.normalize(features, p=2, dim=1)
#         prototypes = F.normalize(prototypes, p=2, dim=1)
#
#         cluster_num = proto_labels.shape[0]
#         batch_size = labels.shape[0]
#
#         # 计算原型向量矩阵和实例特征向量矩阵之间的相似度矩阵
#         sim_matrix = torch.div(torch.matmul(features, prototypes.t()) ,self.temperature)
#         # print(sim_matrix)
#         # 生成 positive mask 以及 negtive_mask
#         labels_broad = labels.unsqueeze(dim=1).expand(labels.shape[0], proto_labels.shape[0])
#         proto_labels_broad = proto_labels.unsqueeze(dim=0).expand(labels.shape[0], proto_labels.shape[0])
#
#         positive_mask = torch.eq(labels_broad, proto_labels_broad).float()
#         negtive_mask = 1.0 - positive_mask
#
#         exp_positive_sim_T = torch.exp(sim_matrix * positive_mask)
#         exp_negtive_sim_T = torch.exp(sim_matrix * negtive_mask)
#
#         #     all_proto_loss = torch.tensor([0.0])
#         #     for i in range(labels.shape[0]):
#         #         anchor_exp_positive_sim_T = exp_positive_sim_T[i][:]
#         #         anchor_exp_negtive_sim_T = exp_negtive_sim_T[i][:]
#         #         anchor_loss =( 1.0 / cluster_num)  * torch.log( torch.sum(anchor_exp_positive_sim_T) / (torch.sum(anchor_exp_positive_sim_T) + torch.sum(anchor_exp_negtive_sim_T)))
#         #         all_proto_loss += anchor_loss
#         # (1.0 / cluster_num)
#         anchor_loss = - (1.0 / cluster_num) * torch.sum(torch.log(
#             torch.sum(exp_positive_sim_T, axis=1, keepdims=True) / (
#                         torch.sum(exp_positive_sim_T, axis=1, keepdims=True) + torch.sum(exp_negtive_sim_T, axis=1,
#                                                                                          keepdims=True))), axis=0)
#         # print(anchor_loss)
#         return anchor_loss.squeeze(0) / batch_size
#
# #
# def proto_supervised_contrastive_loss(prototypes, proto_labels, features, labels, temperature=1.0):
#     """
#     计算将各实例与原型的的对比损失
#
#     Args:
#         prototypes (torch.Tensor): 原型向量矩阵，大小为(num_prototypes, feature_dim)
#         proto_labels (torch.Tensor): 原型向量的标签向量，大小为(num_prototypes,)
#         features (torch.Tensor): 实例特征向量矩阵，大小为(num_instances, feature_dim)
#         labels (torch.Tensor): 实例标签向量，大小为(num_instances,)
#         temperature (float): 温度参数，用于平滑化互信息
#
#     Returns:
#         torch.Tensor: 有监督对比学习损失
#     """
#     cluster_num = proto_labels.shape[0]
#     batch_size = labels.shape[0]
#
#
#     # 计算原型向量矩阵和实例特征向量矩阵之间的相似度矩阵
#     sim_matrix = torch.matmul(features, prototypes.t()) / temperature
#     print(sim_matrix)
#     # 生成 positive mask 以及 negtive_mask
#     labels_broad = labels.unsqueeze(dim=1).expand(labels.shape[0], proto_labels.shape[0])
#     proto_labels_broad = proto_labels.unsqueeze(dim=0).expand(labels.shape[0], proto_labels.shape[0])
#
#     positive_mask = torch.eq(labels_broad, proto_labels_broad).float()
#     negtive_mask = 1.0 - positive_mask
#
#     exp_positive_sim_T = torch.exp(sim_matrix * positive_mask / temperature)
#     exp_negtive_sim_T = torch.exp(sim_matrix * negtive_mask / temperature)
#
#
#     #     all_proto_loss = torch.tensor([0.0])
#     #     for i in range(labels.shape[0]):
#     #         anchor_exp_positive_sim_T = exp_positive_sim_T[i][:]
#     #         anchor_exp_negtive_sim_T = exp_negtive_sim_T[i][:]
#     #         anchor_loss =( 1.0 / cluster_num)  * torch.log( torch.sum(anchor_exp_positive_sim_T) / (torch.sum(anchor_exp_positive_sim_T) + torch.sum(anchor_exp_negtive_sim_T)))
#     #         all_proto_loss += anchor_loss
#
#     anchor_loss = -(1.0 / cluster_num) * torch.sum(torch.log(torch.sum(exp_positive_sim_T, axis=1, keepdims=True) / (torch.sum(exp_positive_sim_T, axis=1, keepdims=True) + torch.sum(exp_negtive_sim_T, axis=1,keepdims=True))), axis=0,keepdims=True)
#
#     return anchor_loss / batch_size
#

# class protClrLoss(nn.Module):
#     def __init__(self, loss_fct = nn.CrossEntropyLoss(), cos_score_transformation=nn.Sigmoid()):
#         super(protClrLoss, self).__init__()
#         self.loss_fct = loss_fct
#         #self.cos_score_transformation = cos_score_transformation
#         self.loss_un = nn.CrossEntropyLoss(reduction='none')
#     def forward(self, support, prototypical_embedding, weight, unsup, y):
#         # print (features)
#         #pro_dist = F.softmax(-euclidean_dist(support, prototypical_embedding), dim=1)
#         pro_dist = F.softmax(-euclidean_dist(support.clone().detach(), prototypical_embedding), dim=1)
#         # print (pro_dist)
#         labels = torch.arange(prototypical_embedding.size(0)).cuda()
#         labels = torch.cat([labels, labels, labels],
#                       dim=0)
#         un_dist = F.softmax(-euclidean_dist(unsup,prototypical_embedding),dim=1)
#         #print (un_dist.shape)
#         unloss = self.loss_un(un_dist,y)
#         #print (weight,y , unsup.shape)
#         un_loss = torch.sum(unloss * weight, dim=-1) / unsup.size(0)
#         #print (un_loss)
#         return self.loss_fct(pro_dist,labels),un_loss


class SupConLoss(nn.Module):

    def __init__(self, temperature=0.5, scale_by_temperature=True):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.scale_by_temperature = scale_by_temperature

    def forward(self, features, labels=None, mask=None):
        """
        输入:
            features: 输入样本的特征，尺寸为 [batch_size, hidden_dim].
            labels: 每个样本的ground truth标签，尺寸是[batch_size].
            mask: 用于对比学习的mask，尺寸为 [batch_size, batch_size], 如果样本i和j属于同一个label，那么mask_{i,j}=1
        输出:
            loss值
        """
        # device = (torch.device('cuda:0')
        #           if features.is_cuda
        #           else torch.device('cpu'))
        features = F.normalize(features, p=2, dim=1)
        batch_size = features.shape[0]
        # 关于labels参数
        if labels is not None and mask is not None:  # labels和mask不能同时定义值，因为如果有label，那么mask是需要根据Label得到的
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:  # 如果没有labels，也没有mask，就是无监督学习，mask是对角线为1的矩阵，表示(i,i)属于同一类
            mask = torch.eye(batch_size, dtype=torch.float32).cuda()
        elif labels is not None:  # 如果给出了labels, mask根据label得到，两个样本i,j的label相等时，mask_{i,j}=1
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().cuda()
        else:
            mask = mask.float().cuda()
        '''
        示例: 
        labels: 
            tensor([[1.],
                    [2.],
                    [1.],
                    [1.]])
        mask:  # 两个样本i,j的label相等时，mask_{i,j}=1
            tensor([[1., 0., 1., 1.],
                    [0., 1., 0., 0.],
                    [1., 0., 1., 1.],
                    [1., 0., 1., 1.]]) 
        '''
        # compute logits
        # anchor_dot_contrast = torch.div(
        #     torch.matmul(features, features.T),
        #     self.temperature)  # 计算两两样本间点乘相似度
        epsilon = 1e-6
        x_norm = torch.div(features, torch.clamp(torch.norm(features, dim=-1, keepdim=True), min=epsilon))  # 方差归一化，即除以各自的模
        cos_sim = torch.mm(x_norm, x_norm.T)# 矩阵乘法
        anchor_dot_contrast = torch.div(
            cos_sim,
            self.temperature)  # 计算两两样本间cosine# 相似度

        # anchor_dot_contrast = torch.div(
        #     F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0),dim=2),
        #     self.temperature)  # 计算两两样本间cosine# 相似度
        # for numerical stability

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        exp_logits = torch.exp(logits)
        '''
        logits是anchor_dot_contrast减去每一行的最大值得到的最终相似度
        示例: logits: torch.size([4,4])
        logits:
            tensor([[ 0.0000, -0.0471, -0.3352, -0.2156],
                    [-1.2576,  0.0000, -0.3367, -0.0725],
                    [-1.3500, -0.1409, -0.1420,  0.0000],
                    [-1.4312, -0.0776, -0.2009,  0.0000]])       
        '''
        # 构建mask
        logits_mask = torch.ones_like(mask).cuda() - torch.eye(batch_size).cuda()
        positives_mask = mask * logits_mask
        negatives_mask = 1. - mask
        '''
        但是对于计算Loss而言，(i,i)位置表示样本本身的相似度，对Loss是没用的，所以要mask掉
        # 第ind行第ind位置填充为0
        得到logits_mask:
            tensor([[0., 1., 1., 1.],
                    [1., 0., 1., 1.],
                    [1., 1., 0., 1.],
                    [1., 1., 1., 0.]])
        positives_mask:
        tensor([[0., 0., 1., 1.],
                [0., 0., 0., 0.],
                [1., 0., 0., 1.],
                [1., 0., 1., 0.]])
        negatives_mask:
        tensor([[0., 1., 0., 0.],
                [1., 0., 1., 1.],
                [0., 1., 0., 0.],
                [0., 1., 0., 0.]])
        '''
        num_positives_per_row = torch.sum(positives_mask, axis=1)  # 除了自己之外，正样本的个数  [2 0 2 2]
        denominator = torch.sum(exp_logits * negatives_mask, axis=1, keepdims=True) + torch.sum(exp_logits * positives_mask, axis=1, keepdims=True)

        log_probs = logits - torch.log(denominator)
        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")

        log_probs = torch.sum(log_probs * positives_mask, axis=1)[num_positives_per_row > 0] / num_positives_per_row[num_positives_per_row > 0]
        '''
        计算正样本平均的log-likelihood
        考虑到一个类别可能只有一个样本，就没有正样本了 比如我们labels的第二个类别 labels[1,2,1,1]
        所以这里只计算正样本个数>0的    
        '''
        # loss
        loss = -log_probs
        if self.scale_by_temperature:
            loss *= self.temperature
        loss = loss.mean()
        return loss





# class protClrLoss(nn.Module):
#
#     def __init__(self, loss_fct=nn.CrossEntropyLoss(), cos_score_transformation=nn.Sigmoid()):
#         super(protClrLoss, self).__init__()
#         self.loss_fct = loss_fct
#         # self.cos_score_transformation = cos_score_transformation
#         self.loss_un = nn.CrossEntropyLoss(reduction='none')
#
#     def forward(self, support, prototypical_embedding, weight, unsup, y):
#         # print (features)
#         # pro_dist = F.softmax(-euclidean_dist(support, prototypical_embedding), dim=1)
#         pro_dist = F.softmax(-euclidean_dist(support.clone().detach(), prototypical_embedding), dim=1)
#         # print (pro_dist)
#         labels = torch.arange(prototypical_embedding.size(0)).cuda()
#         labels = torch.cat([labels, labels, labels],
#                            dim=0)
#         un_dist = F.softmax(-euclidean_dist(unsup, prototypical_embedding), dim=1)
#         # print (un_dist.shape)
#         unloss = self.loss_un(un_dist, y)
#         # print (weight,y , unsup.shape)
#         un_loss = torch.sum(unloss * weight, dim=-1) / unsup.size(0)
#         # print (un_loss)
#         return self.loss_fct(pro_dist, labels), un_loss