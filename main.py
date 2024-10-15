import datetime
import os
import sys
# Solving the problem of not being able to import your own packages under linux
cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, cur_path + "/..")
import arguments
import time
from utils import *
from models import DDPT,GCNBaseline,GCNMLP,IterateGCN,IterateGCN_onlyX,\
    IterateGCN_onlyA,MLP_X,GCN_A,Decoupled_MLP_GCN,MLP_X_IMP0,\
    Decoupled_MLP_GCN_ATT, Decoupled_MLP_GCN_GNN, Decoupled_DMLP_DMLP_GNN,\
    UGCL_S1, UGCL_S2, Decoupled_DMLP_DGCN_GNN, Decoupled_DMLP_GNN,\
    Decoupled_ADMLP_GNN,S1_Decoupled_XDMLP_GNN,S2_Decoupled_XDMLP_GNN,\
    S1_Decoupled_ADMLP_GNN,S2_Decoupled_ADMLP_GNN,MLP_A,\
    S1_Decoupled_ADMLP_GNNV2,S2_Decoupled_ADMLP_GNNV2,UGCL_S2_DropFusion,\
    S1_Decoupled_XDMLP_GNNV3,Decoupled_XMLP_AMLP_XAGCN, S1_Decoupled_XDMLP_GNNV6,\
    Decoupled_XMLP_AMLP_XAGCN_SUM,S1_Decoupled_XDMLP_GNN_COR_REC_CE,\
    S1_Decoupled_ADMLP_GNN_COR_REC_BCE,S2_Decoupled_XMLP_AMLP_XAGCN,\
    Decoupled_XMLP_AMLP_XAGCN_3GCN,Decoupled_XMLP_AMLP_XAGCN_3GCN_3SCL,\
    Decoupled_XAGCN_3GCN,Decoupled_IMPXA_XA_3GCN_3SCL_1CE,Decoupled_XMLP_AMLP_XAGCN_3CE,\
    Decoupled_IMPXA_XA_3GCN,S2_Decoupled_3GCN, S2_Decoupled_3GCN_3SCL_1CE,\
    S1_Decoupled_ADMLP_GNN_COR_REC_BCE_pubmed,S2_Decoupled_3GCN_3SCL_1CE_SUM,\
    S1_Decoupled_XMAE_GNN_CE,MLPX_GCNXA,MLPX_MLPA_GCNXA,S2_Decoupled_1GCN_3SCL_1CE_SUM,\
    S1_Decoupled_XDML_COR_REC_SCL,S1_Decoupled_ADMLP_COR_REC_SCL,S2_Decoupled_1GCN_1CE_SUM,\
    S2_Decoupled_1GCN_1CE_SUM_v2,GCNEncoder,Two_MLP_Layer,S2_Decoupled_1GCN_1CE_SUM_v3,\
    MLP_A_pubmed,S2_Decoupled_GCN_3SCL_1CE_SUM,S2_Decoupled_GCN_3SCL_1CE_ATT_SUM,IterateGCN_V2,\
    S2_Decoupled_GCN_3SCL_1CE_SUM_V2,S2_Decoupled_GCN_3SCL_1CE_SUM_V3,S2_Decoupled_GCN_3SCL_1CE_SUM_ATT_V3,\
    S2_Decoupled_GCN_3SCL_1CE_XIB_SUM,S2_Decoupled_GCN_3SCL_1CE_CATIB_SUM,S1_Decoupled_ADMLP_GNN_COR_REC_BCE_N2V
import time
import GCL.augmentors as A
from sklearn.decomposition import NMF
from autoencoder import GAE

from early_stop import EarlyStopping, Stop_args
from load_data import load_data
import warnings
warnings.simplefilter("ignore", UserWarning)
from contrastive_loss import SupConLoss
import wandb
from ppr_matrix import topk_ppr_matrix
from torch_geometric.utils.convert import to_scipy_sparse_matrix
from torch_geometric.utils.convert import from_scipy_sparse_matrix

from torch_geometric.utils import to_dense_adj,to_edge_index,train_test_split_edges
from GCL.models import DualBranchContrast
import GCL.losses as L
from torch_geometric.loader import DataLoader

from torch_geometric.utils import to_dense_adj,dense_to_sparse,add_self_loops,\
    remove_self_loops,is_undirected,to_undirected
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torchmetrics import Accuracy, Precision, Recall, F1Score
from torchmetrics.classification import BinaryAccuracy, BinaryRecall, BinaryPrecision, BinaryF1Score

from FocalLoss import focal_loss
import cProfile
from torch.optim.lr_scheduler import StepLR
from torch_geometric.utils import train_test_split_edges
from Node2Vec import N2V
# from LabelSmoothing import LabelSmoothingCrossEntropy

args = arguments.parse_args()


os.environ['WANDB_HTTP_TIMEOUT'] = '300'
os.environ['WANDB_INIT_TIMEOUT'] = '1000'
# for debug
if args.run == 'debug':
    # args.run = 'run'

    args.num_trials = 10
    args.missing_link = 0.3
    args.missing_feature = 0.3
    args.split_datasets_type = 't2-gnn'

    args.dataset = 'pubmed'
    # args.dataset = 'cora'
    args.epochs = 2000
    args.lr = 0.01
    args.lr1 = 0.0001
    args.lr2 = 0.0001
    args.weight_decay = 0.005
    args.patience = 500
    args.gcn_num_layer = 2
    args.hidden = 256
    args.dropout = 0.1
    args.num_neighbor = 5
    args.tau=0.15
    args.thresholds = 0.5

    args.lxw = 0.0
    args.law = 0.0
    args.lzw = 1.25
    args.scl_weight = 0.0
    args.X_num_neighbor = 0
    args.A_num_neighbor = 0
    args.X_thresholds = 0.85
    args.A_thresholds = 0.5

    args.X_num_neighbor_2 = 0
    args.A_num_neighbor_2 = 0
    args.X_thresholds_2 = 0.95
    args.A_thresholds_2 = 0.95
    args.beta = 0.000001
    args.beta1 = 0.00001
    args.beta2 = 0.00001
    args.scl_weight2 = 0.0
    args.topk_weight = 0.3

    # args.dataset = 'citeseer'
    # args.epoch = 2000
    # args.lr = 0.01
    # args.lr1 = 0.0001
    # args.lr2 = 0.0001
    # args.weight_decay = 0.005
    # args.patience = 2000
    # args.gcn_num_layer = 2
    # args.hidden = 128
    # args.dropout = 0.4
    # args.num_neighbors = 5
    # args.taus = 0.8
    # args.thresholds = 0.5
    # args.X_num_neighbor = 0
    # args.A_num_neighbor = 0
    # args.lxw = 0.75
    # args.law = 0.0
    # args.lzw = 1.0
    # args.scl_weight = 1.0
    # args.X_thresholds = 0.5
    # args.A_thresholds = 0.95
    # args.beta = 0.00001
    # args.beta1 = 0.00001
    # args.beta2 = 0.00001

    # args.dataset = 'pubmed'
    # args.epoch = 2000
    # args.lr = 0.01
    # args.lr1 = 0.01
    # args.lr2 = 0.01
    # args.weight_decay = 0.005
    # args.patience = 500
    # args.gcn_num_layer = 2
    # args.hidden = 256
    # args.dropout = 0.9
    # args.num_neighbor = 50
    # args.tau=0.15
    # args.thresholds = 0.5
    # args.X_num_neighbor = 6
    # args.A_num_neighbor = 20
    # args.lxw = 0.25
    # args.law = 0.25
    # args.lzw = 0.5



    # args.dataset = 'cornell'
    # args.epoch = 2000
    # args.lr = 0.01
    # args.lr1 = 0.0001
    # args.lr2 = 0.0001
    # args.weight_decay = 0.01
    # args.patience = 500
    # args.gcn_num_layer = 2
    # args.hidden = 64
    # args.dropout = 0.2
    # args.num_neighbors = 5
    # args.taus = 0.05
    # args.X_num_neighbor = 6
    # args.A_num_neighbor = 20
    # args.lxw = 1.0
    # args.law = 1.0
    # args.lzw = 1.0

    # args.dataset = 'texas'
    # args.epoch = 2000
    # args.lr = 0.01
    # args.lr1 = 0.0001
    # args.lr2 = 0.0001
    # args.weight_decay = 0.005
    # args.patience = 500
    # args.gcn_num_layer = 2
    # args.hidden = 128
    # args.dropout = 0.9
    # args.X_num_neighbor = 30
    # args.A_num_neighbor = 95
    # args.num_neighbors = 50
    # args.taus = 0.75
    # args.lxw = 0.1
    # args.law = 0.1
    # args.lzw = 0.5

    # args.dataset = 'wisconsin'
    # args.epoch = 2000
    # args.lr = 0.01
    # args.lr1 = 0.0001
    # args.lr2 = 0.0001
    # args.weight_decay = 0.005
    # args.patience = 2000
    # args.gcn_num_layer = 2
    # args.hidden = 512
    # args.dropout = 0.5
    # args.num_neighbors = 50
    # args.X_num_neighbor = 20
    # args.A_num_neighbor = 1
    # args.num_neighbors = 50
    # args.taus = 0.05
    #
    # args.lxw = 1.0
    # args.law = 1.0
    # args.lzw = 1.0
    #
    # args.dataset = 'chameleon'
    # args.epoch = 5000
    # args.lr = 0.001
    # args.lr1 = 0.0001
    # args.lr2 = 0.0001
    # args.weight_decay = 0.005
    # args.patience = 500
    # args.gcn_num_layer = 4
    # args.hidden = 512
    # args.dropout = 0.5
    # args.num_neighbors = 5
    # args.taus = 0.15
    # args.thresholds = 0.5
    # args.X_num_neighbor = 30
    # args.A_num_neighbor = 10
    # args.lxw = 1.0
    # args.law = 1.0
    # args.lzw = 1.0

    # args.dataset = 'squirrel'
    # args.epoch = 5000
    # args.lr = 0.001
    # args.lr1 = 0.0001
    # args.lr2 = 0.0001
    # args.weight_decay = 0.005
    # args.patience = 500
    # args.gcn_num_layer = 4
    # args.hidden = 512
    # args.dropout = 0.5
    # args.num_neighbors = 5
    # args.taus = 0.75
    # args.thresholds = 0.5
    # args.X_num_neighbor = 5
    # args.A_num_neighbor = 1
    # args.lxw = 1.0
    # args.law = 1.0
    # args.lzw = 1.0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main_TS_Decoupled_IMPXA_UPDATE_IterGCN_3SCL_1CE_SUM_Step1V2(trial):
    data = load_data(args.dataset,
                           missing_link= args.missing_link,
                           missing_feature = args.missing_feature,
                           trial=trial,
                           split_datasets_type=args.split_datasets_type,
                           normalize_features=args.normalize_features)


    data = data.to(device)

    labels = data.y
    idx_train = torch.where(data.train_mask==True)[0]
    idx_val = torch.where(data.val_mask == True)[0]
    idx_test = torch.where(data.test_mask == True)[0]
    num_classes = len(set(data.y.tolist()))

    train_val_labels = labels.to(device)
    train_val_idx = torch.cat([idx_train,idx_val]).to(device)


    X_model_S1  = S1_Decoupled_XDMLP_GNN_COR_REC_CE(num_nodes=data.num_nodes,in_size=data.x.shape[1],
                         hidden_size=args.hidden,
                         num_layer=args.gcn_num_layer,
                         out_size=num_classes,
                         dropout=args.dropout,device=device).to(device)
    A_model_S1  = S1_Decoupled_ADMLP_GNN_COR_REC_BCE(num_nodes=data.num_nodes,in_size=data.x.shape[1],
                         hidden_size=args.hidden,
                         num_layer=args.gcn_num_layer,
                         out_size=num_classes,
                         dropout=args.dropout,device=device).to(device)
    # A_model_S1 = S1_Decoupled_ADMLP_GNN_COR_REC_BCE_N2V(num_nodes=data.num_nodes,
    #                                                    in_size=data.x.shape[1],
    #                                                    hidden_size=args.hidden,
    #                                                    num_layer=args.gcn_num_layer,
    #                                                    out_size=num_classes,
    #                                                    dropout=args.dropout,
    #                                                    device=device).to(device)
    # N2V_model = N2V(edge_index=data.edge_index,device=device,batch_size=data.num_nodes,embedding_dim=args.hidden)
        # N2V_emb = N2V_model()

    if args.dataset == 'pubmed':
        A_model_S1 = S1_Decoupled_ADMLP_GNN_COR_REC_BCE_pubmed(num_nodes=data.num_nodes,
                                                 in_size=data.x.shape[1],
                                                 hidden_size=args.hidden,
                                                 num_layer=args.gcn_num_layer,
                                                 out_size=num_classes,
                                                 dropout=args.dropout,
                                                 device=device).to(device)


    X_optimizer_1 = torch.optim.Adam(X_model_S1.parameters(), lr=args.lr1, weight_decay=args.weight_decay)
    A_optimizer_1 = torch.optim.Adam(A_model_S1.parameters(), lr=args.lr2, weight_decay=args.weight_decay)



    stopping_args1 = Stop_args(patience=args.epochs, max_epochs=args.epochs)
    stopping_args2 = Stop_args(patience=args.epochs, max_epochs=args.epochs)
    stopping_args3 = Stop_args(patience=args.patience, max_epochs=args.epochs)
    X_early_stopping_1 = EarlyStopping(X_model_S1, **stopping_args1)
    A_early_stopping_1 = EarlyStopping(A_model_S1, **stopping_args2)

    # total_steps = args.epochs
    # warmup_steps = int(0.1 * total_steps)  # 10% of total steps for warmup
    # scheduler1 = WarmupCosineSchedule(X_optimizer_1, warmup_steps=warmup_steps,
    #                                   t_total=total_steps)
    # scheduler2 = WarmupCosineSchedule(A_optimizer_1, warmup_steps=warmup_steps,
    #                                   t_total=total_steps)

    # 初始化指标
    accuracy_fuc = BinaryAccuracy().to(device)
    precision_fuc = BinaryPrecision().to(device)
    recall_fuc = BinaryRecall().to(device)
    f1score_fuc = BinaryF1Score().to(device)


    num_ones = torch.sum(data.x[data.features_unmask_train] == 1.0)
    num_zeros = torch.sum(data.x[data.features_unmask_train] == 0.0)
    num_ones = int(
        len(data.x[data.feature_mask]) * (num_ones / (num_ones + num_zeros)))



    if args.dataset == "pubmed":
        X_criterion = torch.nn.MSELoss(reduction="mean").to(device)
    else:
        if args.X_num_neighbor == 0:
            X_criterion = nn.BCEWithLogitsLoss().to(device)
        else:
            X_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.X_num_neighbor)).to(device)

    num_ones = torch.sum(data.raw_adj == 1.0)
    num_zeros = torch.sum(data.raw_adj == 0.0)
    num_ones = int(
        num_zeros * (num_ones / (num_ones + num_zeros)))
    if args.A_num_neighbor == 0:
        A_criterion = nn.BCEWithLogitsLoss().to(device)
    else:
        A_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.A_num_neighbor)).to(device)



    supcloss = SupConLoss(temperature=args.tau)

    def X_run_s1():
        # 初始化采样器
        Masksampler = MaskBalancedSampler(data.x, data.feature_mask)
        # 获取测试集mask
        test_mask = Masksampler.get_test_mask()
        def train_base():
            X_model_S1.train()  #设置模型为训练模式，这将启用 Dropout
            X_optimizer_1.zero_grad()
            feature, node_emb1,node_emb2 = X_model_S1(data.x, data.feature_mask,data.edge_index,data.edge_weight,num_neighbor=args.X_num_neighbor)

            # rec_loss = X_criterion(feature[data.features_unmask_train],
            #                  data.x[data.features_unmask_train])

            # if args.dataset=='pubmed' or args.dataset =='cornell' or args.dataset =='wisconsin':
            if args.dataset=='pubmed':
                rec_loss = X_criterion(feature[data.features_unmask_train],
                                 data.x[data.features_unmask_train])
            else:
                train_mask = Masksampler.get_train_mask()
                rec_loss = X_criterion(feature[train_mask],
                                       data.x[train_mask])
            cor_loss = compute_corr(node_emb1,
                             node_emb2)


            loss = rec_loss + cor_loss

            x1_loss = X_criterion(feature[data.feature_mask],
                             data.groundtruth_x[data.feature_mask])



            loss.backward()
            X_optimizer_1.step()
            if args.run == 'run':
                wandb.log({f"{trial}-X_S1/train/train_X_unmask_rec_loss":loss.item(),
                           f"{trial}-X_S1/train/train_X_mask_rec_loss": x1_loss,})
                           # f"{trial}-S1/train/correct_rate": correct_rate,
                           # f"{trial}-S1/train/wrong_rate": wrong_rate})



        def eval(t):
            X_model_S1.eval()  # 设置模型为评估模式，这将禁用 Dropout
            with torch.no_grad():  # 确保在评估过程中不会计算梯度

                feature, node_emb1, node_emb2 = X_model_S1(data.x, data.feature_mask,data.edge_index, data.edge_weight,num_neighbor=args.X_num_neighbor)

                # rec_loss = X_criterion(feature[data.features_unmask_eval],
                #                        data.x[data.features_unmask_eval]).item()

                # if args.dataset == 'pubmed' or args.dataset == 'cornell' or args.dataset == 'wisconsin':
                if args.dataset == 'pubmed':
                    rec_loss = X_criterion(feature[data.features_unmask_eval],
                                           data.x[data.features_unmask_eval]).item()
                else:
                    rec_loss = X_criterion(feature[test_mask],
                                           data.x[test_mask]).item()



                cor_loss = compute_corr(node_emb1,
                                        node_emb2).item()


                loss = rec_loss + cor_loss

                x1_loss = X_criterion(feature[data.feature_mask],
                                    data.groundtruth_x[data.feature_mask]).item()



                new_X = torch.where(F.sigmoid(feature) > args.X_thresholds , 1.0, 0.0)
                count_t = torch.where(new_X == data.groundtruth_x, 1.0, -1.0)
                _, count = torch.unique(count_t[data.feature_mask],
                                        return_counts=True)
                correct_count, wrong_count = count[-1].item(), count[0].item()
                mask_sum = correct_count + wrong_count
                correct_rate = correct_count / mask_sum
                wrong_rate = wrong_count / mask_sum



                if args.dataset == "pubmed":


                    if X_early_stopping_1.check(
                            [1.0,loss], epoch):
                        return True, t

                    if args.run == 'run':
                        wandb.log(
                            {f"{trial}-X_S1/eval/eval_X_unmask_rec_loss": loss,
                             f"{trial}-X_S1/eval/eval_X_mask_rec_Loss": x1_loss,
                             f"{trial}-X_S1/eval/correct_rate": correct_rate,
                             f"{trial}-X_S1/eval/wrong_rate": wrong_rate,
                            })

                    if epoch % 50 == 0:
                        current_time = time.time()
                        tt = current_time - t
                        t = current_time
                        # print(
                        #     's1: epoch:{}, rec_loss:{:.4f},cor_loss:{:.4f},con_loss1:{:.4f},con_loss2:{:.4f} mask_rec_Loss:{:.4f},X_correct_rate:{:.4f},X_wrong_rate:{:.4f}, time:{:.4f}s'.format(
                        #         epoch, rec_loss,cor_loss,con_loss1,con_loss2, x1_loss, correct_rate, wrong_rate,
                        #         tt))
                        print(
                            'X_s1: epoch:{}, rec_loss:{:.4f},cor_loss:{:.4f}, mask_rec_Loss:{:.4f},X_correct_rate:{:.4f},X_wrong_rate:{:.4f}, time:{:.4f}s'.format(
                                epoch, rec_loss, cor_loss,
                                x1_loss, correct_rate, wrong_rate,
                                tt))
                        # print(
                        #     's1: epoch:{}, eval_unmask_accuracy:{:.4f}, eval_unmask_precision:{:.4f},eval_unmask_recall:{:.4f},eval_unmask_f1:{:.4f}, time:{:.4f}s'.format(
                        #         epoch, eval_unmask_accuracy_value, eval_unmask_precision_value, eval_unmask_recall_value,
                        #         eval_unmask_f1score_value,
                        #         tt))
                        # print(
                        #     's1: epoch:{}, mask_accuracy:{:.4f}, mask_precision:{:.4f},mask_recall:{:.4f},mask_f1:{:.4f}, time:{:.4f}s'.format(
                        #         epoch, mask_accuracy, mask_precision, mask_recall, mask_f1,
                        #         tt))

                    return False, t
                else:


                    eval_unmask_pre = new_X[data.features_unmask_eval].clone()
                    eval_unmask_target = data.groundtruth_x[data.features_unmask_eval].clone()

                    eval_unmask_accuracy_value = accuracy_fuc(eval_unmask_pre, eval_unmask_target)
                    eval_unmask_precision_value = precision_fuc(eval_unmask_pre, eval_unmask_target)
                    eval_unmask_recall_value = recall_fuc(eval_unmask_pre, eval_unmask_target)
                    eval_unmask_f1score_value = f1score_fuc(eval_unmask_pre, eval_unmask_target)

                    mask_pre = new_X[data.feature_mask].clone()
                    mask_target = data.groundtruth_x[data.feature_mask].clone()

                    mask_accuracy_value = accuracy_fuc(mask_pre,mask_target)
                    mask_precision_value = precision_fuc(mask_pre,mask_target)
                    mask_recall_value = recall_fuc(mask_pre,mask_target)
                    mask_f1score_value = f1score_fuc(mask_pre,mask_target)



                    if X_early_stopping_1.check([1.0,loss], epoch):
                        return True, t


                    if args.run == 'run':
                        wandb.log({f"{trial}-X_S1/eval/eval_X_unmask_rec_loss": loss,
                                   f"{trial}-X_S1/eval/eval_X_mask_rec_Loss": x1_loss,
                                   f"{trial}-X_S1/eval/correct_rate": correct_rate,
                                   f"{trial}-X_S1/eval/wrong_rate": wrong_rate,
                                   f"{trial}-X_S1/eval/eval_unmask_accuracy": eval_unmask_accuracy_value,
                                   f"{trial}-X_S1/eval/eval_unmask_precision": eval_unmask_precision_value,
                                   f"{trial}-X_S1/eval/eval_unmask_recall": eval_unmask_recall_value,
                                   f"{trial}-X_S1/eval/eval_unmask_f1score": eval_unmask_f1score_value,
                                   f"{trial}-X_S1/eval/mask_accuracy": mask_accuracy_value,
                                   f"{trial}-X_S1/eval/mask_precision": mask_precision_value,
                                   f"{trial}-X_S1/eval/mask_recall": mask_recall_value,
                                   f"{trial}-X_S1/eval/mask_f1score": mask_f1score_value})

                    if epoch % 50 == 0:
                        current_time = time.time()
                        tt = current_time - t
                        t = current_time

                        print(
                            'X_s1: epoch:{}, rec_loss:{:.4f},cor_loss:{:.4f}, mask_rec_Loss:{:.4f},mask_accuracy:{:.4f}, mask_precision:{:.4f},mask_recall:{:.4f},mask_f1score:{:.4f}, time:{:.4f}s'.format(
                                epoch, rec_loss, cor_loss,x1_loss, mask_accuracy_value, mask_precision_value, mask_recall_value, mask_f1score_value,tt))
                        # print('mask_accuracy:{:.4f}, mask_precision:{:.4f},mask_recall:{:.4f},mask_f1score:{:.4f}, time:{:.4f}s'.format(
                        #         mask_accuracy_value, mask_precision_value, mask_recall_value, mask_f1score_value,
                        #         tt))
                    return False,t

            # show_gpu_allocated(3.3)

        t = time.time()
        for epoch in range(args.epochs):
            if args.run == 'run':
                wandb.log({f"{trial}-X_S1/epoch": epoch})
            # print(epoch)
            # optimizer_1.zero_grad()
            train_base()
            is_break, t = eval(t)
            if is_break:
                break
            # scheduler1.step()
        if args.dataset == "pubmed":
        # if True:
            print('X_s1:Loading {}th epoch'.format(X_early_stopping_1.best_epoch))
            X_model_S1.load_state_dict(X_early_stopping_1.best_state)
            X_model_S1.eval()  # 设置模型为评估模式，这将禁用 Dropout
            with torch.no_grad():  # 确保在评估过程中不会计算梯度
                new_feature, _, _ = X_model_S1(data.x, data.feature_mask,
                                             data.edge_index, data.edge_weight,
                                             num_neighbor=args.X_num_neighbor)
                # new_feature = F.sigmoid(new_feature)
                # new_feature = torch.where(new_feature > 0.5, 1.0, 0.0)
                # data.x, data.feature_mask

                x1_loss = X_criterion(new_feature[data.feature_mask],
                                    data.groundtruth_x[data.feature_mask]).item()
                if args.run == 'run':
                    wandb.log({f"{trial}-S1/best_X_mask_rec_Loss": x1_loss})
                print('X_s1:Loading {}th epoch,best_X_mask_rec_Loss:{:.4f}'.format(
                    X_early_stopping_1.best_epoch, x1_loss))
                # new_X = torch.where(F.sigmoid(new_feature) > 0.5, 1.0, 0.0)
                # new_X = new_feature

                print("X_x total num:{}, mask num:{}, mask not 0 num:{}, mask avg value:{}".format(
                        data.x.shape[0] * data.x.shape[1],
                        len(new_feature[data.feature_mask]),
                        new_feature[data.feature_mask].nonzero().shape[0],
                        torch.sum(new_feature[data.feature_mask]).item() / len(
                            new_feature[data.feature_mask])))
                if args.missing_link > 0.5:
                    new_feature = torch.where(
                        F.sigmoid(new_feature) > args.X_thresholds, new_feature, 0.0)
                    new_X = torch.where(data.feature_mask, new_feature, data.x)
                else:
                    new_X = torch.where(data.feature_mask, new_feature, data.x)
                    new_X = F.relu(new_X)
                # new_X = torch.where(F.sigmoid(new_feature) > args.thresholds, 1.0,
                #                     0.0)
                print("After: X_x total num:{}, mask num:{}, mask not 0 num:{}".format(
                    data.x.shape[0] * data.x.shape[1],
                    len(new_X[data.feature_mask]),
                    new_X[data.feature_mask].nonzero().shape[0]))

                # print("x total num:{}, mask num:{}, mask not 0 num:{}, mask avg value:{}".format(data.x.shape[0]*data.x.shape[1],len(new_X[data.feature_mask]), new_X[data.feature_mask].nonzero().shape[0],torch.sum(new_X[data.feature_mask]).item()/len(new_X[data.feature_mask])))

                # print(new_feature[data.feature_mask])
                return new_X.detach()
        else:

            print('X_s1:Loading {}th epoch'.format(X_early_stopping_1.best_epoch))
            X_model_S1.load_state_dict(X_early_stopping_1.best_state)
            X_model_S1.eval()  # 设置模型为评估模式，这将禁用 Dropout
            with torch.no_grad():  # 确保在评估过程中不会计算梯度
                new_feature,_,_ = X_model_S1(data.x, data.feature_mask,data.edge_index, data.edge_weight,num_neighbor=args.X_num_neighbor)
                # new_feature = F.sigmoid(new_feature)
                # new_feature = torch.where(new_feature > 0.5, 1.0, 0.0)
                # data.x, data.feature_mask
                x1_loss = X_criterion(new_feature[data.feature_mask],data.groundtruth_x[data.feature_mask]).item()
                if args.run == 'run':
                    wandb.log({f"{trial}-X_S1/best_X_mask_rec_Loss": x1_loss})

                print('X_s1:Loading {}th epoch,best_X_mask_rec_Loss:{:.4f}'.format(X_early_stopping_1.best_epoch,x1_loss))
                print("X_x total num:{}, mask num:{}, mask not 0 num:{}, mask avg value:{}".format(data.x.shape[0]*data.x.shape[1],len(new_feature[data.feature_mask]), new_feature[data.feature_mask].nonzero().shape[0],torch.sum(new_feature[data.feature_mask]).item()/len(new_feature[data.feature_mask])))

                new_feature = torch.where(F.sigmoid(new_feature) > args.X_thresholds, 1.0, 0.0)
                new_X = torch.where(data.feature_mask, new_feature, data.x)
                new_mask = new_X == torch.ones_like(new_X)
                new_mask = new_mask & data.feature_mask
                total_num = torch.sum(new_X[new_mask]).item()
                correct_num = torch.sum(new_X[new_mask] == data.groundtruth_x[new_mask]).item()
                wrong_num = total_num - correct_num
                ab_correct_num = correct_num - torch.sum(data.raw_adj).item()
                print("X_x total num:{}, mask num:{}, mask not 0 num:{},correct_num:{},ab correct num:{}, ab wrong num:{}".format(data.x.shape[0]*data.x.shape[1],len(new_X[data.feature_mask]), new_X[data.feature_mask].nonzero().shape[0],correct_num, ab_correct_num, wrong_num))


                return new_X.detach()

    def A_run_s1():
        # 初始化采样器
        sampler=BalancedSampler(data.raw_adj)
        # 获取测试集mask
        test_mask = sampler.get_test_mask()
        def train_base():
            # pos_weight = torch.ones([64])  # All weights are equal to 1
            # criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor(1))

            # show_gpu_allocated(3)
            A_model_S1.train()  # 设置模型为训练模式，这将启用 Dropout
            A_optimizer_1.zero_grad()
            new_A, A_emb1, A_emb2 = A_model_S1(data.x, data.feature_mask,
                                             data.edge_index, data.edge_weight,
                                             num_neighbor=args.num_neighbor)


            train_mask = sampler.get_train_mask()
            rec_loss = A_criterion(new_A[train_mask],
                                 data.raw_adj[train_mask])
            # rec_loss = A_criterion(new_A[data.adj_unmask_train],
            #                        data.raw_adj[data.adj_unmask_train])
            cor_loss = compute_corr(A_emb1,A_emb2)

            loss = rec_loss +cor_loss


            loss.backward()
            A_optimizer_1.step()
            if args.run == 'run':
                wandb.log({f"{trial}-A_S1/train/train_X_unmask_rec_loss": loss.item(), })

        def eval(t):
            A_model_S1.eval()  # 设置模型为评估模式，这将禁用 Dropout
            with torch.no_grad():  # 确保在评估过程中不会计算梯度

                new_A, A_emb1, A_emb2 = A_model_S1(data.x, data.feature_mask,
                                                 data.edge_index,
                                                 data.edge_weight,
                                                 num_neighbor=args.num_neighbor)



                rec_loss = A_criterion(new_A[test_mask],
                                     data.raw_adj[test_mask]).item()

                # rec_loss = A_criterion(new_A[data.adj_unmask_eval],
                #                        data.raw_adj[data.adj_unmask_eval]).item()

                cor_loss = compute_corr(A_emb1,
                                             A_emb2).item()


                loss = rec_loss + cor_loss

                adj_mask_loss = A_criterion(new_A[data.adj_mask],data.groundtruth_adj[data.adj_mask]).item()

                new_A = torch.where(F.sigmoid(new_A) > args.A_thresholds, 1.0, 0.0)
                # count_t = torch.where(new_A == data.groundtruth_adj, 1.0, -1.0)
                # _, count = torch.unique(count_t, return_counts=True)
                # correct_count, wrong_count = count[-1].item(), count[0].item()
                # mask_sum = correct_count + wrong_count
                # correct_rate = correct_count / mask_sum
                # wrong_rate = wrong_count / mask_sum
                #
                # eval_unmask_pre = new_A[data.adj_unmask_eval].clone()
                # eval_unmask_target = data.raw_adj[data.adj_unmask_eval].clone()
                #
                # eval_unmask_accuracy_value = accuracy_fuc(eval_unmask_pre,
                #                                           eval_unmask_target)
                # eval_unmask_precision_value = precision_fuc(eval_unmask_pre,
                #                                             eval_unmask_target)
                # eval_unmask_recall_value = recall_fuc(eval_unmask_pre,
                #                                       eval_unmask_target)
                # eval_unmask_f1score_value = f1score_fuc(eval_unmask_pre,
                #                                         eval_unmask_target)

                mask_pre = new_A.clone()
                mask_target = data.groundtruth_adj.clone()

                mask_accuracy_value = accuracy_fuc(mask_pre, mask_target)
                mask_precision_value = precision_fuc(mask_pre, mask_target)
                mask_recall_value = recall_fuc(mask_pre, mask_target)
                mask_f1score_value = f1score_fuc(mask_pre, mask_target)

                # if A_early_stopping_1.check([eval_unmask_f1score_value.cpu(),loss],epoch):
                if A_early_stopping_1.check([1.0,loss],epoch):
                    return True, t

                if args.run == 'run':
                    wandb.log({f"{trial}-A_S1/eval/eval_X_unmask_rec_loss": loss,
                               # f"{trial}-S1/eval/eval_X_mask_rec_Loss": x1_loss,
                               # f"{trial}-A_S1/eval/correct_rate": correct_rate,
                               # f"{trial}-A_S1/eval/wrong_rate": wrong_rate,
                               # f"{trial}-A_S1/eval/eval_unmask_accuracy": eval_unmask_accuracy_value,
                               # f"{trial}-A_S1/eval/eval_unmask_precision": eval_unmask_precision_value,
                               # f"{trial}-A_S1/eval/eval_unmask_recall": eval_unmask_recall_value,
                               # f"{trial}-A_S1/eval/eval_unmask_f1score": eval_unmask_f1score_value,
                               f"{trial}-A_S1/eval/mask_accuracy": mask_accuracy_value,
                               f"{trial}-A_S1/eval/mask_precision": mask_precision_value,
                               f"{trial}-A_S1/eval/mask_recall": mask_recall_value,
                               f"{trial}-A_S1/eval/mask_f1score": mask_f1score_value})

                if epoch % 50 == 0:
                    current_time = time.time()
                    tt = current_time - t
                    t = current_time

                    print(
                        'A_s1: epoch:{}, rec_loss:{:.4f},cor_loss:{:.4f},mask_rec_loss:{:.4f},mask_accuracy:{:.4f}, mask_precision:{:.4f},mask_recall:{:.4f},mask_f1score:{:.4f}, time:{:.4f}s'.format(
                            epoch, rec_loss, cor_loss,adj_mask_loss,
                            mask_accuracy_value, mask_precision_value,
                            mask_recall_value, mask_f1score_value,tt))

                return False, t



        t = time.time()
        for epoch in range(args.epochs):
            if args.run == 'run':
                wandb.log({f"{trial}-A_S1/epoch": epoch})
            train_base()
            is_break, t = eval(t)
            if is_break:
                break
            # scheduler2.step()

        print('A_s1:Loading {}th epoch'.format(A_early_stopping_1.best_epoch))
        A_model_S1.load_state_dict(A_early_stopping_1.best_state)
        A_model_S1.eval()  # 设置模型为评估模式，这将禁用 Dropout
        with torch.no_grad():  # 确保在评估过程中不会计算梯度
            new_A, _, _ = A_model_S1(data.x, data.feature_mask,
                                         data.edge_index, data.edge_weight,
                                         num_neighbor=args.num_neighbor)
            # new_feature = F.sigmoid(new_feature)
            # new_feature = torch.where(new_feature > 0.5, 1.0, 0.0)
            # data.x, data.feature_mask

            x1_loss = A_criterion(new_A[data.adj_mask], data.groundtruth_adj[data.adj_mask]).item()
            if args.run == 'run':
                wandb.log({f"{trial}-A_S1/best_adj_mask_rec_Loss": x1_loss})
            print('A_s1:Loading {}th epoch,best_adj_mask_rec_Loss:{:.4f}'.format(
                A_early_stopping_1.best_epoch, x1_loss))
            # new_X = torch.where(F.sigmoid(new_feature) > 0.5, 1.0, 0.0)
            # new_X = new_feature

            # new_A[~((data.groundtruth_adj - data.raw_adj) == 1.0)] = 0.0
            # new_A = torch.where(F.sigmoid(new_A) > 0.5,new_A,0.0)

            # new_A[data.degree_mask]=0.0
            # topk_values, topk_indices = torch.topk(new_A, k=1,dim=1)
            # mask = torch.zeros_like(new_A)
            # mask.view(-1)[topk_indices] = 1
            N_k = torch.round(data.node_degree * args.topk_weight).to(torch.int64)
            # N_k[data.node_degree==0.0] = 1
            new_A[data.raw_adj == 1.0] = -float('inf')
            new_A.fill_diagonal_(-float('inf'))
            mask_result = get_top_k_mask(new_A, N_k)
            new_A[mask_result] = 1.0
            new_A[~mask_result] = 0.0
            # new_adj = symmetrize_binary_matrix(new_A)
            # new_adj = new_A


            # new_A = torch.where(F.sigmoid(new_A) > args.A_thresholds,1.0,0.0)

            new_adj = torch.where(data.raw_adj == 1.0, data.raw_adj, new_A)

            new_mask = new_adj == torch.ones_like(new_adj)
            total_num = torch.sum(new_adj).item()
            correct_num = torch.sum(new_adj[new_mask] == data.groundtruth_adj[new_mask]).item()
            wrong_num = total_num - correct_num
            ab_correct_num = correct_num - torch.sum(data.raw_adj).item()




            print(
                "A_x total num:{}, true not 0 num:{}, pred not 0 num:{}, correct_num:{},ab correct num:{}, ab wrong num:{}".format(
                    data.raw_adj.shape[0] * data.raw_adj.shape[1],
                    torch.sum(data.raw_adj).item(), torch.sum(new_adj).item(),
                    correct_num, ab_correct_num, wrong_num))




            new_adj_t = filter_adjacency_matrix(train_val_labels, new_adj,train_val_idx)
            new_adj_t = torch.where(data.raw_adj == 1.0, data.raw_adj, new_adj_t)

            new_adj_t = new_adj_t.to(device)
            # new_adj = torch.where(data.raw_adj == 1.0, data.raw_adj, new_adj)
            print(new_adj_t.shape)
            new_mask = new_adj_t == torch.ones_like(new_adj_t)
            total_num = torch.sum(new_adj_t).item()
            correct_num = torch.sum(
                new_adj_t[new_mask] == data.groundtruth_adj[new_mask]).item()
            wrong_num = total_num - correct_num
            ab_correct_num = correct_num - torch.sum(data.raw_adj).item()

            print(
                "Filter: A_x total num:{}, true not 0 num:{}, pred not 0 num:{}, correct_num:{},ab correct num:{}, ab wrong num:{}".format(
                    data.raw_adj.shape[0] * data.raw_adj.shape[1],
                    torch.sum(data.raw_adj).item(),
                    torch.sum(new_adj).item(),
                    correct_num, ab_correct_num, wrong_num))



        return new_adj.detach()



    if args.dataset == 'pubmed':
        print(X_model_S1)
        new_feature = X_run_s1()
        new_edge_index, new_edge_weight = dense_to_sparse(data.raw_adj)


    else:
        print(X_model_S1)
        new_feature = X_run_s1()
        print(A_model_S1)
        new_adj = A_run_s1()


        new_edge_index, new_edge_weight = dense_to_sparse(new_adj)


    return new_feature,new_edge_index,new_edge_weight


def main_TS_Decoupled_IMPXA_UPDATE_IterGCN_3SCL_1CE_SUM_Step2V4(trial,new_feature,new_edge_index,new_edge_weight):
    data = load_data(args.dataset,
                           missing_link= args.missing_link,
                           missing_feature = args.missing_feature,
                           trial=trial,
                           split_datasets_type=args.split_datasets_type,
                           normalize_features=args.normalize_features)

    data.x = new_feature
    data.edge_index = new_edge_index
    data.edge_weight = new_edge_weight
    data.raw_x = data.x.clone()

    data1 = data.clone()
    data1 = data1.to(device)
    data.train_mask = data.val_mask = data.test_mask = None
    data = train_test_split_edges(data)

    data = data.to(device)

    labels = data1.y
    idx_train = torch.where(data1.train_mask==True)[0]
    idx_val = torch.where(data1.val_mask == True)[0]
    idx_test = torch.where(data1.test_mask == True)[0]
    num_classes = len(set(data1.y.tolist()))
    train_val_labels = labels.to(device)
    train_val_idx = torch.cat([idx_train, idx_val]).to(device)


    if args.dataset == "pubmed":
        X_criterion = torch.nn.MSELoss(reduction="mean").to(device)
    else:
        if args.X_num_neighbor_2 == 0:
            X_criterion = nn.BCEWithLogitsLoss().to(device)
        else:
            X_criterion = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(args.X_num_neighbor_2)).to(device)




    XA_model_S2 = GAE(GCNEncoder(in_channels=data.x.shape[1], out_channels=args.hidden,dropout=args.dropout),
                      decoder2=Two_MLP_Layer(in_size =args.hidden,hidden_size=2*args.hidden,out_size=data.x.shape[1],dropout=args.dropout),
                      criterion=X_criterion).to(device)

    XA_optimizer_2 = torch.optim.Adam(XA_model_S2.parameters(), lr=args.lr1, weight_decay=args.weight_decay)
    # total_steps = args.epochs
    # warmup_steps = int(0.1 * total_steps)  # 10% of total steps for warmup
    # scheduler = WarmupCosineSchedule(XA_optimizer_2, warmup_steps=warmup_steps,
    #                                   t_total=total_steps)

    stopping_args = Stop_args(patience=args.patience, max_epochs=args.epochs)

    XA_early_stopping_2 = EarlyStopping(XA_model_S2, **stopping_args)

    supcloss = SupConLoss(temperature=args.tau)
    def XA_run_s2():
        # 初始化采样器
        Masksampler = MaskBalancedSampler(data1.x, data1.feature_mask)
        # 获取测试集mask
        test_mask = Masksampler.get_test_mask()
        def train_base():
            # show_gpu_allocated(3)
            XA_model_S2.train()  # 设置模型为训练模式，这将启用 Dropout
            XA_optimizer_2.zero_grad()
            z = XA_model_S2(data.x,data.train_pos_edge_index)
            loss2 = XA_model_S2.x_recon_loss(z, data.x,
                                             data.features_unmask_train)

            # if args.dataset == "pubmed":
            #     loss2 = XA_model_S2.x_recon_loss(z, data.x,data.features_unmask_train)
            #
            # else:
            #     train_mask = Masksampler.get_train_mask()
            #     loss2 = XA_model_S2.x_recon_loss(z, data.x,train_mask)

            loss1 = XA_model_S2.recon_loss(z, data.train_pos_edge_index)
            loss3 = supcloss(z[idx_train],labels[idx_train])
            # loss3 = 0

            loss =  loss1 + loss2 +loss3

            loss.backward()
            XA_optimizer_2.step()
            if epoch % 50 == 0:
                print('s2:train_epoch:{} , A_rec_loss: {:.4f}, X_rec_loss: {:.4f}, sup_loss: {:.4f}'.format(epoch, loss1.item(), loss2.item(),loss3))
            if args.run == 'run':
                wandb.log({
                f"{trial}-A_S1/train/tall_loss": loss.item(),
                f"{trial}-A_S1/train/A_rec_loss": loss1.item(),
                f"{trial}-A_S1/train/X_rec_loss": loss2.item(),
                f"{trial}-A_S1/train/supcloss": loss3})

        def eval(t):
            XA_model_S2.eval()  # 设置模型为评估模式，这将禁用 Dropout
            with torch.no_grad():  # 确保在评估过程中不会计算梯度
                z = XA_model_S2.encode(data.x, data.train_pos_edge_index)
                new_adj = XA_model_S2.decode(z)
                new_feature = XA_model_S2.decode2(z)
                loss2 = XA_model_S2.x_recon_loss(z, data.x,
                                                 data.features_unmask_eval)

                # if args.dataset == "pubmed":
                #     loss2 = XA_model_S2.x_recon_loss(z, data.x,data.features_unmask_eval)
                #
                # else:
                #     # train_mask = Masksampler.get_train_mask()
                #     loss2 = XA_model_S2.x_recon_loss(z, data.x, test_mask)
                loss1 = XA_model_S2.recon_loss(z, data.test_pos_edge_index,data.test_neg_edge_index)

                mask_x_rec_loss = XA_model_S2.x_recon_loss(z, data.groundtruth_x,
                                           data.feature_mask)

                loss3 = supcloss(z[idx_val], labels[idx_val])
                # loss3 = 0

                loss = loss1 +  loss2 + loss3
                auc, ap = XA_model_S2.test(z, data.test_pos_edge_index, data.test_neg_edge_index)
                # acc,pre,rec,f1 = XA_model_S2.x_test(z, data.features_unmask_eval,data.x)
                if XA_early_stopping_2.check([auc,loss.item()], epoch):
                    return True, t

                if epoch % 50 == 0:
                    current_time = time.time()
                    tt = current_time - t
                    t = current_time

                    print(
                        's2:eval_epoch:{} , A_rec_loss: {:.4f}, X_rec_loss: {:.4f},mask_x_rec_loss:{:.4f}, sup_loss: {:.4f}, AUC: {:.4f}, AP: {:.4f}'.format(
                            epoch, loss1.item(), loss2.item(),mask_x_rec_loss.item(), loss3,auc, ap))
                if args.run == 'run':
                    wandb.log({
                        f"{trial}-A_S1/eval/tall_loss": loss.item(),
                        f"{trial}-A_S1/eval/A_rec_loss": loss1.item(),
                        f"{trial}-A_S1/eval/X_rec_loss": loss2.item(),
                        f"{trial}-A_S1/eval/supcloss": loss3})

                return False, t


        t = time.time()
        for epoch in range(args.epochs):

            train_base()
            is_break, t = eval(t)
            if is_break:
                break
            # scheduler.step()
        # if args.dataset == "pubmed":
        if True:


            print('IterXA_s2:Loading {}th epoch'.format(XA_early_stopping_2.best_epoch))
            XA_model_S2.load_state_dict(XA_early_stopping_2.best_state)
            Z = XA_model_S2(data1.x, data1.edge_index)
            a_rec_loss = XA_model_S2.recon_loss(Z, data.test_pos_edge_index,data.test_neg_edge_index)
            x_rec_loss = XA_model_S2.x_recon_loss(Z, data.groundtruth_x,data.feature_mask)
            if args.run == 'run':
                wandb.log({f"{trial}-S2/best_X_mask_rec_Loss": x_rec_loss,f"{trial}-S2/best_A_mask_rec_Loss": a_rec_loss})
            print('IterXA_s2:Loading {}th epoch,best_X_mask_rec_Loss:{:.4f},best_A_mask_rec_Loss:{:.4f}'.format(XA_early_stopping_2.best_epoch, x_rec_loss,a_rec_loss))

            ############################################################################

            new_feature = XA_model_S2.decode2(Z)

            if args.dataset != "pubmed":
                new_feature= torch.where(F.sigmoid(new_feature) > args.X_thresholds_2, 1.0, 0.0)
                new_X = torch.where(data.feature_mask, new_feature, data.x)

                new_mask = new_X == torch.ones_like(new_X)
                new_mask = new_mask & data.feature_mask
                total_num = torch.sum(new_X[new_mask]).item()
                correct_num = torch.sum(new_X[new_mask] == data.groundtruth_x[new_mask]).item()
                wrong_num = total_num - correct_num
                # ab_correct_num = correct_num - torch.sum(data.x).item()
                print("X_x total num:{}, mask num:{}, mask not 0 num:{},correct_num:{}, ab wrong num:{}".format(
                        data.x.shape[0] * data.x.shape[1],
                        len(new_X[data.feature_mask]),
                        new_X[data.feature_mask].nonzero().shape[0],
                        correct_num,
                        wrong_num))
            else:
                if args.missing_link > 0.5:
                    new_feature = torch.where(
                        F.sigmoid(new_feature) > args.X_thresholds_2, new_feature, 0.0)
                    new_X = torch.where(data.feature_mask, new_feature, data.x)
                else:
                    new_X = torch.where(data.feature_mask, new_feature, data.x)
                    new_X = F.relu(new_X)
                print("X_x total num:{}, mask num:{}, mask not 0 num:{}".format(
                    data.x.shape[0] * data.x.shape[1],
                    len(new_X[data.feature_mask]),
                    new_X[data.feature_mask].nonzero().shape[0]))

            ##############################################################################
            new_A = XA_model_S2.decode(Z)

            # TOPK
            N_k = torch.round(data.node_degree * args.topk_weight).to(torch.int64)
            # N_k[data.node_degree==0.0] = 1
            new_A[data.raw_adj == 1.0] = -float('inf')
            new_A.fill_diagonal_(-float('inf'))
            mask_result = get_top_k_mask(new_A, N_k)
            new_A[mask_result] = 1.0
            new_A[~mask_result] = 0.0
            new_adj = new_A

            # 未filter
            new_adj_t = torch.where(data.raw_adj == 1.0, data.raw_adj, new_adj)
            new_mask = new_adj_t == torch.ones_like(new_adj_t)
            total_num = torch.sum(new_adj_t).item()
            correct_num = torch.sum(new_adj_t[new_mask] == data.groundtruth_adj[new_mask]).item()
            wrong_num = total_num - correct_num
            ab_correct_num = correct_num - torch.sum(data.raw_adj).item()
            print("1A_x total num:{}, true not 0 num:{}, pred not 0 num:{}, correct_num:{},ab correct num:{}, ab wrong num:{}".format(
                    data.raw_adj.shape[0] * data.raw_adj.shape[1],
                    torch.sum(data.raw_adj).item(),
                    torch.sum(new_adj).item(),
                    correct_num, ab_correct_num, wrong_num))

            # 已filter
            new_adj = filter_adjacency_matrix(train_val_labels, new_adj,train_val_idx)
            new_adj = torch.where(data.raw_adj == 1.0, data.raw_adj, new_adj)
            new_mask = new_adj == torch.ones_like(new_adj)
            total_num = torch.sum(new_adj).item()
            correct_num = torch.sum(new_adj[new_mask] == data.groundtruth_adj[new_mask]).item()
            wrong_num = total_num - correct_num
            ab_correct_num = correct_num - torch.sum(data.raw_adj).item()
            print(
                "Filter:2A_x total num:{}, true not 0 num:{}, pred not 0 num:{}, correct_num:{},ab correct num:{}, ab wrong num:{}".format(
                    data.raw_adj.shape[0] * data.raw_adj.shape[1],
                    torch.sum(data.raw_adj).item(),
                    torch.sum(new_adj).item(),
                    correct_num, ab_correct_num, wrong_num))


            return new_X.detach(),new_adj.detach()

    new_feature2, new_adj2 = XA_run_s2()
    new_edge_index2, new_edge_weight2 = dense_to_sparse(new_adj2)
    return new_feature2,new_edge_index2, new_edge_weight2
def main_TS_Decoupled_IMPXA_UPDATE_IterGCN_3SCL_1CE_SUM_Step2V6(trial,new_feature,new_edge_index,new_edge_weight):
    data = load_data(args.dataset,
                           missing_link= args.missing_link,
                           missing_feature = args.missing_feature,
                           trial=trial,
                           split_datasets_type=args.split_datasets_type,
                           normalize_features=args.normalize_features)

    data.x = new_feature
    data.edge_index = new_edge_index
    data.edge_weight = new_edge_weight
    # data.raw_x = data.x.clone()

    data1 = data.clone()
    data1 = data1.to(device)
    data.train_mask = data.val_mask = data.test_mask = None
    data = train_test_split_edges(data)

    data = data.to(device)

    labels = data1.y
    idx_train = torch.where(data1.train_mask==True)[0]
    idx_val = torch.where(data1.val_mask == True)[0]
    idx_test = torch.where(data1.test_mask == True)[0]
    num_classes = len(set(data1.y.tolist()))
    train_val_labels = labels.to(device)
    train_val_idx = torch.cat([idx_train, idx_val]).to(device)


    if args.dataset == "pubmed":
        X_criterion = torch.nn.MSELoss(reduction="mean").to(device)
    else:
        if args.X_num_neighbor_2 == 0:
            X_criterion = nn.BCEWithLogitsLoss().to(device)
        else:
            X_criterion = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(args.X_num_neighbor_2)).to(device)




    XA_model_S2 = GAE(GCNEncoder(in_channels=data.x.shape[1], out_channels=args.hidden,dropout=args.dropout),
                      decoder2=Two_MLP_Layer(in_size =args.hidden,hidden_size=2*args.hidden,out_size=data.x.shape[1],dropout=args.dropout),
                      criterion=X_criterion).to(device)

    XA_optimizer_2 = torch.optim.Adam(XA_model_S2.parameters(), lr=args.lr1, weight_decay=args.weight_decay)
    # total_steps = args.epochs
    # warmup_steps = int(0.1 * total_steps)  # 10% of total steps for warmup
    # scheduler = WarmupCosineSchedule(XA_optimizer_2, warmup_steps=warmup_steps,
    #                                   t_total=total_steps)

    stopping_args = Stop_args(patience=args.patience, max_epochs=args.epochs)

    XA_early_stopping_2 = EarlyStopping(XA_model_S2, **stopping_args)

    supcloss = SupConLoss(temperature=args.tau)
    def XA_run_s2(data,data1):

        def train_base(data):
            # 初始化采样器

            # show_gpu_allocated(3)
            XA_model_S2.train()  # 设置模型为训练模式，这将启用 Dropout
            XA_optimizer_2.zero_grad()
            z = XA_model_S2(data.x,data.train_pos_edge_index)
            loss2 = XA_model_S2.x_recon_loss(z, data.raw_x,
                                             data.features_unmask_train)

            # if args.dataset == "pubmed":
            #     loss2 = XA_model_S2.x_recon_loss(z, data.x,data.features_unmask_train)
            #
            # else:
            #     train_mask = Masksampler.get_train_mask()
            #     loss2 = XA_model_S2.x_recon_loss(z, data.x,train_mask)

            loss1 = XA_model_S2.recon_loss(z, data.train_pos_edge_index)
            loss3 = supcloss(z[idx_train],labels[idx_train])
            # loss3 = 0

            loss =  loss1 + loss2 +loss3

            loss.backward()
            XA_optimizer_2.step()
            if epoch % 50 == 0:
                print('s2:train_epoch:{} , A_rec_loss: {:.4f}, X_rec_loss: {:.4f}, sup_loss: {:.4f}'.format(epoch, loss1.item(), loss2.item(),loss3.item()))
            if args.run == 'run':
                wandb.log({
                f"{trial}-A_S1/train/tall_loss": loss.item(),
                f"{trial}-A_S1/train/A_rec_loss": loss1.item(),
                f"{trial}-A_S1/train/X_rec_loss": loss2.item(),
                f"{trial}-A_S1/train/supcloss": loss3.item()})

        def eval(t,data):
            XA_model_S2.eval()  # 设置模型为评估模式，这将禁用 Dropout
            with torch.no_grad():  # 确保在评估过程中不会计算梯度
                z = XA_model_S2.encode(data.x, data.train_pos_edge_index)
                new_adj = XA_model_S2.decode(z)
                new_feature = XA_model_S2.decode2(z)
                # 获取测试集mask
                # test_mask = Masksampler.get_test_mask()
                loss2 = XA_model_S2.x_recon_loss(z, data.raw_x,data.features_unmask_eval)

                # if args.dataset == "pubmed":
                #     loss2 = XA_model_S2.x_recon_loss(z, data.x,data.features_unmask_eval)
                #
                # else:
                #     # train_mask = Masksampler.get_train_mask()
                #     loss2 = XA_model_S2.x_recon_loss(z, data.x, test_mask)
                loss1 = XA_model_S2.recon_loss(z, data.test_pos_edge_index,data.test_neg_edge_index)

                mask_x_rec_loss = XA_model_S2.x_recon_loss(z, data.groundtruth_x,data.feature_mask)

                loss3 = supcloss(z[idx_val], labels[idx_val])
                # loss3 = 0

                loss = loss1 +  loss2 + loss3
                auc, ap = XA_model_S2.test(z, data.test_pos_edge_index, data.test_neg_edge_index)
                # acc,pre,rec,f1 = XA_model_S2.x_test(z, data.features_unmask_eval,data.x)
                if XA_early_stopping_2.check([auc,loss.item()], epoch):
                    return True, t

                if epoch % 50 == 0:
                    current_time = time.time()
                    tt = current_time - t
                    t = current_time

                    print(
                        's2:eval_epoch:{} , A_rec_loss: {:.4f}, X_rec_loss: {:.4f},mask_x_rec_loss:{:.4f}, sup_loss: {:.4f}, AUC: {:.4f}, AP: {:.4f}'.format(
                            epoch, loss1.item(), loss2.item(),mask_x_rec_loss.item(), loss3.item(),auc, ap))
                if args.run == 'run':
                    wandb.log({
                        f"{trial}-A_S1/eval/tall_loss": loss.item(),
                        f"{trial}-A_S1/eval/A_rec_loss": loss1.item(),
                        f"{trial}-A_S1/eval/X_rec_loss": loss2.item(),
                        f"{trial}-A_S1/eval/supcloss": loss3.item()})

                return False, t

        # Masksampler = MaskBalancedSampler(data1.x, data1.feature_mask)

        t = time.time()
        for epoch in range(args.epochs):

            train_base(data)
            is_break, t = eval(t,data)
            if is_break:
                break
            if epoch % 50 == 0:
                # 使用模型，得到最新的 X A,得到拼接后 X A
                if args.dataset!="pubmed":
                    XA_model_S2.load_state_dict(XA_early_stopping_2.best_state)
                    XA_model_S2.eval()  # 设置模型为评估模式，这将禁用 Dropout
                    with torch.no_grad():  # 确保在评估过程中不会计算梯度
                        ##############################################################################
                        Z = XA_model_S2(data1.x, data1.edge_index)
                        new_feature = XA_model_S2.decode2(Z)
                        if args.dataset != "pubmed":
                            new_feature = torch.where(F.sigmoid(new_feature) > args.X_thresholds_2, 1.0, 0.0)

                            # new_X = torch.relu(data.raw_x * (
                            #             data.feature_mask == False) + 0.8 * data.x * (
                            #                              data.feature_mask == True) + 0.2 * new_feature * (
                            #                              data.feature_mask == True))

                            new_X = torch.where(data.feature_mask, new_feature, data.x)

                            new_mask = new_X == torch.ones_like(new_X)
                            new_mask = new_mask & data.feature_mask
                            total_num = torch.sum(new_X[new_mask]).item()
                            correct_num = torch.sum(
                                new_X[new_mask] == data.groundtruth_x[new_mask]).item()
                            wrong_num = total_num - correct_num
                            # ab_correct_num = correct_num - torch.sum(data.x).item()
                            print(
                                "X_x total num:{}, mask num:{}, mask not 0 num:{},correct_num:{}, ab wrong num:{}".format(
                                    data.x.shape[0] * data.x.shape[1],
                                    len(new_X[data.feature_mask]),
                                    new_X[data.feature_mask].nonzero().shape[0],
                                    correct_num,
                                    wrong_num))
                        else:
                            new_X = torch.where(data.feature_mask, new_feature, data.x)
                            new_X = F.relu(new_X)
                            # new_X = torch.relu(data.raw_x * (
                            #         data.feature_mask == False) + 0.8 * data.x * (
                            #                            data.feature_mask == True) + 0.2 * new_feature * (
                            #                            data.feature_mask == True))

                            print(
                                "X_x total num:{}, mask num:{}, mask not 0 num:{}".format(
                                    data.x.shape[0] * data.x.shape[1],
                                    len(new_X[data.feature_mask]),
                                    new_X[data.feature_mask].nonzero().shape[0]))

                        ##############################################################################
                        new_A = XA_model_S2.decode(Z)
                        new_A = new_A.detach()
                        # TOPK
                        N_k = torch.round(data.node_degree * args.topk_weight).to(torch.int64)
                        # N_k[data.node_degree==0.0] = 1
                        new_A[data.raw_adj == 1.0] = -float('inf')
                        new_A.fill_diagonal_(-float('inf'))
                        mask_result = get_top_k_mask(new_A, N_k)
                        new_A[mask_result] = 1.0
                        new_A[~mask_result] = 0.0
                        new_adj = new_A

                        # 未filter
                        new_adj_t = torch.where(data.raw_adj == 1.0, data.raw_adj,new_adj)
                        new_mask = new_adj_t == torch.ones_like(new_adj_t)
                        total_num = torch.sum(new_adj_t).item()
                        correct_num = torch.sum(new_adj_t[new_mask] == data.groundtruth_adj[new_mask]).item()
                        wrong_num = total_num - correct_num
                        ab_correct_num = correct_num - torch.sum(data.raw_adj).item()
                        print(
                            "1A_x total num:{}, true not 0 num:{}, pred not 0 num:{}, correct_num:{},ab correct num:{}, ab wrong num:{}".format(
                                data.raw_adj.shape[0] * data.raw_adj.shape[1],
                                torch.sum(data.raw_adj).item(),
                                torch.sum(new_adj).item(),
                                correct_num, ab_correct_num, wrong_num))

                        # 已filter
                        new_adj = filter_adjacency_matrix(train_val_labels, new_adj,
                                                          train_val_idx)
                        new_adj = torch.where(data.raw_adj == 1.0, data.raw_adj,new_adj)
                        new_mask = new_adj == torch.ones_like(new_adj)
                        total_num = torch.sum(new_adj).item()
                        correct_num = torch.sum(
                            new_adj[new_mask] == data.groundtruth_adj[new_mask]).item()
                        wrong_num = total_num - correct_num
                        ab_correct_num = correct_num - torch.sum(data.raw_adj).item()
                        print(
                            "Filter:2A_x total num:{}, true not 0 num:{}, pred not 0 num:{}, correct_num:{},ab correct num:{}, ab wrong num:{}".format(
                                data.raw_adj.shape[0] * data.raw_adj.shape[1],
                                torch.sum(data.raw_adj).item(),
                                torch.sum(new_adj).item(),
                                correct_num, ab_correct_num, wrong_num))


                        # 更新输入 和 标签
                        data1.x = new_X.detach()
                        # data1.raw_adj = new_adj
                        data1.edge_index, data1.edge_weight = dense_to_sparse(new_adj.detach())
                        data = data1.clone()
                        data = train_test_split_edges(data)
                # Masksampler = MaskBalancedSampler(data1.x, data1.feature_mask)

            # scheduler.step()
        # if args.dataset == "pubmed":
        if True:


            print('IterXA_s2:Loading {}th epoch'.format(XA_early_stopping_2.best_epoch))
            XA_model_S2.load_state_dict(XA_early_stopping_2.best_state)
            Z = XA_model_S2(data1.x, data1.edge_index)
            a_rec_loss = XA_model_S2.recon_loss(Z, data.test_pos_edge_index,data.test_neg_edge_index)
            x_rec_loss = XA_model_S2.x_recon_loss(Z, data.groundtruth_x,data.feature_mask)
            if args.run == 'run':
                wandb.log({f"{trial}-S2/best_X_mask_rec_Loss": x_rec_loss,f"{trial}-S2/best_A_mask_rec_Loss": a_rec_loss})
            print('IterXA_s2:Loading {}th epoch,best_X_mask_rec_Loss:{:.4f},best_A_mask_rec_Loss:{:.4f}'.format(XA_early_stopping_2.best_epoch, x_rec_loss,a_rec_loss))

            ############################################################################

            new_feature = XA_model_S2.decode2(Z)

            if args.dataset != "pubmed":
                new_feature= torch.where(F.sigmoid(new_feature) > args.X_thresholds_2, 1.0, 0.0)
                new_X = torch.where(data.feature_mask, new_feature, data.x)

                new_mask = new_X == torch.ones_like(new_X)
                new_mask = new_mask & data.feature_mask
                total_num = torch.sum(new_X[new_mask]).item()
                correct_num = torch.sum(new_X[new_mask] == data.groundtruth_x[new_mask]).item()
                wrong_num = total_num - correct_num
                # ab_correct_num = correct_num - torch.sum(data.x).item()
                print("X_x total num:{}, mask num:{}, mask not 0 num:{},correct_num:{}, ab wrong num:{}".format(
                        data.x.shape[0] * data.x.shape[1],
                        len(new_X[data.feature_mask]),
                        new_X[data.feature_mask].nonzero().shape[0],
                        correct_num,
                        wrong_num))
            else:
                if args.missing_link > 0.5:
                    new_feature = torch.where(
                        F.sigmoid(new_feature) > args.X_thresholds_2, new_feature, 0.0)
                    new_X = torch.where(data.feature_mask, new_feature, data.x)
                else:
                    new_X = torch.where(data.feature_mask, new_feature, data.x)
                    new_X = F.relu(new_X)
                print("X_x total num:{}, mask num:{}, mask not 0 num:{}".format(
                    data.x.shape[0] * data.x.shape[1],
                    len(new_X[data.feature_mask]),
                    new_X[data.feature_mask].nonzero().shape[0]))

            ##############################################################################
            new_A = XA_model_S2.decode(Z)

            # TOPK
            N_k = torch.round(data.node_degree * args.topk_weight).to(torch.int64)
            # N_k[data.node_degree==0.0] = 1
            new_A[data.raw_adj == 1.0] = -float('inf')
            new_A.fill_diagonal_(-float('inf'))
            mask_result = get_top_k_mask(new_A, N_k)
            new_A[mask_result] = 1.0
            new_A[~mask_result] = 0.0
            new_adj = new_A

            # 未filter
            new_adj_t = torch.where(data.raw_adj == 1.0, data.raw_adj, new_adj)
            new_mask = new_adj_t == torch.ones_like(new_adj_t)
            total_num = torch.sum(new_adj_t).item()
            correct_num = torch.sum(new_adj_t[new_mask] == data.groundtruth_adj[new_mask]).item()
            wrong_num = total_num - correct_num
            ab_correct_num = correct_num - torch.sum(data.raw_adj).item()
            print("1A_x total num:{}, true not 0 num:{}, pred not 0 num:{}, correct_num:{},ab correct num:{}, ab wrong num:{}".format(
                    data.raw_adj.shape[0] * data.raw_adj.shape[1],
                    torch.sum(data.raw_adj).item(),
                    torch.sum(new_adj).item(),
                    correct_num, ab_correct_num, wrong_num))

            # 已filter
            new_adj = filter_adjacency_matrix(train_val_labels, new_adj,train_val_idx)
            new_adj = torch.where(data.raw_adj == 1.0, data.raw_adj, new_adj)
            new_mask = new_adj == torch.ones_like(new_adj)
            total_num = torch.sum(new_adj).item()
            correct_num = torch.sum(new_adj[new_mask] == data.groundtruth_adj[new_mask]).item()
            wrong_num = total_num - correct_num
            ab_correct_num = correct_num - torch.sum(data.raw_adj).item()
            print(
                "Filter:2A_x total num:{}, true not 0 num:{}, pred not 0 num:{}, correct_num:{},ab correct num:{}, ab wrong num:{}".format(
                    data.raw_adj.shape[0] * data.raw_adj.shape[1],
                    torch.sum(data.raw_adj).item(),
                    torch.sum(new_adj).item(),
                    correct_num, ab_correct_num, wrong_num))


            return new_X.detach(),new_adj.detach()

    new_feature2, new_adj2 = XA_run_s2(data,data1)
    new_edge_index2, new_edge_weight2 = dense_to_sparse(new_adj2)
    return new_feature2,new_edge_index2, new_edge_weight2


def main_TS_Decoupled_IMPXA_UPDATE_IterGCN_3SCL_1CE_SUM_Step3V3(trial,new_feature,new_edge_index,new_edge_weight,new_feature2,new_edge_index2,new_edge_weight2):
    data = load_data(args.dataset,
                           missing_link= args.missing_link,
                           missing_feature = args.missing_feature,
                           trial=trial,
                           split_datasets_type=args.split_datasets_type,
                           normalize_features=args.normalize_features)

    data = data.to(device)

    labels = data.y
    idx_train = torch.where(data.train_mask==True)[0]
    idx_val = torch.where(data.val_mask == True)[0]
    idx_test = torch.where(data.test_mask == True)[0]
    num_classes = len(set(data.y.tolist()))



    XA_model_S2 = S2_Decoupled_GCN_3SCL_1CE_SUM_V3(num_nodes=data.num_nodes, in_size=data.x.shape[1],
                    hidden_size=args.hidden,
                    num_layer=args.gcn_num_layer,
                    out_size=num_classes,
                    dropout=args.dropout, device=device,lxw =args.lxw,law = args.law, lzw = args.lzw).to(device)

    XA_optimizer_2 = torch.optim.Adam(XA_model_S2.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # total_steps = args.epochs
    # warmup_steps = int(0.1 * total_steps)  # 10% of total steps for warmup
    # scheduler = WarmupCosineSchedule(XA_optimizer_2, warmup_steps=warmup_steps,
    #                                   t_total=total_steps)

    stopping_args3 = Stop_args(patience=args.patience, max_epochs=args.epochs)

    XA_early_stopping_2 = EarlyStopping(XA_model_S2, **stopping_args3)

    supcloss = SupConLoss(temperature=args.tau)

    # criterion = LabelSmoothingCrossEntropy()

    def XA_run_s2():

        def train_base():
            # show_gpu_allocated(3)
            XA_model_S2.train()  # 设置模型为训练模式，这将启用 Dropout
            XA_optimizer_2.zero_grad()
            attr_emb,stru_emb,doub_emb,output = XA_model_S2(new_feature, new_edge_index,new_edge_weight,new_feature2, new_edge_index2,new_edge_weight2)

            ce_loss = F.nll_loss(output[idx_train], labels[idx_train])
            loss1 = supcloss(attr_emb[idx_train], labels[idx_train])
            loss2 = supcloss(stru_emb[idx_train], labels[idx_train])
            loss3 = supcloss(doub_emb[idx_train], labels[idx_train])

            loss = ce_loss +  args.scl_weight*(loss1 + loss2 +  loss3)

            acc_train = accuracy(output[idx_train], labels[idx_train])

            loss.backward()


            XA_optimizer_2.step()


        def eval(t):
            XA_model_S2.eval()  # 设置模型为评估模式，这将禁用 Dropout
            with torch.no_grad():  # 确保在评估过程中不会计算梯度

                attr_emb, stru_emb, doub_emb, output = XA_model_S2(new_feature,new_edge_index, new_edge_weight,new_feature2, new_edge_index2,new_edge_weight2)


                loss_val1 = (F.nll_loss(output[idx_val], labels[idx_val])).item()
                loss_train = (F.nll_loss(output[idx_train], labels[idx_train])).item()
                loss1 = supcloss(attr_emb[idx_val], labels[idx_val])
                loss2 = supcloss(stru_emb[idx_val], labels[idx_val])
                loss3 = supcloss(doub_emb[idx_val], labels[idx_val])

                loss4 = 0
                loss5 = 0

                acc_val = accuracy(output[idx_val], labels[idx_val]).item()
                if XA_early_stopping_2.check([acc_val, loss_val1], epoch):
                    return True, t
                acc_test = accuracy(output[idx_test], labels[idx_test]).item()
                acc_train = accuracy(output[idx_train], labels[idx_train]).item()
                if args.run == 'run':
                    wandb.log({f'{trial}-validation/ce_loss': loss_val1,
                         f'{trial}-validation/acc_val': acc_val,
                         f'{trial}-validation/acc_test': acc_test})


                if epoch % 50 == 0:
                    current_time = time.time()
                    tt = current_time - t
                    t = current_time
                    print(
                        's2:epoch:{} , acc_train:{:.4f}, acc_val:{:.4f} , acc_test:{:.4f},train_celoss:{:.4f},val_celoss:{:.4f},'
                        'attr_suploss:{:.4f}, stru_suploss:{:.4f} , doub_suploss:{:.4f},newx_suploss:{:.4f},newa_suploss:{:.4f}, time:{:.4f}s'.format(
                            epoch, acc_train,acc_val, acc_test, loss_train,loss_val1,loss1,loss2,loss3,loss4,loss5,  tt))
                return False, t

        def test():
            XA_model_S2.eval()
            with torch.no_grad():
                attr_emb, stru_emb, doub_emb, output = XA_model_S2(
                    new_feature,
                    new_edge_index, new_edge_weight,new_feature2, new_edge_index2,new_edge_weight2)

                loss_test1 = F.nll_loss(output[idx_test], labels[idx_test])
                loss1 = supcloss(attr_emb[idx_test], labels[idx_test])
                loss2 = supcloss(stru_emb[idx_test], labels[idx_test])
                loss3 = supcloss(doub_emb[idx_test], labels[idx_test])

                test_acc = accuracy(output[idx_test], labels[idx_test])

                print("Test set results:",
                      "loss= {:.4f}".format(loss_test1.item()),
                      "accuracy= {:.4f}".format(test_acc.item()),
                      )

                if args.run == 'run':
                    wandb.log(
                        {
                         "test/ce_loss": loss_test1.item(),
                         'test/acc': test_acc.item()})
                return test_acc.item()

        t = time.time()
        for epoch in range(args.epochs):
            # print(epoch)
            XA_optimizer_2.zero_grad()
            train_base()
            is_break, t = eval(t)
            if is_break:
                break
            # scheduler.step()
        print('s2: Loading {}th epoch'.format(XA_early_stopping_2.best_epoch))
        XA_model_S2.load_state_dict(XA_early_stopping_2.best_state)
        test_acc = test()
        return test_acc


    print(XA_model_S2)

    test_acc = XA_run_s2()

    return test_acc


def main_TS_Decoupled_IMPXA_UPDATE_IterGCN_3SCL_1CE_SUM_Step3V6(trial,new_feature,new_edge_index,new_edge_weight,new_feature2,new_edge_index2,new_edge_weight2):
    data = load_data(args.dataset,
                           missing_link= args.missing_link,
                           missing_feature = args.missing_feature,
                           trial=trial,
                           split_datasets_type=args.split_datasets_type,
                           normalize_features=args.normalize_features)

    data = data.to(device)

    labels = data.y
    idx_train = torch.where(data.train_mask==True)[0]
    idx_val = torch.where(data.val_mask == True)[0]
    idx_test = torch.where(data.test_mask == True)[0]
    num_classes = len(set(data.y.tolist()))



    XA_model_S2 = S2_Decoupled_GCN_3SCL_1CE_SUM_V3(num_nodes=data.num_nodes, in_size=data.x.shape[1],
                    hidden_size=args.hidden,
                    num_layer=args.gcn_num_layer,
                    out_size=num_classes,
                    dropout=args.dropout, device=device,lxw =args.lxw,law = args.law, lzw = args.lzw,GNN_type=args.GNN_type).to(device)

    XA_optimizer_2 = torch.optim.Adam(XA_model_S2.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # total_steps = args.epochs
    # warmup_steps = int(0.1 * total_steps)  # 10% of total steps for warmup
    # scheduler = WarmupCosineSchedule(XA_optimizer_2, warmup_steps=warmup_steps,
    #                                   t_total=total_steps)

    stopping_args3 = Stop_args(patience=args.patience, max_epochs=args.epochs)

    XA_early_stopping_2 = EarlyStopping(XA_model_S2, **stopping_args3)

    supcloss = SupConLoss(temperature=args.tau)
    contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=args.tau),mode='L2L').to(device)

    # criterion = LabelSmoothingCrossEntropy()

    def XA_run_s2():

        def train_base():
            # show_gpu_allocated(3)
            XA_model_S2.train()  # 设置模型为训练模式，这将启用 Dropout
            XA_optimizer_2.zero_grad()
            attr_emb,stru_emb,doub_emb,output = XA_model_S2(new_feature, new_edge_index,new_edge_weight,new_feature2, new_edge_index2,new_edge_weight2)

            ce_loss = F.nll_loss(output[idx_train], labels[idx_train])
            loss1 = supcloss(attr_emb[idx_train], labels[idx_train])
            loss2 = supcloss(stru_emb[idx_train], labels[idx_train])
            loss3 = supcloss(doub_emb[idx_train], labels[idx_train])
            if args.dataset == 'pubmed':
                loss4 = torch.tensor(0.0)
                loss5 = torch.tensor(0.0)
                loss = ce_loss +  args.scl_weight*(loss1 + loss2 +  loss3)

            else:
                loss4 = contrast_model(attr_emb, doub_emb)
                loss5 = contrast_model(stru_emb, doub_emb)
                loss = ce_loss +  args.scl_weight*(loss1 + loss2 +  loss3) + args.scl_weight2*(loss4 + loss5)

            acc_train = accuracy(output[idx_train], labels[idx_train])

            loss.backward()


            XA_optimizer_2.step()


        def eval(t):
            XA_model_S2.eval()  # 设置模型为评估模式，这将禁用 Dropout
            with torch.no_grad():  # 确保在评估过程中不会计算梯度

                attr_emb, stru_emb, doub_emb, output = XA_model_S2(new_feature,new_edge_index, new_edge_weight,new_feature2, new_edge_index2,new_edge_weight2)


                loss_val1 = (F.nll_loss(output[idx_val], labels[idx_val])).item()
                loss_train = (F.nll_loss(output[idx_train], labels[idx_train])).item()
                loss1 = supcloss(attr_emb[idx_val], labels[idx_val])
                loss2 = supcloss(stru_emb[idx_val], labels[idx_val])
                loss3 = supcloss(doub_emb[idx_val], labels[idx_val])
                if args.dataset == 'pubmed':
                    loss4 = torch.tensor(0.0)
                    loss5 = torch.tensor(0.0)
                else:
                    loss4 = contrast_model(attr_emb, doub_emb)
                    loss5 = contrast_model(stru_emb, doub_emb)

                acc_val = accuracy(output[idx_val], labels[idx_val]).item()
                if XA_early_stopping_2.check([acc_val, loss_val1], epoch):
                    return True, t
                acc_test = accuracy(output[idx_test], labels[idx_test]).item()
                acc_train = accuracy(output[idx_train], labels[idx_train]).item()
                if args.run == 'run':
                    wandb.log({f'{trial}-validation/ce_loss': loss_val1,
                         f'{trial}-validation/acc_val': acc_val,
                         f'{trial}-validation/acc_test': acc_test})


                if epoch % 50 == 0:
                    current_time = time.time()
                    tt = current_time - t
                    t = current_time
                    print(
                        's2:epoch:{} , acc_train:{:.4f}, acc_val:{:.4f} , acc_test:{:.4f},train_celoss:{:.4f},val_celoss:{:.4f},'
                        'attr_suploss:{:.4f}, stru_suploss:{:.4f} , doub_suploss:{:.4f},xg_suploss:{:.4f},ag_suploss:{:.4f}, time:{:.4f}s'.format(
                            epoch, acc_train,acc_val, acc_test, loss_train,loss_val1,loss1,loss2,loss3,loss4,loss5,  tt))
                return False, t

        def test():
            XA_model_S2.eval()
            with torch.no_grad():
                attr_emb, stru_emb, doub_emb, output = XA_model_S2(
                    new_feature,
                    new_edge_index, new_edge_weight,new_feature2, new_edge_index2,new_edge_weight2)

                loss_test1 = F.nll_loss(output[idx_test], labels[idx_test])
                # loss1 = supcloss(attr_emb[idx_test], labels[idx_test])
                # loss2 = supcloss(stru_emb[idx_test], labels[idx_test])
                # loss3 = supcloss(doub_emb[idx_test], labels[idx_test])

                test_acc = accuracy(output[idx_test], labels[idx_test])

                print("Test set results:",
                      "loss= {:.4f}".format(loss_test1.item()),
                      "accuracy= {:.4f}".format(test_acc.item()),
                      )

                if args.run == 'run':
                    wandb.log(
                        {
                         "test/ce_loss": loss_test1.item(),
                         'test/acc': test_acc.item()})
                return test_acc.item()

        t = time.time()
        for epoch in range(args.epochs):
            # print(epoch)
            XA_optimizer_2.zero_grad()
            train_base()
            is_break, t = eval(t)
            if is_break:
                break
            # scheduler.step()
        print('s2: Loading {}th epoch'.format(XA_early_stopping_2.best_epoch))
        XA_model_S2.load_state_dict(XA_early_stopping_2.best_state)
        test_acc = test()
        return test_acc


    print(XA_model_S2)

    test_acc = XA_run_s2()

    return test_acc



if __name__ == "__main__":
    if args.run == 'debug':

        pass

    # step1: wandb init
    if args.run == 'run':
        wandb.init(project="IGL_"+args.dataset,name=args.wandb_run_name,config=args)

    accs = []
    seed_num= args.seed

    for trial in range(1, args.num_trials + 1):

        torch.cuda.empty_cache()  # 释放显存

        # test_acc = main_TS_Decoupled_IMPXA_UPDATE_IterGCN_3SCL_1CE_SUM_Step3V3(trial,new_feature2,new_edge_index2,new_edge_weight2,new_feature2,new_edge_index2,new_edge_weight2)
        test_acc = main_TS_Decoupled_IMPXA_UPDATE_IterGCN_3SCL_1CE_SUM_Step3V6(trial)



        print('Trial:{}, Test_acc:{:.4f}'.format(trial, test_acc))
        accs.append(test_acc)

    avg_acc = np.mean(accs) * 100
    std_acc = np.std(accs) * 100
    print('[FINAL RESULT] AVG_ACC:{:.2f}+-{:.2f}'.format(avg_acc, std_acc))

    if args.run == 'run':
        wandb.log({"AVG_ACC": '{:.2f}+-{:.2f}'.format(avg_acc, std_acc)})
        wandb.finish()

