import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_trials', type=int, default=10)
    parser.add_argument('--missing_link', type=float, default=0.3)
    parser.add_argument('--missing_feature', type=float, default=0.3)
    # parser.add_argument('--train_per_class', type=int, default=20)
    # parser.add_argument('--val_per_class', type=int, default=30)
    # parser.add_argument('--ogb_train_ratio', type=float, default=1.0)
    parser.add_argument('--split_datasets_type', type=str,  default='t2-gnn', choices=['geom-gcn','t2-gnn','public'])

    parser.add_argument('--dataset', type=str, default='cora',choices=['cora','citeseer','pubmed','wisconsin','texas','cornell','chameleon','squirrel'])
    # parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr1', type=float, default=0.0001)
    parser.add_argument('--lr2', type=float, default=0.001)
    parser.add_argument('--lxw', type=float, default=1.0)
    parser.add_argument('--law', type=float, default=1.0)
    parser.add_argument('--lzw', type=float, default=1.0)
    parser.add_argument('--lx_weight', type=float, default=0.0)
    parser.add_argument('--la_weight', type=float, default=0.0)
    parser.add_argument('--lz_weight', type=float, default=0.0)
    parser.add_argument('--scl_weight', type=float, default=1.0)

    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=0.005)
    parser.add_argument('--patience', type=int, default=2000)

    parser.add_argument('--gcn_num_layer', type=int, default=2)
    parser.add_argument('--hidden', type=int, default=64)

    parser.add_argument('--normalize_features', type=bool, default=True)

    parser.add_argument('--tau', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--num_neighbor', type=int, default=5)
    parser.add_argument('--X_num_neighbor', type=int, default=6)
    parser.add_argument('--A_num_neighbor', type=int, default=20)
    parser.add_argument('--knn_metric', type=str, default='cosine', choices=['cosine','minkowski'])
    parser.add_argument('--thresholds', type=float, default=0.5)
    parser.add_argument('--X_thresholds', type=float, default=0.5)
    parser.add_argument('--A_thresholds', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--X_num_neighbor_2', type=int, default=0)
    parser.add_argument('--A_num_neighbor_2', type=int, default=0)
    parser.add_argument('--X_thresholds_2', type=float, default=0.5)
    parser.add_argument('--A_thresholds_2', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=0.001)
    parser.add_argument('--beta1', type=float, default=0.001)
    parser.add_argument('--beta2', type=float, default=0.1)
    parser.add_argument('--topk_weight', type=float, default=0.3)
    parser.add_argument('--scl_weight2', type=float, default=0.0)
    parser.add_argument('--GNN_type', type=str, default='GCN')

    # parser.add_argument('--wandb_project_name', type=str, default='cora',choices=['cora','citeseer','pubmed','wisconsin','texas','cornell','chameleon','squirrel'])
    parser.add_argument('--wandb_run_name', type=str, default='Iter_Graph_Test')
    parser.add_argument('--run', type=str, default='debug')

    # parser.add_argument('--use_bn', action='store_true', default=False)

    # parser.add_argument('--T', type=int, default=20)
    # parser.add_argument('--alpha', type=float, default=0.01)
    # new


    # parser.add_argument('--lambda_pa', type=float, default=4)
    # parser.add_argument('--lambda_ce_aug', type=float, default=0.2)

    # parser.add_argument('--batch_size', type=int, default=0)

    return parser.parse_args()