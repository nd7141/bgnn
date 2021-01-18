import time
import numpy as np
import torch
from torch.nn import Dropout, ELU
import torch.nn.functional as F
from torch import nn
from dgl.nn.pytorch import GATConv as GATConvDGL, GraphConv, ChebConv as ChebConvDGL, \
    AGNNConv as AGNNConvDGL, APPNPConv
from torch.nn import Sequential, Linear, ReLU, Identity
from tqdm import tqdm
from .Base import BaseModel
from torch.autograd import Variable
from collections import defaultdict as ddict
from .MLP import MLPRegressor


class ElementWiseLinear(nn.Module):
    def __init__(self, size, weight=True, bias=True, inplace=False):
        super().__init__()
        if weight:
            self.weight = nn.Parameter(torch.Tensor(size))
        else:
            self.weight = None
        if bias:
            self.bias = nn.Parameter(torch.Tensor(size))
        else:
            self.bias = None
        self.inplace = inplace

        self.reset_parameters()

    def reset_parameters(self):
        if self.weight is not None:
            nn.init.ones_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.inplace:
            if self.weight is not None:
                x.mul_(self.weight)
            if self.bias is not None:
                x.add_(self.bias)
        else:
            if self.weight is not None:
                x = x * self.weight
            if self.bias is not None:
                x = x + self.bias
        return x

class GATDGL(torch.nn.Module):
    '''
    Implementation of leaderboard GAT network for OGB datasets.
    https://github.com/Espylapiza/dgl/blob/master/examples/pytorch/ogb/ogbn-arxiv/models.py
    '''
    def __init__(
        self,
        in_feats,
        n_classes,
        n_layers=3,
        n_heads=3,
        activation=F.relu,
        n_hidden=250,
        dropout=0.75,
        input_drop=0.1,
        attn_drop=0.0,
    ):
        super().__init__()
        self.in_feats = in_feats
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.num_heads = n_heads

        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        for i in range(n_layers):
            in_hidden = n_heads * n_hidden if i > 0 else in_feats
            out_hidden = n_hidden if i < n_layers - 1 else n_classes
            num_heads = n_heads if i < n_layers - 1 else 1
            out_channels = n_heads

            self.convs.append(
                GATConvDGL(
                    in_hidden,
                    out_hidden,
                    num_heads=num_heads,
                    attn_drop=attn_drop,
                    residual=True,
                )
            )

            if i < n_layers - 1:
                self.norms.append(torch.nn.BatchNorm1d(out_channels * out_hidden))

        self.bias_last = ElementWiseLinear(n_classes, weight=False, bias=True, inplace=True)

        self.input_drop = nn.Dropout(input_drop)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, graph, feat):
        h = feat
        h = self.input_drop(h)

        for i in range(self.n_layers):
            conv = self.convs[i](graph, h)

            h = conv

            if i < self.n_layers - 1:
                h = h.flatten(1)
                h = self.norms[i](h)
                h = self.activation(h, inplace=True)
                h = self.dropout(h)

        h = h.mean(1)
        h = self.bias_last(h)

        return h



class GNNModelDGL(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim,
                 dropout=0., name='gat', residual=True, use_mlp=False, join_with_mlp=False):
        super(GNNModelDGL, self).__init__()
        self.name = name
        self.use_mlp = use_mlp
        self.join_with_mlp = join_with_mlp
        self.normalize_input_columns = True
        if use_mlp:
            self.mlp = MLPRegressor(in_dim, hidden_dim, out_dim)
            if join_with_mlp:
                in_dim += out_dim
            else:
                in_dim = out_dim
        if name == 'gat':
            self.l1 = GATConvDGL(in_dim, hidden_dim//8, 8, feat_drop=dropout, attn_drop=dropout, residual=False,
                              activation=F.elu)
            self.l2 = GATConvDGL(hidden_dim, out_dim, 1, feat_drop=dropout, attn_drop=dropout, residual=residual, activation=None)
        elif name == 'gcn':
            self.l1 = GraphConv(in_dim, hidden_dim, activation=F.elu)
            self.l2 = GraphConv(hidden_dim, out_dim, activation=F.elu)
            self.drop = Dropout(p=dropout)
        elif name == 'cheb':
            self.l1 = ChebConvDGL(in_dim, hidden_dim, k = 3)
            self.l2 = ChebConvDGL(hidden_dim, out_dim, k = 3)
            self.drop = Dropout(p=dropout)
        elif name == 'agnn':
            self.lin1 = Sequential(Dropout(p=dropout), Linear(in_dim, hidden_dim), ELU())
            self.l1 = AGNNConvDGL(learn_beta=False)
            self.l2 = AGNNConvDGL(learn_beta=True)
            self.lin2 = Sequential(Dropout(p=dropout), Linear(hidden_dim, out_dim), ELU())
        elif name == 'appnp':
            self.lin1 = Sequential(Dropout(p=dropout), Linear(in_dim, hidden_dim),
                       ReLU(), Dropout(p=dropout), Linear(hidden_dim, out_dim))
            self.l1 = APPNPConv(k=10, alpha=0.1, edge_drop=0.)


    def forward(self, graph, features):
        h = features
        if self.use_mlp:
            if self.join_with_mlp:
                h = torch.cat((h, self.mlp(features)), 1)
            else:
                h = self.mlp(features)
        if self.name == 'gat':
            h = self.l1(graph, h).flatten(1)
            logits = self.l2(graph, h).mean(1)
        elif self.name in ['appnp']:
            h = self.lin1(h)
            logits = self.l1(graph, h)
        elif self.name == 'agnn':
            h = self.lin1(h)
            h = self.l1(graph, h)
            h = self.l2(graph, h)
            logits = self.lin2(h)
        elif self.name in ['gcn', 'cheb']:
            h = self.drop(h)
            h = self.l1(graph, h)
            logits = self.l2(graph, h)


        return logits

class GNN(BaseModel):
    def __init__(self, task='regression', lr=0.01, hidden_dim=64, dropout=0.,
                 name='gat', residual=True, lang='dgl',
                gbdt_predictions=None, mlp=False, use_leaderboard=False, only_gbdt=False):
        super(GNN, self).__init__()

        self.dropout = dropout
        self.learning_rate = lr
        self.hidden_dim = hidden_dim
        self.task = task
        self.model_name = name
        self.use_residual = residual
        self.lang = lang
        self.use_mlp = mlp
        self.use_leaderboard = use_leaderboard
        self.gbdt_predictions = gbdt_predictions
        self.only_gbdt = only_gbdt

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def __name__(self):
        if self.gbdt_predictions is None:
            return 'GNN'
        else:
            return 'ResGNN'

    def init_model(self):
        if self.lang == 'pyg':
            self.model = GNNModelPYG(in_dim=self.in_dim, hidden_dim=self.hidden_dim, out_dim=self.out_dim,
                                  heads=self.heads, dropout=self.dropout, name=self.model_name,
                                  residual=self.use_residual).to(self.device)
        elif self.lang == 'dgl':
            if self.use_leaderboard:
                self.model = GATDGL(in_feats=self.in_dim, n_classes=self.out_dim).to(self.device)
            else:
                self.model = GNNModelDGL(in_dim=self.in_dim, hidden_dim=self.hidden_dim, out_dim=self.out_dim,
                                         dropout=self.dropout, name=self.model_name,
                                         residual=self.use_residual, use_mlp=self.use_mlp,
                                         join_with_mlp=self.use_mlp).to(self.device)

    def init_node_features(self, X, optimize_node_features):
        node_features = Variable(X, requires_grad=optimize_node_features)
        return node_features

    def fit(self, networkx_graph, X, y, train_mask, val_mask, test_mask, num_epochs,
            cat_features=None, patience=200, logging_epochs=1, optimize_node_features=False,
            loss_fn=None, metric_name='loss', normalize_features=True, replace_na=True):

        # initialize for early stopping and metrics
        if metric_name in ['r2', 'accuracy']:
            best_metric = [np.float('-inf')] * 3  # for train/val/test
        else:
            best_metric = [np.float('inf')] * 3  # for train/val/test
        best_val_epoch = 0
        epochs_since_last_best_metric = 0
        metrics = ddict(list) # metric_name -> (train/val/test)
        if cat_features is None:
            cat_features = []

        if self.gbdt_predictions is not None:
            X = X.copy()
            X['predict'] = self.gbdt_predictions
            if self.only_gbdt:
                cat_features = []
                X = X[['predict']]

        self.in_dim = X.shape[1]
        self.hidden_dim = self.hidden_dim
        if self.task == 'regression':
            self.out_dim = y.shape[1]
        elif self.task == 'classification':
            self.out_dim = len(set(y.iloc[:, 0]))

        if len(cat_features):
            X = self.encode_cat_features(X, y, cat_features, train_mask, val_mask, test_mask)
        if normalize_features:
            X = self.normalize_features(X, train_mask, val_mask, test_mask)
        if replace_na:
            X = self.replace_na(X, train_mask)

        X, y = self.pandas_to_torch(X, y)
        if len(X.shape) == 1:
            X = X.unsqueeze(1)

        if self.lang == 'dgl':
            graph = self.networkx_to_torch(networkx_graph)
        elif self.lang == 'pyg':
            graph = self.networkx_to_torch2(networkx_graph)
        self.init_model()
        node_features = self.init_node_features(X, optimize_node_features)

        self.node_features = node_features
        self.graph = graph
        optimizer = self.init_optimizer(node_features, optimize_node_features, self.learning_rate)

        pbar = tqdm(range(num_epochs))
        for epoch in pbar:
            start2epoch = time.time()

            model_in = (graph, node_features)
            loss = self.train_and_evaluate(model_in, y, train_mask, val_mask, test_mask, optimizer,
                                           metrics, gnn_passes_per_epoch=1)
            self.log_epoch(pbar, metrics, epoch, loss, time.time() - start2epoch, logging_epochs,
                           metric_name=metric_name)

            # check early stopping
            best_metric, best_val_epoch, epochs_since_last_best_metric = \
                self.update_early_stopping(metrics, epoch, best_metric, best_val_epoch, epochs_since_last_best_metric,
                                           metric_name, lower_better=(metric_name not in ['r2', 'accuracy']))
            if patience and epochs_since_last_best_metric > patience:
                break

        if loss_fn:
            self.save_metrics(metrics, loss_fn)

        print('Best {} at iteration {}: {:.3f}/{:.3f}/{:.3f}'.format(metric_name, best_val_epoch, *best_metric))
        return metrics

    def predict(self, graph, node_features, target_labels, test_mask):
        return self.evaluate_model((graph, node_features), target_labels, test_mask)