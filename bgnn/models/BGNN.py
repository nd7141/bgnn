import itertools
import time
import numpy as np
import torch

from catboost import Pool, CatBoostClassifier, CatBoostRegressor, sum_models
from .GNN import GNNModelDGL, GATDGL
from .Base import BaseModel
from tqdm import tqdm
from collections import defaultdict as ddict

class BGNN(BaseModel):
    def __init__(self,
                 task='regression', iter_per_epoch = 10, lr=0.01, hidden_dim=64, dropout=0.,
                 only_gbdt=False, train_non_gbdt=False,
                 name='gat', use_leaderboard=False, depth=6, gbdt_lr=0.1):
        super(BaseModel, self).__init__()
        self.learning_rate = lr
        self.hidden_dim = hidden_dim
        self.task = task
        self.dropout = dropout
        self.only_gbdt = only_gbdt
        self.train_residual = train_non_gbdt
        self.name = name
        self.use_leaderboard = use_leaderboard
        self.iter_per_epoch = iter_per_epoch
        self.depth = depth
        self.lang = 'dgl'
        self.gbdt_lr = gbdt_lr

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def __name__(self):
        return 'BGNN'

    def init_gbdt_model(self, num_epochs, epoch):
        if self.task == 'regression':
            catboost_model_obj = CatBoostRegressor
            catboost_loss_fn = 'RMSE' #''RMSEWithUncertainty'
        else:
            if epoch == 0:
                catboost_model_obj = CatBoostClassifier
                catboost_loss_fn = 'MultiClass'
            else:
                catboost_model_obj = CatBoostRegressor
                catboost_loss_fn = 'MultiRMSE'

        return catboost_model_obj(iterations=num_epochs,
                                  depth=self.depth,
                                  learning_rate=self.gbdt_lr,
                                  loss_function=catboost_loss_fn,
                                  random_seed=0,
                                  nan_mode='Min')

    def fit_gbdt(self, pool, trees_per_epoch, epoch):
        gbdt_model = self.init_gbdt_model(trees_per_epoch, epoch)
        gbdt_model.fit(pool, verbose=False)
        return gbdt_model

    def init_gnn_model(self):
        if self.use_leaderboard:
            self.model = GATDGL(in_feats=self.in_dim, n_classes=self.out_dim).to(self.device)
        else:
            self.model = GNNModelDGL(in_dim=self.in_dim,
                                     hidden_dim=self.hidden_dim,
                                     out_dim=self.out_dim,
                                     name=self.name,
                                     dropout=self.dropout).to(self.device)

    def append_gbdt_model(self, new_gbdt_model, weights):
        if self.gbdt_model is None:
            return new_gbdt_model
        return sum_models([self.gbdt_model, new_gbdt_model], weights=weights)

    def train_gbdt(self, gbdt_X_train, gbdt_y_train, cat_features, epoch,
                   gbdt_trees_per_epoch, gbdt_alpha):

        pool = Pool(gbdt_X_train, gbdt_y_train, cat_features=cat_features)
        epoch_gbdt_model = self.fit_gbdt(pool, gbdt_trees_per_epoch, epoch)
        if epoch == 0 and self.task=='classification':
            self.base_gbdt = epoch_gbdt_model
        else:
            self.gbdt_model = self.append_gbdt_model(epoch_gbdt_model, weights=[1, gbdt_alpha])

    def update_node_features(self, node_features, X, encoded_X):
        if self.task == 'regression':
            predictions = np.expand_dims(self.gbdt_model.predict(X), axis=1)
            # predictions = self.gbdt_model.virtual_ensembles_predict(X,
            #                                                         virtual_ensembles_count=5,
            #                                                         prediction_type='TotalUncertainty')
        else:
            predictions = self.base_gbdt.predict_proba(X)
            # predictions = self.base_gbdt.predict(X, prediction_type='RawFormulaVal')
            if self.gbdt_model is not None:
                predictions_after_one = self.gbdt_model.predict(X)
                predictions += predictions_after_one

        if not self.only_gbdt:
            if self.train_residual:
                predictions = np.append(node_features.detach().cpu().data[:, :-self.out_dim], predictions,
                                        axis=1)  # append updated X to prediction
            else:
                predictions = np.append(encoded_X, predictions, axis=1)  # append X to prediction

        predictions = torch.from_numpy(predictions).to(self.device)

        node_features.data = predictions.float().data

    def update_gbdt_targets(self, node_features, node_features_before, train_mask):
        return (node_features - node_features_before).detach().cpu().numpy()[train_mask, -self.out_dim:]

    def init_node_features(self, X):
        node_features = torch.empty(X.shape[0], self.in_dim, requires_grad=True, device=self.device)
        if not self.only_gbdt:
            node_features.data[:, :-self.out_dim] = torch.from_numpy(X.to_numpy(copy=True))
        return node_features

    def init_node_parameters(self, num_nodes):
        return torch.empty(num_nodes, self.out_dim, requires_grad=True, device=self.device)

    def init_optimizer2(self, node_parameters, learning_rate):
        params = [self.model.parameters(), [node_parameters]]
        return torch.optim.Adam(itertools.chain(*params), lr=learning_rate)

    def update_node_features2(self, node_parameters, X):
        if self.task == 'regression':
            predictions = np.expand_dims(self.gbdt_model.predict(X), axis=1)
        else:
            predictions = self.base_gbdt.predict_proba(X)
            if self.gbdt_model is not None:
                predictions += self.gbdt_model.predict(X)

        predictions = torch.from_numpy(predictions).to(self.device)
        node_parameters.data = predictions.float().data

    def fit(self, networkx_graph, X, y, train_mask, val_mask, test_mask, cat_features,
            num_epochs, patience, logging_epochs=1, loss_fn=None, metric_name='loss',
            normalize_features=True, replace_na=True,
            ):

        # initialize for early stopping and metrics
        if metric_name in ['r2', 'accuracy']:
            best_metric = [np.float('-inf')] * 3  # for train/val/test
        else:
            best_metric = [np.float('inf')] * 3  # for train/val/test
        best_val_epoch = 0
        epochs_since_last_best_metric = 0
        metrics = ddict(list)
        if cat_features is None:
            cat_features = []

        if self.task == 'regression':
            self.out_dim = y.shape[1]
        elif self.task == 'classification':
            self.out_dim = len(set(y.iloc[test_mask, 0]))
        # self.in_dim = X.shape[1] if not self.only_gbdt else 0
        # self.in_dim += 3 if uncertainty else 1
        self.in_dim = self.out_dim + X.shape[1] if not self.only_gbdt else self.out_dim

        self.init_gnn_model()

        gbdt_X_train = X.iloc[train_mask]
        gbdt_y_train = y.iloc[train_mask]
        gbdt_alpha = 1
        self.gbdt_model = None

        encoded_X = X.copy()
        if not self.only_gbdt:
            if len(cat_features):
                encoded_X = self.encode_cat_features(encoded_X, y, cat_features, train_mask, val_mask, test_mask)
            if normalize_features:
                encoded_X = self.normalize_features(encoded_X, train_mask, val_mask, test_mask)
            if replace_na:
                encoded_X = self.replace_na(encoded_X, train_mask)

        node_features = self.init_node_features(encoded_X)
        optimizer = self.init_optimizer(node_features, optimize_node_features=True, learning_rate=self.learning_rate)

        y, = self.pandas_to_torch(y)
        self.y = y
        if self.lang == 'dgl':
            graph = self.networkx_to_torch(networkx_graph)
        elif self.lang == 'pyg':
            graph = self.networkx_to_torch2(networkx_graph)

        self.graph = graph

        pbar = tqdm(range(num_epochs))
        for epoch in pbar:
            start2epoch = time.time()

            # gbdt part
            self.train_gbdt(gbdt_X_train, gbdt_y_train, cat_features, epoch,
                            self.iter_per_epoch, gbdt_alpha)

            self.update_node_features(node_features, X, encoded_X)
            node_features_before = node_features.clone()
            model_in=(graph, node_features)
            loss = self.train_and_evaluate(model_in, y, train_mask, val_mask, test_mask,
                                           optimizer, metrics, self.iter_per_epoch)
            gbdt_y_train = self.update_gbdt_targets(node_features, node_features_before, train_mask)

            self.log_epoch(pbar, metrics, epoch, loss, time.time() - start2epoch, logging_epochs,
                           metric_name=metric_name)
            # check early stopping
            best_metric, best_val_epoch, epochs_since_last_best_metric = \
                self.update_early_stopping(metrics, epoch, best_metric, best_val_epoch, epochs_since_last_best_metric,
                                           metric_name, lower_better=(metric_name not in ['r2', 'accuracy']))
            if patience and epochs_since_last_best_metric > patience:
                break
            if np.isclose(gbdt_y_train.sum(), 0.):
                print('Nodes do not change anymore. Stopping...')
                break

        if loss_fn:
            self.save_metrics(metrics, loss_fn)

        print('Best {} at iteration {}: {:.3f}/{:.3f}/{:.3f}'.format(metric_name, best_val_epoch, *best_metric))
        return metrics

    def predict(self, graph, X, y, test_mask):
        node_features = torch.empty(X.shape[0], self.in_dim).to(self.device)
        self.update_node_features(node_features, X, X)
        return self.evaluate_model((graph, node_features), y, test_mask)