import itertools
import torch
from sklearn import preprocessing
import pandas as pd
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import r2_score, accuracy_score

class BaseModel(torch.nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def pandas_to_torch(self, *args):
        return [torch.from_numpy(arg.to_numpy(copy=True)).float().squeeze().to(self.device) for arg in args]

    def networkx_to_torch(self, networkx_graph):
        import dgl
        # graph = dgl.DGLGraph()
        graph = dgl.from_networkx(networkx_graph)
        graph = dgl.remove_self_loop(graph)
        graph = dgl.add_self_loop(graph)
        graph = graph.to(self.device)
        return graph

    def networkx_to_torch2(self, networkx_graph):
        from torch_geometric.utils import convert
        import torch_geometric.transforms as T
        graph = convert.from_networkx(networkx_graph)
        transform = T.Compose([T.TargetIndegree()])
        graph = transform(graph)
        return graph.to(self.device)

    def move_to_device(self, *args):
        return [arg.to(self.device) for arg in args]

    def init_optimizer(self, node_features, optimize_node_features, learning_rate):

        params = [self.model.parameters()]
        if optimize_node_features:
            params.append([node_features])
        optimizer = torch.optim.Adam(itertools.chain(*params), lr=learning_rate)
        return optimizer

    def log_epoch(self, pbar, metrics, epoch, loss, epoch_time, logging_epochs, metric_name='loss'):
        train_rmse, val_rmse, test_rmse = metrics[metric_name][-1]
        if epoch and epoch % logging_epochs == 0:
            pbar.set_description(
                "Epoch {:05d} | Loss {:.3f} | Loss {:.3f}/{:.3f}/{:.3f} | Time {:.4f}".format(epoch, loss,
                                                                                              train_rmse,
                                                                                              val_rmse, test_rmse,
                                                                                              epoch_time))

    def normalize_features(self, X, train_mask, val_mask, test_mask):
        min_max_scaler = preprocessing.MinMaxScaler()
        A = X.to_numpy(copy=True)
        A[train_mask] = min_max_scaler.fit_transform(A[train_mask])
        A[val_mask + test_mask] = min_max_scaler.transform(A[val_mask + test_mask])
        return pd.DataFrame(A, columns=X.columns).astype(float)

    def replace_na(self, X, train_mask):
        if X.isna().any().any():
            return X.fillna(X.iloc[train_mask].min() - 1)
        return X

    def encode_cat_features(self, X, y, cat_features, train_mask, val_mask, test_mask):
        from category_encoders import CatBoostEncoder
        enc = CatBoostEncoder()
        A = X.to_numpy(copy=True)
        b = y.to_numpy(copy=True)
        A[np.ix_(train_mask, cat_features)] = enc.fit_transform(A[np.ix_(train_mask, cat_features)], b[train_mask])
        A[np.ix_(val_mask + test_mask, cat_features)] = enc.transform(A[np.ix_(val_mask + test_mask, cat_features)])
        A = A.astype(float)
        return pd.DataFrame(A, columns=X.columns)

    def train_model(self, model_in, target_labels, train_mask, optimizer):
        y = target_labels[train_mask]

        self.model.train()
        logits = self.model(*model_in).squeeze()
        pred = logits[train_mask]

        if self.task == 'regression':
            loss = torch.sqrt(F.mse_loss(pred, y))
        elif self.task == 'classification':
            loss = F.cross_entropy(pred, y.long())
        else:
            raise NotImplemented("Unknown task. Supported tasks: classification, regression.")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss

    def evaluate_model(self, logits, target_labels, mask):
        metrics = {}
        y = target_labels[mask]
        with torch.no_grad():
            pred = logits[mask]
            if self.task == 'regression':
                metrics['loss'] = torch.sqrt(F.mse_loss(pred, y).squeeze() + 1e-8)
                metrics['rmsle'] = torch.sqrt(F.mse_loss(torch.log(pred + 1), torch.log(y + 1)).squeeze() + 1e-8)
                metrics['mae'] = F.l1_loss(pred, y)
                metrics['r2'] = torch.Tensor([r2_score(y.cpu().numpy(), pred.cpu().numpy())])
            elif self.task == 'classification':
                metrics['loss'] = F.cross_entropy(pred, y.long())
                metrics['accuracy'] = torch.Tensor([(y == pred.max(1)[1]).sum().item()/y.shape[0]])

            return metrics

    def train_val_test_split(self, X, y, train_mask, val_mask, test_mask):
        X_train, y_train = X.iloc[train_mask], y.iloc[train_mask]
        X_val, y_val = X.iloc[val_mask], y.iloc[val_mask]
        X_test, y_test = X.iloc[test_mask], y.iloc[test_mask]
        return X_train, y_train, X_val, y_val, X_test, y_test

    def train_and_evaluate(self, model_in, target_labels, train_mask, val_mask, test_mask,
                           optimizer, metrics, gnn_passes_per_epoch):
        loss = None

        for _ in range(gnn_passes_per_epoch):
            loss = self.train_model(model_in, target_labels, train_mask, optimizer)

        self.model.eval()
        logits = self.model(*model_in).squeeze()
        train_results = self.evaluate_model(logits, target_labels, train_mask)
        val_results = self.evaluate_model(logits, target_labels, val_mask)
        test_results = self.evaluate_model(logits, target_labels, test_mask)
        for metric_name in train_results:
            metrics[metric_name].append((train_results[metric_name].detach().item(),
                               val_results[metric_name].detach().item(),
                               test_results[metric_name].detach().item()
                               ))
        return loss

    def update_early_stopping(self, metrics, epoch, best_metric, best_val_epoch, epochs_since_last_best_metric, metric_name,
                              lower_better=False):
        train_metric, val_metric, test_metric = metrics[metric_name][-1]
        if (lower_better and val_metric < best_metric[1]) or (not lower_better and val_metric > best_metric[1]):
            best_metric = metrics[metric_name][-1]
            best_val_epoch = epoch
            epochs_since_last_best_metric = 0
        else:
            epochs_since_last_best_metric += 1
        return best_metric, best_val_epoch, epochs_since_last_best_metric

    def save_metrics(self, metrics, fn):
        with open(fn, "w+") as f:
            for key, value in metrics.items():
                print(key, value, file=f)

    def plot(self, metrics, legend, title, output_fn=None, logx=False, logy=False, metric_name='loss'):
        import matplotlib.pyplot as plt
        metric_results = metrics[metric_name]
        xs = [range(len(metric_results))] * len(metric_results[0])
        ys = list(zip(*metric_results))

        plt.rcParams.update({'font.size': 40})
        plt.rcParams["figure.figsize"] = (20, 10)
        lss = ['-', '--', '-.', ':']
        colors = ['#4053d3', '#ddb310', '#b51d14', '#00beff', '#fb49b0', '#00b25d', '#cacaca']
        colors = [(235, 172, 35), (184, 0, 88), (0, 140, 249), (0, 110, 0), (0, 187, 173), (209, 99, 230), (178, 69, 2),
                  (255, 146, 135), (89, 84, 214), (0, 198, 248), (135, 133, 0), (0, 167, 108), (189, 189, 189)]
        colors = [[p / 255 for p in c] for c in colors]
        for i in range(len(ys)):
            plt.plot(xs[i], ys[i], lw=4, color=colors[i])
        plt.legend(legend, loc=1, fontsize=30)
        plt.title(title)

        plt.xscale('log') if logx else None
        plt.yscale('log') if logy else None
        plt.xlabel('Iteration')
        plt.ylabel('RMSE')
        plt.grid()
        plt.tight_layout()

        plt.savefig(output_fn, bbox_inches='tight') if output_fn else None
        plt.show()

    def plot_interactive(self, metrics, legend, title, logx=False, logy=False, metric_name='loss', start_from=0):
        import plotly.graph_objects as go
        metric_results = metrics[metric_name]
        xs = [list(range(len(metric_results)))] * len(metric_results[0])
        ys = list(zip(*metric_results))

        fig = go.Figure()
        for i in range(len(ys)):
            fig.add_trace(go.Scatter(x=xs[i][start_from:], y=ys[i][start_from:],
                                     mode='lines+markers',
                                     name=legend[i]))

        fig.update_layout(
            title=title,
            title_x=0.5,
            xaxis_title='Epoch',
            yaxis_title='RMSE',
            font=dict(
                size=40,
            ),
            height=600,
        )

        if logx:
            fig.update_layout(xaxis_type="log")
        if logy:
            fig.update_layout(yaxis_type="log")

        fig.show()
