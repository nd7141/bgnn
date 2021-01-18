import os

from dgl.data import citation_graph as citegrh, TUDataset
import torch as th
from dgl import DGLGraph
import numpy as np
from sklearn.model_selection import KFold
import itertools
from sklearn.preprocessing import OneHotEncoder as OHE
import random
import json

def load_cora_data():
    data = citegrh.load_cora()
    features = th.FloatTensor(data.features)
    labels = th.LongTensor(data.labels)
    train_mask = th.BoolTensor(data.train_mask)
    test_mask = th.BoolTensor(data.test_mask)
    g = DGLGraph(data.graph)
    return g, features, labels, train_mask, test_mask

def get_degree_features(graph):
    return graph.out_degrees().unsqueeze(-1).numpy()

def get_categorical_features(features):
    return np.argmax(features, axis=-1).unsqueeze(dim=1).numpy()

def get_random_int_features(shape, num_categories=100):
    return np.random.randint(0, num_categories, size=shape)

def get_random_norm_features(shape):
    return np.random.normal(size=shape)

def get_random_uniform_features(shape):
    return np.random.unifor(-1, 1, size=shape)

def merge_features(*args):
    return np.hstack(args)

def get_train_data(graph, features, num_random_features=10, num_random_categories=100):
    return merge_features(
        get_categorical_features(features),
        get_degree_features(graph),
        get_random_int_features(shape=(features.shape[0], num_random_features), num_categories=num_random_categories),
    )


def save_folds(dataset_name, n_splits=3):
    dataset = TUDataset(dataset_name)
    i = 0
    kfold = KFold(n_splits=n_splits, shuffle=True)
    dir_name = f'kfold_{dataset_name}'
    for trix, teix in kfold.split(range(len(dataset))):
        os.makedirs(f'{dir_name}/fold{i}', exist_ok=True)
        np.savetxt(f'{dir_name}/fold{i}/train.idx', trix, fmt='%i')
        np.savetxt(f'{dir_name}/fold{i}/test.idx', teix, fmt='%i')
        i += 1


def graph_to_node_label(graphs, labels):
    targets = np.array(list(itertools.chain(*[[labels[i]] * graphs[i].number_of_nodes() for i in range(len(graphs))])))
    enc = OHE(dtype=np.float32)
    return np.asarray(enc.fit_transform(targets.reshape(-1, 1)).todense())


def get_masks(N, train_size=0.6, val_size=0.2, random_seed=42):
    if not random_seed:
        seed = random.randint(0, 100)
    else:
        seed = random_seed

    # print('seed', seed)
    random.seed(seed)

    indices = list(range(N))
    random.shuffle(indices)

    train_mask = indices[:int(train_size * len(indices))]
    val_mask = indices[int(train_size * len(indices)):int((train_size + val_size) * len(indices))]
    train_val_mask = indices[:int((train_size + val_size) * len(indices))]
    test_mask = indices[int((train_size + val_size) * len(indices)):]

    return train_mask, val_mask, train_val_mask, test_mask


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)