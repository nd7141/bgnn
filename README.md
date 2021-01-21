### Boosted Graph Neural Networks
The code and data for the ICLR 2021 paper: [Boost then Convolve: Gradient Boosting Meets Graph Neural Networks](https://openreview.net/pdf?id=ebS5NUfoMKL)

This code contains implementation of the following models for graphs: 
* **CatBoost**
* **LightGBM**
* **Fully-Connected Neural Network** (FCNN)
* **GNN** (GAT, GCN, AGNN, APPNP)
* **FCNN-GNN** (GAT, GCN, AGNN, APPNP)
* **ResGNN** (CatBoost + {GAT, GCN, AGNN, APPNP})
* **BGNN** (end-to-end {CatBoost + {GAT, GCN, AGNN, APPNP}})

## Installation
To run the models you have to download the repo, install the requirements, and extract the datasets.

First, let's create a python environment:
```bash
mkdir envs
cd envs
python -m venv bgnn_env
source bgnn_env/bin/activate
cd ..
```
---
Second, let's download the code and install requirements
```bash
git clone https://github.com/nd7141/bgnn.git 
cd bgnn
unzip datasets.zip
make install
```
---
Next we need to install a proper version of [PyTorch](https://pytorch.org/) and [DGL](https://www.dgl.ai/), depending on the cuda version of your machine.
We strongly encourage to use GPU-supported versions of DGL (the speed up in training can be 100x).

First, determine your cuda version with `nvcc --version`. 
Then, check installation instructions for [pytorch](https://pytorch.org/get-started/locally/).
For example for cuda version 9.2, install it as follows:
```bash
pip install torch==1.7.1+cu92 torchvision==0.8.2+cu92 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```

If you don't have GPU, use the following: 
```bash
pip install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```
---
Similarly, you need to install [DGL library](https://docs.dgl.ai/en/0.4.x/install/). 
For example, cuda==9.2:

```bash
pip install dgl-cu92
```

For cpu version of DGL: 
```bash
pip install dgl
```

Tested versions of `torch` and `dgl` are:
* torch==1.7.1+cu92
* dgl_cu92==0.5.3

## Running
Starting point is file `scripts/run.py`:
```bash
python scripts/run.py dataset models 
    (optional) 
            --save_folder: str = None
            --task: str = 'regression',
            --repeat_exp: int = 1,
            --max_seeds: int = 5,
            --dataset_dir: str = None,
            --config_dir: str = None
```
Available options for dataset: 
* house (regression)
* county (regression)
* vk (regression)
* wiki (regression)
* avazu (regression)
* vk_class (classification)
* house_class (classification)
* dblp (classification)
* slap (classification)
* path/to/your/dataset
    
Available options for models are `catboost`, `lightgbm`, `gnn`, `resgnn`, `bgnn`, `all`.

Each model is specifed by its config. Check [`configs/`](https://github.com/nd7141/bgnn/tree/master/configs/model) folder to specify parameters of the model and run.

Upon completion, the results wil be saved in the specifed folder (default: `results/{dataset}/day_month/`).
This folder will contain `aggregated_results.json`, which will contain aggregated results for each model.
Each model will have 4 numbers in this order: `mean metric` (RMSE or accuracy), `std metric`, `mean runtime`, `std runtime`.
File `seed_results.json` will have results for each experiment and each seed. 
Additional folders will contain loss values during training. 

---

###Examples

The following script will launch all models on `House` dataset.  
```bash
python scripts/run.py house all
```

The following script will launch CatBoost and GNN models on `SLAP` classification dataset.  
```bash
python scripts/run.py slap catboost gnn --task classification
```

The following script will launch LightGBM model for 5 splits of data, repeating each experiment for 3 times.  
```bash
python scripts/run.py vk lightgbm --repeat_exp 3 --max_seeds 5
```

The following script will launch resgnn and bgnn models saving results to custom folder.  
```bash
python scripts/run.py county resgnn bgnn --save_folder ./county_resgnn_bgnn
```

### Running on your dataset
To run the code on your dataset, it's necessary to prepare the files in the right format. 

You can check examples in `datasets/` folder. 

There should be at least `X.csv` (node features), `y.csv` (target labels), `graph.graphml` (graph in graphml format).

Make sure to keep _these_ filenames for your dataset.

You can also have `cat_features.txt` specifying names of categorical columns.

You can also have `masks.json` specifying train/val/test splits. 

After that run the script as usual: 
```bash
python scripts/run.py path/to/your/dataset gnn catboost 
```

## Citation
```
@inproceedings{
ivanov2021boost,
title={Boost then Convolve: Gradient Boosting Meets Graph Neural Networks},
author={Sergei Ivanov and Liudmila Prokhorenkova},
booktitle={International Conference on Learning Representations (ICLR)},
year={2021},
url={https://openreview.net/forum?id=ebS5NUfoMKL}
}
```
