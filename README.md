### Boosted Graph Neural Networks

Paper: [Boost then Convolve: Gradient Boosting Meets Graph Neural Networks](https://openreview.net/forum?id=ebS5NUfoMKL)

## Installing
To install the repo you have to download the repo, install the requirements, extract the datasets, and run the script. 

## Running
There are two required arguments, dataset and the models you want to run. 

```
python run.py house all
```
will run all models (CatBoost, LightGBM, FCNN, GNN, ResGNN, BGNN) on house dataset. 

You can run just a few of these models: 
```
python run.py house catboost gnn bgnn
```
will run CatBoost, GNN, BGNN models on house dataset. 

## Configs
You can specify the parameters of each model in the corresponding config. Hyperparameters are defined under `hp` key. You need to provide the list of values for each hyperparameter. You can take a look at [the example configs](https://github.com/nd7141/bgnn/tree/master/configs/model).
