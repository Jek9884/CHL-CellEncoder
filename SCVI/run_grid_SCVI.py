import os
import torch
import ray
from ray import tune
from grid_SCVI import grid_search

os.environ["CUDA_VISIBLE_DEVICES"]="3"
device = torch.device(0)

SEED = 42
      
ray.init()

# Hyper-parameters for grid search
params = {
    "n_hidden": tune.grid_search([8, 16, 32, 64, 128]), # Number of nodes per hidden layer
    "n_latent": tune.grid_search([8, 16, 32, 64]), # Dimensionality of the latent space 
    "n_layers": tune.grid_search([1, 2, 3, 4, 5]), # Number of hidden layers used for encoder and decoder NNs 
    'dropout_rate': tune.grid_search([0.1, 0.3, 0.5]),
    'gene_likelihood': 'nb',
    'latent_distribution': tune.grid_search(['normal', 'ln']),
    'max_epochs': tune.grid_search([2, 5, 8, 10]),
    'use_gpu': True,
    'train_size': 1,
    'batch_size': 4,
    'early_stopping': False,
    'data_path': '../../../CHL-CellEncoder/dataset/all_tissue.h5ad',
    'file_path': '../../../CHL-CellEncoder/grid_SCVI_all/grid_SCVI_all.csv',
    'multi_batch': False
}


grid_search(params, device)