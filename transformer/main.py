import os

import ray
import torch
from ray import tune

from finetuning import grid_search

# select the GPU to use
os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = torch.device(0)

# init raytune
ray.shutdown()
ray.init(num_gpus=1) 

# grid-search parameters
params = {
    "lr": tune.grid_search([1e-4, 1e-2]), 
    "epochs": tune.grid_search([1, 2, 3]),
    "pos_embed_using": False,
    "gradient_accumulate_every": tune.grid_search([10, 60, 150])
}

data_path = "../../../CHL-CellEncoder/dataset/all_tissue.h5ad"

# perform grid-search
grid_search(data_path, params, device)

