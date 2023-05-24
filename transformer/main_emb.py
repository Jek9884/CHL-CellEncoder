import os
import warnings

import ray
import torch
from ray import tune

warnings.filterwarnings('ignore')
from gen_emb_ray import grid_search

# select the GPU to use
os.environ["CUDA_VISIBLE_DEVICES"]="2"
device = torch.device(0)

# init raytune
ray.shutdown()
ray.init(num_gpus=1) 

# number of examples for the tissue
n_examples = 4536 

# grid-search parameters
params = {
    "example_index": tune.grid_search(list(range(0,n_examples))),
    "tiss": "Muscle",
    "data_path": "../../../CHL-CellEncoder/SCVI/muscle.h5ad"
}

# perform grid-search
grid_search(params, device)