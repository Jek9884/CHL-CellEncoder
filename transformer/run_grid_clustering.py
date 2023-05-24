import numpy as np
import anndata as ad
import os
import torch
from os import path
import anndata
import sys
sys.path.insert(1, '../')
from utils import *
import warnings
warnings.filterwarnings('ignore')

# Load encodings and data
all_sub_anndata = ad.read('./full_emb_anndata.h5ad')

# Hyper-parameters to test
metrics = ['euclidean', 'cosine', 'l1', 'l2']
resolutions = [1, 2, 3, 4, 5, 10, 20] 
init_pos = ['random', 'spectral']
use_rep = "X_enc"
n_neigh = [5, 15, 30, 50]
filename = "./clustering_full.csv"

# Start grid
grid_clustering(all_sub_anndata, n_pcs=all_sub_anndata.obsm['X_enc'].shape[1], metrics = metrics, resolutions = resolutions, init_pos = init_pos, 
                use_rep=use_rep, n_neigh = n_neigh, filename=filename)
