from utils_SCVI import retrain, gen_embeddings
import os

# select GPU
os.environ["CUDA_VISIBLE_DEVICES"]="3"

# hyper-parameters to re-train on
params = {
    "n_hidden": 128, # Number of nodes per hidden layer
    "n_latent": 32, # Dimensionality of the latent space
    "n_layers": 1, # Number of hidden layers used for encoder and decoder NNs
    'dropout_rate': 0.1,
    'gene_likelihood': 'nb',
    'latent_distribution': 'normal',
    'max_epochs': 5,
    'use_gpu': True,
    'batch_size': 4,
    'data_path': './all_tissue.h5ad',
    'model_path': './scvi_model',
    'multi_batch': True
}

retrain(params)

# Generate and store embeddings
gen_embeddings('./scvi_model', './all_tissue.h5ad', './embeddings/SCVI_emb.pt')