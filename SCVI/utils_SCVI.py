from scipy.io import mmread
import anndata as ad
import scanpy as sc
from sklearn.model_selection import ShuffleSplit
import numpy as np
from scvi.model import SCVI
import torch
import pandas as pd
import scvi
SEED = 42

def get_data(data_path, multi_batch=False, split=True):
    '''
    Function to load training and validation data
    for the SCVI model
    '''
    
    # Read anndata from file
    data = ad.read(data_path)
    
    # Create the "counts" column
    data.layers["counts"] = data.X.copy()
    
    # Normalize counts per cell
    sc.pp.normalize_total(data, target_sum=1e4)
    
    # Log-transform the data
    sc.pp.log1p(data)

    #Find variable genes based on log dispersion
    sc.pp.highly_variable_genes(data,  min_disp=0.5, min_mean=0.1, max_mean=np.inf)
    
    # Create "batch" column
    data.obs['batch'] = 0
    
    # If there are multiple batches...
    if multi_batch:
        # Create them based on channels (i.e. experiments)
        channels = data.obs['channel'].unique()
        for i, channel in enumerate(channels):
            # To each row of a certain experiments corresponds a specific batch
            data.obs.loc[data.obs['channel'] == channel, 'batch'] = i
            
            
    if split:
        # Split the data between train and validation, shuffling the data based on the seed
        splits = ShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
        for index_train, index_val in splits.split(data):
                data_train = data[index_train]
                data_val = data[index_val]

        return data_train, data_val
    else:
        return data


def retrain(config_dict):
    '''
    Function to retrain the SCVI model
    on a configuration of hyper-parameters
    '''
    
    get_data(config_dict['data_path'], multi_batch= config_dict['multi_batch'], split=False)
    
    # Set up data
    SCVI.setup_anndata(data, layer="counts", batch_key="batch")
        
    model = SCVI(data, n_hidden=config_dict['n_hidden'], n_latent=config_dict['n_latent'], 
               n_layers=config_dict['n_layers'], dropout_rate=config_dict['dropout_rate'], 
               gene_likelihood=config_dict['gene_likelihood'], latent_distribution=config_dict['latent_distribution'])
    
    # Train model
    model.train(max_epochs=config_dict['max_epochs'], use_gpu=config_dict['use_gpu'], 
              train_size=1, batch_size=config_dict['batch_size'], early_stopping=False)
  
    # Save training loss
    print(model.history['reconstruction_loss_train'])
    config_dict['train_loss'] = model.history['reconstruction_loss_train'].iloc[-1].reconstruction_loss_train
    
    # Save model
    if config_dict['model_path']:
        model.save(config_dict['model_path'])
    
        

def load_data():
    '''
    Create anndata with filtered droplet data
    composed of both counts and metadata
    '''
    
    # Read filtered droplet data which is annotated
    df_filter_ann = pd.read_csv("./droplet_filter_annotated.csv")
    
    # Read all tissues droplet data (no filtering)
    all_df = pd.read_csv("./all_tissue_droplet.csv")
    
    # Drop useless columns and rename unnamed columns
    df_filter_ann.drop(columns=['orig.ident', 'percent.ercc'], inplace=True)
    df_filter_ann.rename(columns={'Unnamed: 0': 'cell_id'}, inplace=True)

    all_df.drop(columns=['orig.ident', 'percent.ercc'], inplace=True)
    all_df.rename(columns={'Unnamed: 0': 'cell_id'}, inplace=True)
    
    # Load raw counts of droplet data of all tissues (cells x genes matrix)
    raw = mmread("raw_data_all_tissue_droplet.txt")
    df_raw = pd.DataFrame.sparse.from_spmatrix(raw.T)
    
    # Merge raw counts with metadata on all tissues
    merge_df = all_df.join(df_raw)
    '''
    Filter cells, keep only ones with number of genes per cell greater than 500
    and number of UMIs per cell greather than 1000
    '''
    df_merge_filter = merge_df[(merge_df['nFeature_RNA'] > 500) & (merge_df['nCount_RNA'] > 1000)]
    
    # Merge filtered data with filtered annotated data to add more metadata
    df_merge_ann = df_filter_ann.merge(df_merge_filter, left_on='cell_id', right_on='cell_id', how='right')
    
    # Drop repeated columns and rename the kept columns
    df_merge_ann.drop(columns=['nCount_RNA_y', 'nFeature_RNA_y', 'channel_y', 'mouse.id_y', 'tissue_y', 'subtissue_y', 'mouse.sex_y', 'percent.ribo_y', 'percent.Rn45s_y'], inplace=True)
    
    df_merge_ann.rename(columns={'nCount_RNA_x':'nCount_RNA', 'nFeature_RNA_x':'nFeature_RNA', 'channel_x':'channel',
                   'mouse.id_x':'mouse.id', 'tissue_x':'tissue', 'subtissue_x':'subtissue', 'mouse.sex_x':'mouse.sex',
               'percent.ribo_x':'percent.ribo', 'percent.Rn45s_x':'percent.Rn45s'}, inplace=True)
    
    # Load genes names
    genes = pd.read_csv("genes.tsv", sep="\t", header=None)
    
    # Extract metadata and gene columns
    metadata_cols = ['cell_id', 'nCount_RNA', 'nFeature_RNA', 'channel', 'mouse.id', 'tissue', 'subtissue', 'mouse.sex',
                 'percent.ribo', 'percent.Rn45s', 'cell', 'cell_ontology_class', 'cell_ontology_term_iri', 'cell_ontology_id']
    
    gene_cols = [g for g in df_merge_ann.columns if g not in metadata_cols]
    
    # Rename gene indices with respective gene names
    df_merge_ann.rename(columns=dict(zip(gene_cols, genes[0])), inplace=True)
    
    gene_cols = [g for g in df_merge_ann.columns if g not in metadata_cols]
    
    # Create anndata with counts and metadata
    all_ann = ad.AnnData(X = df_merge_ann[gene_cols], obs = df_merge_ann[metadata_cols])
    
    return all_ann


def filter_anndata(column, val):
    '''
    Filter anndata on a certain column
    equal on a certain value
    '''
    
    all_ann = ad.read('./all_tissue.h5ad')
    exp = all_ann[ all_ann.obs[column] == val ]
    
    return exp


def gen_embeddings(model_path, data_path, embeddings_path):
    '''
    Generate and store embeddings
    '''
    
    # Load data
    data = get_data(data_path, multi_batch = True, split = False)            
    
    # Set up data
    SCVI.setup_anndata(data, layer="counts", batch_key="batch")
    
    model = scvi.model.SCVI.load(model_path, data)
    
    # Generate embeddings
    emb = model.get_latent_representation(data)
    
    # Store embeddings
    torch.save(emb, embeddings_path)