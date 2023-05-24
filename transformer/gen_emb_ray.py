import os
import sys
sys.path.insert(1, 'scBERT')
from scBERT.performer_pytorch import PerformerLM, AutoregressiveWrapper
import scanpy as sc
import numpy as np
import warnings
import torch
from torch.utils.data import DataLoader, Dataset
from custom_dataset import TextSamplerDataset
from ray import tune
from ray import air

warnings.filterwarnings('ignore')

CLASS = 7
SEED = 42
GENE_NUM = 19022
SEQ_LEN = GENE_NUM+1

def grid_search(params, device):
    """
        Perform the grid search on the indicated hyperparameters
    """
    
    params['device'] = device
    
    # set logs to be shown on the Command Line Interface every 120 seconds
    reporter = tune.CLIReporter(max_report_frequency=120)
    
    # starts grid search using RayTune
    tuner = tune.Tuner(tune.with_resources(trainable,
                                          {"cpu":2, "gpu":1}), 
                       param_space = params, 
                       tune_config = tune.tune_config.TuneConfig(reuse_actors = False),
                       run_config=air.RunConfig(name="scBERT", verbose=1, progress_reporter=reporter))
    results = tuner.fit()
    
def trainable(config_dict):
    """
        Funtion used by raytune library to perform training for each hyperparameter configuration
    """
    
    data = sc.read_h5ad(config_dict["data_path"])
    # remove all the columns having a count smaller than 3
    sc.pp.filter_genes(data, min_counts = 3)
    
    # pre-processing step
    sc.pp.normalize_total(data, target_sum=1e4)
    logged = sc.pp.log1p(data, base=2, copy=True)

    tiss = config_dict['tiss']
    
    data = data.X
    
    # take the i-th example
    data = data[config_dict['example_index'], :]
    
    dataset = TextSamplerDataset(data, config_dict['device'])

    data_loader = DataLoader(dataset, batch_size=1)
    
    get_embedding(data_loader, config_dict['device'], tiss)
    
    
def get_embedding(data_loader, device, tiss, save=True):
    """
        Generate embeddings using the model selected in the fine-tuning phase
    """
    
    # load model from checkpoint
    model = PerformerLM(
        num_tokens = CLASS,
        dim = 200,
        depth = 6,
        max_seq_len = SEQ_LEN,
        heads = 10,
        g2v_position_emb = False
    )
 
    ckpt = torch.load("../../../CHL-CellEncoder/transformer/performer_best.pth")
    model.load_state_dict(ckpt['model_performer_state_dict'])
    model = AutoregressiveWrapper(model)
    model = model.to(device)
    model.eval()
    
    # perform inference
    for i, exmp in enumerate(data_loader):
        loss, embedding = model(exmp, return_loss = False, return_encodings = True)
    
    # save embedding
    if save:
        y = np.load(f"../../../CHL-CellEncoder/transformer/embedding_files_v2/{tiss}.npy") if os.path.isfile(f"../../../CHL-CellEncoder/transformer/embedding_files_v2/{tiss}.npy") else np.array([]) # get data if exist
        
        print(np.shape(y))
        print(tiss)
        
        if len(y):
            np.save(f"../../../CHL-CellEncoder/transformer/embedding_files_v2/{tiss}.npy", np.vstack((y, embedding.cpu().detach().numpy())))
        else:
            np.save(f"../../../CHL-CellEncoder/transformer/embedding_files_v2/{tiss}.npy", embedding.cpu().detach().numpy())
        
    else:
        return embedding
    
    


