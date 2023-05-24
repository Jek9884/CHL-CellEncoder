import os
import sys

import torch
import torch.optim as optim
from sklearn.model_selection import ShuffleSplit
from torch.utils.data import DataLoader

sys.path.insert(1, 'scBERT')
from scBERT.performer_pytorch import PerformerLM, AutoregressiveWrapper
import scanpy as sc
from utils import *
from custom_dataset import TextSamplerDataset
from torch.cuda.amp import GradScaler
from ray import tune
from ray import air
import csv

CLASS = 7
SEED = 42
BATCH_SIZE = 4
GENE_NUM = 19022
SEQ_LEN = GENE_NUM+1
VALIDATE_EVERY = 1
DIM_LATENT = 200

def load_train_data(device, path = "all_tissue.h5ad"):
    """
        Load all the data from the indicated path, apply the preprocessing pipeline
        and load the data on the device
    """
    data = sc.read_h5ad(path)
    # remove all the columns having a count smaller than 3
    sc.pp.filter_genes(data, min_counts = 3)
    print(np.shape(data.X))
    #Pre-processing step
    sc.pp.normalize_total(data, target_sum=1e4)
    logged = sc.pp.log1p(data, base=2, copy=True)
    data = data.X
    
    # create a TextSamplerDataset with shuffled data
    sss = ShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    
    for index_train, index_val in sss.split(data):
        data_train = data[index_train]
        data_val = data[index_val]
        train_dataset = TextSamplerDataset(data_train, device)
        val_dataset = TextSamplerDataset(data_val, device)
    
    return train_dataset, val_dataset
    
def train(train_loader, val_loader, device, ckpt_path = "../../../CHL-CellEncoder/data/panglao_pretrain.pth", lr = 1e-4, epochs=1, pos_embed_using = False, gradient_accumulate_every = 60, save_ckpt=False):
    """
        Train function to fine-tune the scBERT Performer model
    """
    
    # init model and import pre-training weights
    model = PerformerLM(
        num_tokens = CLASS,
        dim = DIM_LATENT,
        depth = 6,
        max_seq_len = SEQ_LEN,
        heads = 10,
        g2v_position_emb = pos_embed_using
    )
    
    # load pre-trained parameters
    ckpt = torch.load(ckpt_path)
    if ~pos_embed_using:
        ckpt['model_state_dict'].pop('pos_emb.emb.weight')
    model.load_state_dict(ckpt['model_state_dict'])
    for param in model.parameters():
        param.requires_grad = False
    for param in model.norm.parameters():
        param.requires_grad = True
    for param in model.performer.net.layers[-2].parameters():
        param.requires_grad = True

    model = AutoregressiveWrapper(model)

    model = model.to(device)

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = GradScaler()
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=15,
        cycle_mult=2,
        max_lr=lr,
        min_lr=1e-6,
        warmup_steps=5,
        gamma=0.9
    )

    train_loss = []
    val_loss = []
    
    # training phase
    for i in range(1, epochs+1):
    
        model.train()
        print_idx = 0
        
        
        train_loss_ep = 0
        for index, data in enumerate(train_loader):
            index += 1

            if index % gradient_accumulate_every != 0:
                loss, _ = model(data, return_loss = True)
                scaler.scale(loss).backward()

            if index % gradient_accumulate_every == 0:
                print_idx += 1
                loss, _ = model(data, return_loss = True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            train_loss_ep += loss
                
        train_loss_ep = train_loss_ep/len(train_loader)
        train_loss.append(train_loss_ep)
        
        # validation phase
        val_loss_ep = 0
        if i % VALIDATE_EVERY == 0:
            model.eval()
            for index, data in enumerate(val_loader):

                with torch.no_grad():
                    loss, _ = model(data, return_loss = True)
                    val_loss_ep += loss
            
        val_loss_ep = val_loss_ep/len(val_loader)
        
        val_loss.append(val_loss_ep)
    
    # if indicated save the model parameters
    if save_ckpt:
        if not os.path.exists("ckpt_folder"):
            os.makedirs("ckpt_folder")

        torch.save(
            {
                'model_state_dict': model.module.state_dict(),
            },
            f'performer_best.pth'
        )
        
    return train_loss, val_loss


def trainable(config_dict):
    """
        Funtion used by raytune library to perform training for each hyperparameters configuration
    """
    
    # load data 
    data_path = config_dict['data_path']
    train_data, val_data = load_train_data(config_dict['device'], path = data_path)
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)
    
    # start training
    train_loss, val_loss = train(train_loader = train_loader, val_loader = val_loader, 
                                 device = config_dict['device'], lr = config_dict['lr'], epochs = config_dict['epochs'], 
                                 pos_embed_using = config_dict['pos_embed_using'], gradient_accumulate_every = config_dict['gradient_accumulate_every'])
    
    # save results
    for idx, loss in enumerate(train_loss):
        
        train_loss[idx] = train_loss[idx].tolist()
        val_loss[idx] = val_loss[idx].tolist()
    
    config_dict['train_loss'] = train_loss
    config_dict['val_loss'] = val_loss
    config_dict.pop('device')
    config_dict.pop('data_path')
    
    
    with open('../../../CHL-CellEncoder/transformer/grid_scbert_results.csv', 'a', newline='') as csvfile:
        # create a CSV writer object
        writer = csv.DictWriter(csvfile, fieldnames=config_dict.keys(), delimiter='#')
    
        if csvfile.tell() == 0:
            writer.writeheader()

        writer.writerow(config_dict)
            
            

def grid_search(data_path, params, device):
    """
        Perform the grid search on the indicated hyperparameters
    """
    
    params['device'] = device
    params['data_path'] = data_path
    
    # set logs to be shown on the Command Line Interface every 30 seconds
    reporter = tune.CLIReporter(max_report_frequency=30)
    
    # starts grid search using RayTune
    tuner = tune.Tuner(tune.with_resources(trainable,
                                          {"cpu":2, "gpu":1}), 
                       param_space = params, 
                       tune_config = tune.tune_config.TuneConfig(reuse_actors = False),
                       run_config=air.RunConfig(name="scBERT", verbose=1, progress_reporter=reporter))
    results = tuner.fit()
    
    