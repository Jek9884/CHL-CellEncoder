from os import path

import pandas as pd
from ray import air
from ray import tune
from scvi.model import SCVI

from utils_SCVI import get_data


def grid_search(params, device):
    '''
    Function to start grid search
    on GPU
    '''
        
    params['device'] = device
    
    reporter = tune.CLIReporter(max_report_frequency=60)
    
    # Starts grid search using RayTune
    tuner = tune.Tuner(tune.with_resources(trainable,
                                          {"cpu":2, "gpu":1}), 
                       param_space = params, 
                       tune_config = tune.tune_config.TuneConfig(reuse_actors = False),
                       run_config=air.RunConfig(name=params['gene_likelihood'], verbose=1, progress_reporter=reporter))
    
    results = tuner.fit()
    

def trainable(config_dict):
    '''
    Function to load data
    and train SCVI model
    reporting train and validation loss
    '''
    
    # Load data
    train_data, val_data = get_data(config_dict['data_path'], config_dict['multi_batch'])
    
    train_data = train_data.copy()
    
    # Set up data
    SCVI.setup_anndata(train_data, layer="counts", batch_key="batch")
        
    model = SCVI(train_data, n_hidden=config_dict['n_hidden'], n_latent=config_dict['n_latent'], 
               n_layers=config_dict['n_layers'], dropout_rate=config_dict['dropout_rate'], 
               gene_likelihood=config_dict['gene_likelihood'], latent_distribution=config_dict['latent_distribution'])
    
    # Train model
    model.train(max_epochs=config_dict['max_epochs'], use_gpu=config_dict['use_gpu'], 
              train_size=config_dict['train_size'], batch_size=config_dict['batch_size'], early_stopping=config_dict['early_stopping'])
    
    # Save train and validation loss
    config_dict['train_loss'] = model.history['reconstruction_loss_train'].iloc[-1].reconstruction_loss_train
    
    config_dict['val_loss'] = model.get_reconstruction_error(val_data)['reconstruction_loss']
    
    config_dict.pop('device')
    file_path = config_dict['file_path']
    config_dict.pop('file_path')
    config_dict.pop('data_path')
    config_dict.pop('multi_batch')
    
    print(model.history['reconstruction_loss_train'])

    # Save results of config on dataframe
    for key, value in config_dict.items():
        config_dict[key] = [config_dict[key]]
    
    df = pd.DataFrame(config_dict)    
    
    # Store results (if file already exists, append the results otherwise create the .csv file)
    df.to_csv(file_path, mode='a', sep='#', index=False, 
            header=False if path.exists(file_path) else True)
    
