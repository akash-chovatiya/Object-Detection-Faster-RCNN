# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 14:55:35 2023

@author: Chovatiya
"""

import os
import sys
import inspect
import yaml

config_file = open('config.yaml','r')
config = yaml.load(config_file, Loader = yaml.FullLoader)

currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))

maindir = os.path.dirname(currentdir)
sys.path.insert(0, maindir)

subdir = os.path.dirname(maindir)
sys.path.insert(0, subdir)

subsubdir = os.path.dirname(subdir)
sys.path.insert(0, subsubdir)

#%% import libraries

import pickle
from lib.datasets import LoadDatasets
from lib.dataloader import ModelDataLoader
import torch
from datetime import date
from lib.utilities import collate_fn
from optunatrainer import optuna_optimizer

#%% Standalone Run
if __name__ == "__main__":
    read_pickle = open(os.path.join(maindir, config['path']['labels_path']), 'rb')
    all_classes = pickle.load(read_pickle)
    time_stamp = date.today().strftime('%d-%m-20%y')
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    DATA_DIR = os.path.join(maindir, config['path']['dataset_path'])
    OUT_DIR = os.path.join(maindir, config['path']['result_path'], time_stamp)
    # os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.exists(OUT_DIR):
            os.makedirs(OUT_DIR)
    batch_size = config['dl']['batch_size']
    shuffle = config['dl']['shuffle']
    num_workers = config['dl']['num_workers']
    VISUALIZE_TRANSFORMED_IMAGES = config['dl']['visualize_img']
    MAP_PER_CLASS = config['dl']['MAP_PER_CLASS']
            
    train_data = LoadDatasets(DATA_DIR, all_classes, 'training')
    val_data = LoadDatasets(DATA_DIR, all_classes, 'validation')
    test_data = LoadDatasets(DATA_DIR, all_classes, 'testing')
    
    training_dataloader = ModelDataLoader(train_data, batch_size=batch_size, 
                                          shuffle = shuffle,
                                          num_workers = num_workers,
                                          collate_fn = collate_fn)
    validation_dataloader = ModelDataLoader(val_data, batch_size=batch_size, 
                                          shuffle = shuffle,
                                          num_workers = num_workers,
                                          collate_fn = collate_fn)
    testing_dataloader = ModelDataLoader(test_data, batch_size=batch_size, 
                                          shuffle = shuffle,
                                          num_workers = num_workers,
                                          collate_fn = collate_fn)
    
    n_trials = config['optuna']['trials']
    if not config['optuna']['saved_study']:
        save_study = None
    else:
        save_study = os.path.join(maindir, config['path']['result_path'],
                                  config['optuna']['time-stamp'],
                                  'optuna_study.pkl')
    opt_optim = optuna_optimizer(OUT_DIR, all_classes, training_dataloader,
                                 testing_dataloader, validation_dataloader, 
                                 train_data, test_data, val_data, n_trials,
                                 save_study)
    opt_optim.optuna_study()
    opt_optim.create_summary()