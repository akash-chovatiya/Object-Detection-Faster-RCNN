# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 18:44:22 2023

@author: Chovatiya
"""

#%% Initialization of Libraries and Directory

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

import tqdm
import torch

#%% trainer

def trainer(train_itr, train_loss_list, train_loss_hist, optimizer, train_data_loader, model, DEVICE):
    print('Training')
    
    # initialize tqdm progress bar
    prog_bar = tqdm.tqdm(train_data_loader, total=len(train_data_loader))
    
    for i, data in enumerate(prog_bar):
        optimizer.zero_grad()
        images, targets = data
        
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        
        del images, targets

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        train_loss_list.append(loss_value)

        train_loss_hist.send(loss_value)

        losses.backward()
        optimizer.step()

        train_itr += 1
    
        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
        
        torch.cuda.empty_cache()
    return train_itr, train_loss_list, train_loss_hist, optimizer

#%% Standalone Run

if __name__ == "__main__":
    pass