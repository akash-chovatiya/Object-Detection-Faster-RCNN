# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 18:44:34 2023

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

import torch, tqdm

#%% validater

def validater(val_itr, val_loss_list, val_loss_hist, valid_data_loader, model, DEVICE):
    print('Validating')
    
    # initialize tqdm progress bar
    prog_bar = tqdm.tqdm(valid_data_loader, total=len(valid_data_loader))
    
    for i, data in enumerate(prog_bar):
        images, targets = data
        
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        
        with torch.no_grad():
            loss_dict = model(images, targets)
            
        del images, targets
        
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        val_loss_list.append(loss_value)

        val_loss_hist.send(loss_value)

        val_itr += 1

        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
        torch.cuda.empty_cache()
        
    return val_itr, val_loss_list, val_loss_hist
#%% Standalone Run

if __name__ == "__main__":
    pass