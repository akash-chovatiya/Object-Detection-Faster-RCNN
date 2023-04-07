# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 18:20:58 2023

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

from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torch, tqdm
#%% meanaverageprecision

def meanaverageprecision(map_itr, map_list, map_hist, valid_data_loader, model,
                         class_metrics, DEVICE):
    print('Calculating Mean Average Precision')
    # load MeanAveragePrecision class
    metric = MeanAveragePrecision(class_metrics=class_metrics)
    # initialize tqdm progress bar
    prog_bar = tqdm.tqdm(valid_data_loader, total=len(valid_data_loader))
    model.eval()
    
    for i, data in enumerate(prog_bar):
        images, targets = data
        
        images = list(image.to(DEVICE) for image in images)
        targets = list({k: v.to(DEVICE) for k, v in t.items()} for t in targets)
               
        
        with torch.no_grad():
            preds = model(images)        
        
        metric.update(preds, targets)
        
        del images, targets
        
        if i == len(prog_bar) - 1:
            map_results = metric.compute()
            
            map_global = map_results["map"].numpy().tolist()
            
            map_list.append(map_global)
            map_hist.send(map_global)
            
            map_itr += 1
    
            # update the map value beside the progress bar for each iteration
            prog_bar.set_description(desc=f"Mean Average Precision:\
                                     {map_global:.4f}")
            prog_bar.refresh()
            
            torch.cuda.empty_cache()
        
    model.train()
    return map_itr, map_list, map_hist, map_results, map_global

#%% Standalone Run

if __name__ == "__main__":
    pass