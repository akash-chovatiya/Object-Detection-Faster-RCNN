# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 16:04:00 2023

@author: Chovatiya
"""

import os
import sys
import inspect
import yaml

config_file = open('config.yaml', 'r')
config = yaml.load(config_file, Loader=yaml.FullLoader)

currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))

maindir = os.path.dirname(currentdir)
sys.path.insert(0, maindir)

subdir = os.path.dirname(maindir)
sys.path.insert(0, subdir)

subsubdir = os.path.dirname(subdir)
sys.path.insert(0, subsubdir)

#%%
from torch.utils.data import DataLoader

#%%
class ModelDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=20, shuffle=False, num_workers=5, pin_memory=True, collate_fn=None):
        super().__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)
        
#%% Standalone Run

if __name__ == "__main__":
    pass