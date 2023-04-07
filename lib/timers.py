# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 11:59:49 2022

@author: Tahasanul
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

#%% tic

def tic(timeName="Default_timer"):
    #Homemade version of matlab tic and toc functions
    import time
    globals()[timeName] = time.time()
    print("Timer set for '" + timeName + "'")

#%% toc
    
def toc(timeName="Default_timer"):
    import time
    if timeName in globals():
        time_then = globals()[timeName]
        del globals()[timeName]
        secs = round(time.time() - time_then)
        print(convert_secs_to_DD_HH_MM_SS(secs, timeName))        
        return secs
    else:
        print("Toc: start time for '" + timeName + "' is not set")
        return False
    
#%% convert_secs_to_DD_HH_MM_SS

def convert_secs_to_DD_HH_MM_SS(secs, timeName=""):
    days = secs // 86400
    secs %= 86400
    hours = secs // 3600
    secs %= 3600
    mins = secs // 60
    secs %= 60
    if not len(timeName) == 0:
        result = "Elapsed time for '{}' is".format(timeName)
    else:
        result = "Elapsed time is"
    if days >0:
        result += f" {days} days"
    if hours > 0:
        result += f" {hours} hours"
    if mins > 0:
        result += f" {mins} minutes"
    if secs > 0:
        result += f" {secs} seconds"
        
    return result
#%% Standalone Run

if __name__ == "__main__":
    pass