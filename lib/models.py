# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 16:22:18 2022

@author: Tahasanul
"""

#%% Initialization of Libraries and Directory
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

import torchvision

def get_model(num_classes, pretrained_model=False, model_name="Resnet50"):
    
    model = None
    if model_name.lower() == "resnet50":
        # load model from torchvision
        if pretrained_model is True:
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
                )
        else:
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
        
        # get the number of input features 
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        
        # define a new head for the detector with required number of classes
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes) 

    return model

#%% Standalone Run

if __name__ == "__main__":
    pass