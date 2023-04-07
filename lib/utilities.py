# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 16:07:49 2023

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
import numpy as np
import pickle, torch, cv2
import matplotlib.pyplot as plt
import torchvision.transforms as T
from PIL import ImageFont, ImageDraw
import xml.etree.ElementTree as ET
from statistics import mean

#%%
#%% Averager

class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0
        
    def send(self, value):
        self.current_total += value
        self.iterations += 1
    
    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations
    
    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0

#%% SaveBestModel

class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(
        self, best_valid_loss=float('inf')
    ):
        self.best_valid_loss = best_valid_loss
        
    def __call__(
        self, OUT_DIR, current_valid_loss, 
        epoch, model, optimizer, DATA_NAME
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            os.makedirs(OUT_DIR, exist_ok=True)
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, os.path.join(OUT_DIR,'best_model_' + str(DATA_NAME) + '.pth'))

#%%
def extract_classes(root_dir):
    targets = []
    annot_files = [label for label in sorted(os.listdir(root_dir)) 
                   if label[-4:] == '.xml']
    
    for annot_file in annot_files:
        annot_file_path = os.path.join(root_dir, annot_file)
        tree = ET.parse(annot_file_path)
        root = tree.getroot()
            
        for member in root.findall('object'):
            targets.append(member.find('name').text)
        
    classes = np.array(targets)
    classes = np.unique(classes)
    classes = classes.tolist()
    return classes

def plot_image(img, target):
    img_clone = T.ToPILImage()(img).convert('RGB')
    img_clone = np.asarray(img_clone)
    bbox = target['boxes'].numpy()
    llabels = target['labels'].numpy()
    
    for i in range(len(bbox)):
        cv2.rectangle(img_clone, (int(bbox[i][0]), int(bbox[i][1])), (int(bbox[i][2]), int(bbox[i][3])), color=(0, 255, 0), thickness=1)
        #cv2.putText(img_clone, str(int(llabels[i])), (int(bbox[i][2]), int(bbox[i][3])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),thickness=3)
    plt.imshow(img_clone)
    plt.show()

#%% save_model
            
def save_model(OUT_DIR, epoch, model, optimizer, DATA_NAME):
    """
    Function to save the trained model till current epoch, or whenver called
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, os.path.join(OUT_DIR,'last_model_' + str(DATA_NAME) +'.pth'))

#%% save_loss_plot

def save_loss_plot(OUT_DIR, train_loss, val_loss, map_value, DATA_NAME):
    window = 5
    def running_avg(loss):
        average_y = []
        for ind in range(len(loss)-window+1):
            average_y.append(mean(loss[ind:ind+window]))
        for ind in range(window - 1):
            average_y.insert(0, np.nan)
        return average_y
    figure_1, train_ax = plt.subplots()
    figure_2, valid_ax = plt.subplots()    
    figure_3, map_ax = plt.subplots()
    train_ax.plot(train_loss, color='tab:blue')
    train_ax.plot(running_avg(train_loss), color='tab:red')
    train_ax.set_xlabel('iterations')
    train_ax.set_ylabel('train_loss')
    valid_ax.plot(val_loss, color='tab:red')
    valid_ax.plot(running_avg(val_loss), color='tab:blue')
    valid_ax.set_xlabel('iterations')
    valid_ax.set_ylabel('validation_loss')
    map_ax.plot(map_value, color='tab:green')
    map_ax.plot(running_avg(map_value), color='tab:red')
    map_ax.set_xlabel('epochs')
    map_ax.set_ylabel('mean_average_precision')
    os.makedirs(OUT_DIR, exist_ok=True)
    figure_1.savefig(os.path.join(OUT_DIR,"train_loss_" + str(DATA_NAME) + ".png"))
    figure_2.savefig(os.path.join(OUT_DIR,"valid_loss_" + str(DATA_NAME) + ".png"))
    figure_3.savefig(os.path.join(OUT_DIR,"mean_average_precision_" + str(DATA_NAME) + ".png"))
    print('SAVING PLOTS COMPLETE...')

    plt.close('all')

#%% collate_fn

def collate_fn(batch):
    """
    To handle the data loading as different images may have different number 
    of objects and to handle varying size tensors as well.
    """
    return tuple(zip(*batch))

#%% storeVariableWP

def storeVariableWP (variable, filename, dir_path, database_type = None):
    
    if database_type is None:
        location = os.path.join( dir_path )
    else:        
        location = os.path.join( dir_path, database_type )
    
    os.makedirs(location, exist_ok=True)
    location = os.path.join( location, filename )
    
    variablePkl = open(location,'wb')
    print (location)
    pickle.dump(variable,variablePkl)
    variablePkl.close()
    
    return

#%% retrieveVariableWP
    
def retrieveVariableWP (filename, dir_path, database_types = None):    
    try:
        if database_types is None:            
            location = os.path.join( dir_path, filename )
        else:
            if isinstance(database_types, list) or isinstance(database_types, tuple):
                location = os.path.join( dir_path )
                for database_type in database_types:
                    location = os.path.join( location, database_type )
            else:
                location = os.path.join( dir_path, database_types )
            location = os.path.join( location, filename )
        variablePkl = open(location, 'rb')    
        variable = pickle.load(variablePkl)
        
    except Exception as e:
        if not len(str(e)) == 0:
            print ("**********************************************************************")
            print ("*** Exception cuaght -> {} ***".format(e))
            print ("**********************************************************************")
        variable = None
    
    return variable


#%%
def save_data(OUT_DIR, train_loss, val_loss, map_value, DATA_NAME):
    def results(loss,dir_name):
        outdir = os.path.join(OUT_DIR, dir_name)
        os.makedirs(outdir, exist_ok=True)
        file = open(os.path.join(outdir, dir_name + "_" + str(DATA_NAME) + ".txt"), 'w')
        for item in loss:
            file.write(str(item)+"\n")
        file.close()
    results(train_loss,"train_loss")
    results(val_loss,"val_loss")
    results(map_value,"map_value")
    
#%% show_tranformed_image

def show_tranformed_image(train_loader, all_classes):
    """
    This function shows the transformed images from the `train_loader`.
    Helps to check whether the tranformed images along with the corresponding
    labels are correct or not.
    Only runs if `VISUALIZE_TRANSFORMED_IMAGES = True` in config.py.
    """
    
    if len(train_loader) > 0:
        for i in range(1):
            images, targets = next(iter(train_loader))
            images = list(image for image in images)
            targets = list({k: v for k, v in t.items()} for t in targets)
            boxes = targets[i]['boxes'].cpu().numpy().astype(np.int32)
            labels = targets[i]['labels'].cpu().numpy().astype(np.int32)
            sample = images[i].permute(1, 2, 0).cpu().numpy()
            for box_num, box in enumerate(boxes):
                cv2.rectangle(sample,
                            (box[0], box[1]),
                            (box[2], box[3]),
                            (0, 0, 255), 2)
                cv2.putText(sample, all_classes[labels[box_num]], 
                            (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 
                            1.0, (0, 0, 255), 2)
            cv2.imshow('Transformed image', sample)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
#%% show_tranformed_image_v2

def show_tranformed_image_v2(train_loader, all_classes):
    """
    This function shows the transformed images from the `train_loader`.
    Helps to check whether the tranformed images along with the corresponding
    labels are correct or not.
    Only runs if `VISUALIZE_TRANSFORMED_IMAGES = True` in config.py.
    """
    
    if len(train_loader) > 0:
        PIL_transform = T.ToPILImage()
        for i in range(5):
            plt.figure("Sample - {}".format(str(i).zfill(2)))
            images, targets = next(iter(train_loader))
            images = list(image for image in images)
            targets = list({k: v for k, v in t.items()} for t in targets)
            boxes = targets[i]['boxes'].cpu().numpy().astype(np.int32)
            labels = targets[i]['labels'].cpu().numpy().astype(np.int32)
            sample = PIL_transform(images[i])            
            draw = ImageDraw.Draw(sample)
            for box_num, box in enumerate(boxes):
                color = tuple(np.random.randint(256, size=3))
                draw.rectangle(
                                    (
                                        (box[0], box[1]), 
                                        (box[2], box[3])
                                    ),
                                    outline=color
                                )
                draw.text(
                                (box[0], box[1]-10),
                                all_classes[labels[box_num]], 
                                font=ImageFont.truetype("arial.ttf", 12),
                                fill=color
                            )

            # sample.show()
            plt.imshow(sample)
            plt.show()