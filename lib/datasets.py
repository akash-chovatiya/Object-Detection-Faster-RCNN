# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 09:23:34 2023

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
from torch.utils.data import Dataset
import PIL
import torch, torchvision
import xml.etree.ElementTree as ET

#%%

class LoadDatasets(Dataset):
    def __init__(self, path, classes, mode, resize=False):
        super().__init__()
        self.path = os.path.join(path, mode)
        self.classes = classes
        self.images = [image for image in sorted(
            os.listdir(self.path)) if image[-4:].lower() == '.png']
        self.resize = resize
        if not self.resize:
            self.transforms = torchvision.transforms.Compose(
                [torchvision.transforms.ToTensor()])
        else:
            self.width = 128
            self.height = 128
            self.transforms = torchvision.transforms.Compose(
                [torchvision.transforms.Resize((self.width, self.height)),
                 torchvision.transforms.ToTensor()])

    def __getitem__(self, index):
        image_name = self.images[index]
        image = PIL.Image.open(os.path.join(self.path, image_name))
        
        image = self.__remove_transparency__(image)

        boxes, labels, classes = self.__getxmlinfo__(image_name)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.as_tensor([index], dtype=torch.int64)

        # prepare the final `target` dictionary
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        #target["classes"] = classes
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["image_id"] = image_id

        return self.transforms(image), target

    def __getxmlinfo__(self, image_name):
        annot_file_path = os.path.join(self.path, image_name[:-4] + '.xml')
        labels = []
        boxes = []
        classes = []
        tree = ET.parse(annot_file_path)
        root = tree.getroot()

        for member in root.findall('object'):
            labels.append(self.classes.index(member.find('name').text))
            classes.append(member.find('name').text)

            xmin = int(member.find('bndbox').find('xmin').text)
            xmax = int(member.find('bndbox').find('xmax').text)
            ymin = int(member.find('bndbox').find('ymin').text)
            ymax = int(member.find('bndbox').find('ymax').text)

            if self.resize:
                image = PIL.Image.open(os.path.join(self.path, image_name))
                wt = image[0]
                ht = image[1]
                xmin = (xmin/wt)*self.width
                xmax = (xmax/wt)*self.width
                ymin = (ymin/ht)*self.height
                ymax = (ymax/ht)*self.height

            boxes.append([xmin, ymin, xmax, ymax])
        return boxes, labels, classes
    
    def __remove_transparency__(self, im, bg_colour=(255, 255, 255)):
        '''
        Remove transparency of the alpha value of PIL images.
        Parameters
        ----------
        im : Image
            PIL image which should be converted to RBG mode.
        bg_colour : tuple, optional
            The background color which replace the transparency. The default is (255, 255, 255).
        Returns
        -------
        PIL.Image
            The new image if convertation is possible.
        '''
        if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):
            alpha = im.convert('RGBA').split()[-1]
            bg = PIL.Image.new("RGB", im.size, bg_colour + (255,))
            bg.paste(im, mask=alpha)
            return bg
        else:
            return im

    def __len__(self):
        return len(self.images)

#%%
if __name__ == "__main__":
    pass