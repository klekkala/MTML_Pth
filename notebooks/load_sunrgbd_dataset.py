import numpy as np
import glob
import os
import re



REL_PATH = '/home4/shubham/MTML_Pth/datasets/SUNRGBD/'

PATH_TRAIN = '/home4/shubham/MTML_Pth/datasets/sunrgbd-meta-data/SUNRGBD-train_images/'
PATH_TRAIN_LABEL = '/home4/shubham/MTML_Pth/datasets/sunrgbd-meta-data/train13labels/'

PATH_TEST = '/home4/shubham/MTML_Pth/datasets/sunrgbd-meta-data/SUNRGBD-test_images/'
PATH_TEST_LABEL = '/home4/shubham/MTML_Pth/datasets/sunrgbd-meta-data/test13labels/'

def load_dataset_original():
    """
    returns a dictionary of train, test and validation images and their corresponding segmentation maps
    """
    data = {}
    label = {}
    
    images = []
    seg = []

    for filename in glob.iglob(REL_PATH + '**/image/*.jpg', recursive=True):
        images.append(filename)
        path = '/'.join(filename.split('/')[:-2])+"/seg.mat"
        seg.append(path)
        
    index = np.random.permutation(len(images))

    images = np.array(images)[index]
    labels = np.array(seg)[index]
    
    train = int(len(images)*0.8)
    val = int(len(images)*0.9)
    
    data["train"], data["val"], data["test"] = images[:train], images[train:val], images[val:]
    label["train"], label["val"], label["test"] = labels[:train], labels[train:val], images[val:]
    
    return data, label

def load_dataset():
    """
    load dataset from pre-split data 5k train and 5k test
    """
    data, label = {}, {}
    
    train = glob.glob(PATH_TRAIN+'*.jpg')
    lab = glob.glob(PATH_TRAIN_LABEL+'*.png')
    data["train"] = sorted(train, key=lambda x: re.findall(r"\d+",x)[1])
    label["train"] = sorted(lab, key=lambda x: re.findall(r"\d+",x)[-1])
    
    test = glob.glob(PATH_TEST+'*.jpg')
    lab = glob.glob(PATH_TEST_LABEL+'*.png')
    test = sorted(test, key=lambda x: re.findall(r"\d+",x)[1])
    lab = sorted(lab, key=lambda x: re.findall(r"\d+",x)[-1])
    
    data["val"] = test[:2000]
    label["val"] = lab[:2000]
    data["test"] = test[2000:]
    label["test"] = lab[2000:]
    
    return data, label
    