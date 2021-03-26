import numpy as np
import re
import glob

PATH = '/drive/diode/'
TRAIN = PATH + 'train'
TEST = PATH + 'val'

def load_dataset():
    """
    returns train and test images with their corresponding normals as labels
    """
    
    data = {}
    label = {}
    
    train_images = []
    val_images = []
    train_normals = []
    val_normals = []

    for filename in glob.iglob(TRAIN + '/**/*normal.npy', recursive=True):
        path = filename[:-11] + ".png"
        train_images.append(path)
        train_normals.append(filename)

    for filename in glob.iglob(TEST + '/**/*normal.npy', recursive=True):
        path = filename[:-11] + ".png"
        val_images.append(path)
        val_normals.append(filename)
    
    index = np.random.permutation(len(train_images))

    data['train'] = np.array(train_images)[index]
    label['train'] = np.array(train_normals)[index]
    
    data['val'] = val_images
    label['val'] = val_normals
    
    return data, label