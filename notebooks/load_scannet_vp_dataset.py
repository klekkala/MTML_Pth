import glob
import numpy as np
import os

PATH_TRAIN = '/drive/scannet-vp/'

def load_dataset():
    
    data = {}
    label = {}
    
    lst = os.listdir(PATH_TRAIN)

    final_images = []
    final_vpoints = []

    for folder in lst:
        images = glob.glob(PATH_TRAIN+folder+"/*color.png")
        vpoints = glob.glob(PATH_TRAIN+folder+"/*vanish.npz")
        index = np.random.permutation(len(images))
        images = np.array(images)[index]
        vpoints = np.array(vpoints)[index]
        final_images.extend(images[:40])
        final_vpoints.extend(vpoints[:40])
    
    split = int(len(final_images)*0.80)
    
    data['train'], data['val'] = final_images[:split], final_images[split:]
    label['train'], label['val'] = final_vpoints[:split], final_vpoints[split:]
    
    return data, label