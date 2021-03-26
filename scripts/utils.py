# Libraries
import torch
import numpy as np
import re
import glob
import os
import torchvision
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn
from scipy import io


# constant variables
IMG_SIZE = (288, 384)

# checkpoint path
DEFAULT_CHECKPOINT_DIR = '/home4/shubham/MTML_Pth/checkpoints/' 
VIS_RESULTS_PATH = '/home4/shubham/MTML_Pth/results/'
# paths to datasets
# diode
PATH_DIODE = '/drive/diode/'
TRAIN_DIODE = PATH_DIODE + 'train'
TEST_DIODE = PATH_DIODE + 'val'

# NYUv2
TRAIN_NYU_RGB_PATH = "/home4/shubham/MTML_Pth/datasets/nyuv2/train_rgb/"
TRAIN_NYU_SEG_PATH = "/home4/shubham/MTML_Pth/datasets/nyuv2/train_seg13/"
TEST_NYU_RGB_PATH = "/home4/shubham/MTML_Pth/datasets/nyuv2/test_rgb/"
TEST_NYU_SEG_PATH = "/home4/shubham/MTML_Pth/datasets/nyuv2/test_seg13/"
TRAIN_NYU_SN_PATH = "/home4/shubham/MTML_Pth/datasets/nyuv2/train_sn/"
TEST_NYU_SN_PATH = "/home4/shubham/MTML_Pth/datasets/nyuv2/test_sn/"

# Scannet VP
PATH_SCANNET_VP_TRAIN = '/drive/scannet-vp/'

# SUNRGBD
SUNRGBD_REL_PATH = '/home4/shubham/MTML_Pth/datasets/SUNRGBD/'
PATH_SUNRGBD_TRAIN = '/home4/shubham/MTML_Pth/datasets/sunrgbd-meta-data/SUNRGBD-train_images/'
PATH_SUNRGBD_TRAIN_LABEL = '/home4/shubham/MTML_Pth/datasets/sunrgbd-meta-data/train13labels/'
PATH_SUNRGBD_TEST = '/home4/shubham/MTML_Pth/datasets/sunrgbd-meta-data/SUNRGBD-test_images/'
PATH_SUNRGBD_TEST_LABEL = '/home4/shubham/MTML_Pth/datasets/sunrgbd-meta-data/test13labels/'


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    
    """
    def __init__(self, patience=5, verbose=False, delta=0, path=DEFAULT_CHECKPOINT_DIR+'early_stopping_model.pth'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 10
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'early_stopping_vgg16model.pth'   
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
    
    def __call__(self, val_loss, model):
        
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            
        elif score < self.best_score + self.delta:
            self.counter += 1
            
            if self.counter >= self.patience:
                self.early_stop = True
                
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0   
    
    def save_checkpoint(self, val_loss, model):
        """
        saves the current best version of the model if there is decrease in validation loss
        """
        torch.save(model.state_dict(), self.path)
        self.vall_loss_min = val_loss

        
# returns Pixel mIoU and Pixel level accuracy

class ConfMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, pred, target):
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=pred.device)
        with torch.no_grad():
            k = (target >= 0) & (target < n)
            inds = n * target[k].to(torch.int64) + pred[k]
            self.mat += torch.bincount(inds, minlength=n ** 2).reshape(n, n)

    def get_metrics(self):
        h = self.mat.float()
        acc = torch.diag(h).sum() / h.sum()
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return torch.mean(iu), acc


# transformations  

def train_transformation(task, image, label):
    """
    applies transformations on input images and their labels
    """
    p = np.random.uniform(0, 1)
    
    if task == "surface_normal":
        
        image = torchvision.transforms.ToTensor()(image)
        label = torchvision.transforms.ToTensor()(label)        
        label = label.type(torch.float32)
        
    elif task == "vanishing_point":
        
        image = torchvision.transforms.ToTensor()(image)      
        label = torch.Tensor(label)
        label = label.type(torch.float32)
      
    elif task == "segmentation":
        
        if p <= 0.5 :
            img_out = image.transpose(Image.FLIP_LEFT_RIGHT)
            image = torchvision.transforms.ToTensor()(img_out)
            label = np.array(label.transpose(Image.FLIP_LEFT_RIGHT))
            label = torch.from_numpy(label).type(torch.LongTensor)
            label = torch.squeeze(label,1)

        else:
            image = torchvision.transforms.ToTensor()(image)
            label = torch.from_numpy(np.array(label)).type(torch.LongTensor)
            label = torch.squeeze(label,1)
            
    return image, label


def test_transformation(task, image, label):
    """
    applies transformations on input images and their labels
    """
    if task == "surface_normal":
        
        image = torchvision.transforms.ToTensor()(image)
        label = torchvision.transforms.ToTensor()(label)        

    elif task == "vanishing_point":
        
        image = torchvision.transforms.ToTensor()(image)  
        label = np.array(label)
    
    elif task == "segmentation":
        
        image = torchvision.transforms.ToTensor()(image)
        label = torch.from_numpy(np.array(label)).type(torch.LongTensor)
        label = torch.squeeze(label,1)
        
    return image, label


def get_data_loader(task, data, label, flag, batch_size):
    """
    returns train/test/val dataloaders
    params: flag - train/test/val
            task - segmentation, surface_normal, vanishing_point
            batch_size - batch_size of the dataset
            data, label - dataset with its ground truth label
    """
    
    if flag == "train":
        dataset = DatasetLoader(task, data[flag], label[flag], transform=train_transformation) 
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle=True, num_workers=4)
 
    else:
        dataset = DatasetLoader(task, data[flag], label[flag], transform=test_transformation)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle=True, num_workers=4)

    return dataloader

# DataLoader class
class DatasetLoader(Dataset):
    
    def __init__(self, task, data, label, transform = None):
        self.data = data
        self.label = label
        self.length = len(data)
        self.transform = transform
        self.task = task

    def __len__(self):
        return self.length
  
    def __getitem__(self, idx):
        
        image = Image.open(self.data[idx])
        image = image.resize(IMG_SIZE, Image.BILINEAR)
        
        if self.task == "segmentation":
            
            label = Image.open(self.label[idx])
            label = label.resize(IMG_SIZE, Image.NEAREST)
        
        elif self.task == "surface_normal":
            
            label = Image.fromarray((np.load(self.label[idx])*255.).astype(np.uint8))
            label = label.resize(IMG_SIZE, Image.BILINEAR)
        
        elif self.task == "vanishing_point":
            label = []
            vps = np.load(self.label[idx])
            label.append(vps['x'][0])
            label.append(vps['y'][0])
            label.append(vps['z'][0])
            
        if self.transform:
            image, label = self.transform(self.task, image, label)
        
            
        return image, label
    

def draw_training_curves(train_losses, test_losses, curve_name):
    """
    plots loss/accuracy curves
    :params: 
             train_losses, test_losses - loss values
             curve_name - loss/accuracy curves
    """
    plt.clf()
    plt.plot(train_losses, label='Training {}'.format(curve_name))
    plt.plot(test_losses, label='Testing {}'.format(curve_name))
    plt.legend(frameon=False)
    plt.savefig(VIS_RESULTS_PATH+"{}_curves.png".format(curve_name))
    

# Load datasets

# 1. Diode Dataset for Surface Normal

def load_diode_sn_dataset():
    """
    returns train and test images with their corresponding normals as labels
    """
    
    data = {}
    label = {}
    
    train_images = []
    val_images = []
    train_normals = []
    val_normals = []

    for filename in glob.iglob(TRAIN_DIODE + '/**/*normal.npy', recursive=True):
        path = filename[:-11] + ".png"
        train_images.append(path)
        train_normals.append(filename)

    for filename in glob.iglob(TEST_DIODE + '/**/*normal.npy', recursive=True):
        path = filename[:-11] + ".png"
        val_images.append(path)
        val_normals.append(filename)
    
    index = np.random.permutation(len(train_images))

    data['train'] = np.array(train_images)[index]
    label['train'] = np.array(train_normals)[index]
    
    data['val'] = val_images
    label['val'] = val_normals
    
    return data, label



# 2. Load nyuv2 dataset for segmentation and surface normal

def load_nyuv2_dataset(flag):
    """
    returns dictionary of images and their corresponding annotations split into train, val and test
    :params: flag - (task) segmentation, depth or surface normal
    """ 
    data = {}
    label = {}
    
    TRAIN_PATH_IMG = None
    TRAIN_PATH_LAB = None
    TEST_PATH_IMG = None
    TEST_PATH_LAB = None
    
    if flag == "segmentation":
        TRAIN_PATH_IMG =  TRAIN_NYU_RGB_PATH
        TRAIN_PATH_LAB = TRAIN_NYU_SEG_PATH
        TEST_PATH_IMG = TEST_NYU_RGB_PATH
        TEST_PATH_LAB = TEST_NYU_SEG_PATH
        
    elif flag == "normal":
        TRAIN_PATH_IMG =  TRAIN_NYU_RGB_PATH
        TRAIN_PATH_LAB = TRAIN_NYU_SN_PATH
        TEST_PATH_IMG = TEST_NYU_RGB_PATH
        TEST_PATH_LAB = TEST_NYU_SN_PATH
        
    train_images = glob.glob(TRAIN_PATH_IMG + "*.png")
    train_labels = glob.glob(TRAIN_PATH_LAB + "*.png")
    
    index = np.random.permutation(len(train_images))
    images = np.array(train_images)[index]
    labels = np.array(train_labels)[index]
    
    length = int(len(images)*0.85)
   
    data["train"], data["val"] = images[:length], images[length:]
    label["train"], label["val"] = labels[:length], labels[length:]
    data["test"] = glob.glob(TEST_PATH_IMG + "*.png")
    label["test"] = glob.glob(TEST_PATH_LAB + "*.png")
    
    return data, label


# 3. Load scannet vanishing points dataset

def load_scannet_vp_dataset():
    """
    returns training and validation images and their corresponding vanishing points labels.
    """
    data = {}
    label = {}
    
    lst = os.listdir(PATH_SCANNET_VP_TRAIN)

    final_images = []
    final_vpoints = []

    for folder in lst:
        images = glob.glob(PATH_SCANNET_VP_TRAIN+folder+"/*color.png")
        vpoints = glob.glob(PATH_SCANNET_VP_TRAIN+folder+"/*vanish.npz")
        index = np.random.permutation(len(images))
        images = np.array(images)[index]
        vpoints = np.array(vpoints)[index]
        final_images.extend(images[:40])
        final_vpoints.extend(vpoints[:40])
    
    split = int(len(final_images)*0.80)
    
    data['train'], data['val'] = final_images[:split], final_images[split:]
    label['train'], label['val'] = final_vpoints[:split], final_vpoints[split:]
    
    return data, label


# 4. Load SUNRGBD dataset

def load_sunrgbd_dataset_original():
    """
    returns a dictionary of train, test and validation images and their corresponding segmentation maps
    """
    data = {}
    label = {}
    
    images = []
    seg = []

    for filename in glob.iglob(SUNRGBD_REL_PATH + '**/image/*.jpg', recursive=True):
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

def load_sunrgbd_dataset():
    """
    load dataset from pre-split data 5k train and 5k test
    """
    data, label = {}, {}
    
    train = glob.glob(PATH_SUNRGBD_TRAIN+'*.jpg')
    lab = glob.glob(PATH_SUNRGBD_TRAIN_LABEL+'*.png')
    data["train"] = sorted(train, key=lambda x: re.findall(r"\d+",x)[1])
    label["train"] = sorted(lab, key=lambda x: re.findall(r"\d+",x)[-1])
    
    test = glob.glob(PATH_SUNRGBD_TEST+'*.jpg')
    lab = glob.glob(PATH_SUNRGBD_TEST_LABEL+'*.png')
    test = sorted(test, key=lambda x: re.findall(r"\d+",x)[1])
    lab = sorted(lab, key=lambda x: re.findall(r"\d+",x)[-1])
    
    data["val"] = test[:2000]
    label["val"] = lab[:2000]
    data["test"] = test[2000:]
    label["test"] = lab[2000:]
    
    return data, label

def load_sunrgbd_mappings():
    """
    returns 40 -> 13 class mappings
    """
    mapping = { 
  
            1	:	12,
            2	:	5,
            3	:	6,
            4	:	1,
            5	:	4,
            6	:	9,
            7	:	10,
            8	:	12,
            9	:	13,
            10	:	6,
            11	:	8,
            12	:	6,
            13	:	13,
            14	:	10,
            15	:	6,
            16	:	13,
            17	:	6,
            18	:	7,
            19	:	7,
            20	:	5,
            21	:	7,
            22	:	3,
            23	:	2,
            24	:	6,
            25	:	11,
            26	:	7,
            27	:	7,
            28	:	7,
            29	:	7,
            30	:	7,
            31	:	7,
            32	:	6,
            33	:	7,
            34	:	7,
            35	:	7,
            36	:	7,
            37	:	7,
            38	:	7,
            39	:	6,
            40	:	7,
       }
    
    return mappings
