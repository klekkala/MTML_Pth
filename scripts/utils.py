# Libraries
import torch
import numpy as np
import re
import glob
import os
import torchvision
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
import matplotlib.pyplot as plt
import torch.nn as nn
from scipy import io
import time
import random
timestr = time.strftime("%Y%m%d-%H%M%S")
ImageFile.LOAD_TRUNCATED_IMAGES = True

# constant variables
IMG_SIZE = (288, 384)

# checkpoint path
REL = os.getcwd()
DEFAULT_CHECKPOINT_DIR = REL + '/checkpoints/' 
VIS_RESULTS_PATH = REL + '/results/'

# paths to datasets
# diode
SN_TRAIN = REL + '/datasets/diode_dataset/train/'
SN_VAL = REL + '/datasets/diode_dataset/val/'

# NYUv2
TRAIN_NYU_RGB_PATH = REL + "/datasets/nyuv2/train_rgb/"
TRAIN_NYU_SEG_PATH = REL + "/datasets/nyuv2/train_seg13/"
TEST_NYU_RGB_PATH = REL + "/datasets/nyuv2/test_rgb/"
TEST_NYU_SEG_PATH = REL + "/datasets/nyuv2/test_seg13/"
TRAIN_NYU_SN_PATH = REL + "/datasets/nyuv2/train_sn/"
TEST_NYU_SN_PATH = REL + "/datasets/nyuv2/test_sn/"

# Scannet VP
TRAIN_VP = REL + '/datasets/scannet_vp/train/'
VAL_VP = REL + '/datasets/scannet_vp/val/'
TEST_VP = REL + '/datasets/scannet_vp/test/'

#SUNRGBD
SUNRGBD_TRAIN_IMAGES = REL + '/datasets/sunrgbd_dataset/sunrgbd_images_train_final.txt'
SUNRGBD_TEST_IMAGES = REL + '/datasets/sunrgbd_dataset/sunrgbd_images_test_final.txt'

SUNRGBD_TRAIN_DEPTH = REL + '/datasets/sunrgbd_dataset/sunrgbd_depth_train_final.txt'
SUNRGBD_TEST_DEPTH = REL + '/datasets/sunrgbd_dataset/sunrgbd_depth_test_final.txt'

SUNRGBD_TRAIN_SEG = REL + '/datasets/sunrgbd_dataset/sunrgbd_seg_train_final.txt'
SUNRGBD_TEST_SEG = REL + '/datasets/sunrgbd_dataset/sunrgbd_seg_test_final.txt'

#SUNRGBD NYUv2
NYU_TEST_IMAGES = REL + '/datasets/sun_nyu2/nyu_test_images.txt'
NYU_TEST_DEPTH = REL + '/datasets/sun_nyu2/nyu_test_depth.txt'
NYU_TEST_SEG = REL + '/datasets/sun_nyu2/nyu_test_seg.txt'

#SN NYUv2
NYU_RGB = REL + '/datasets/nyuv2/test_rgb/'
NYU_DEPTH = REL + '/datasets/nyuv2/test_depth/'
NYU_SN = REL + '/datasets/nyuv2/test_sn/'

MAPPING = { 
            
            0	:	0,
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
            37	:	7
            #38	:	7,
            #39	:	6,
            #40	:	7,
       }

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    
    """
    def __init__(self, patience=12, verbose=False, delta=0.001, path=DEFAULT_CHECKPOINT_DIR+'/early_stopping_model.pth', task=None, timestamp='current'):
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
        self.task = task
        self.timestr = timestamp
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
        self.path = DEFAULT_CHECKPOINT_DIR + self.task + '/'+ self.timestr +'/early_stopping_model.pth'
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
            k = (target >= 1) & (target <= n)

            inds = n * target[k].to(torch.int64) + pred[k]
            self.mat += torch.bincount(inds, minlength=n ** 2).reshape(n, n)

    def get_metrics(self):
        h = self.mat.float()
        h= h[1:,1:]
        acc = torch.diag(h).sum() / h.sum()
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return torch.mean(iu), acc

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def multitask_transformation(task, image, label, depth):
    """
    applies transformations on input images and their labels
    """
    if task == "surface_normal":
        
        image = torchvision.transforms.ToTensor()(image)
        label = torch.from_numpy(label).type(torch.float32)  
        depth = torch.from_numpy(np.array(depth)).type(torch.float32)
        
    elif task == "vanishing_point":
        
        image = torchvision.transforms.ToTensor()(image)  
        label = torch.Tensor(label)
        label = label.type(torch.float32)
        depth = torch.from_numpy(np.array(depth)/255.).type(torch.float32)
    
    elif task == "segmentation":
        
        image = torchvision.transforms.ToTensor()(image)
        label = torch.from_numpy(np.array(label)).type(torch.LongTensor)
        label = torch.squeeze(label,1)
        depth = torch.from_numpy(np.array(depth)/255.).type(torch.float32)
    
    return image, label, depth

def get_multitask_data_loader(task, data, label, depth,  flag, batch_size):
    """
    returns train/test/val dataloaders
    params: flag - train/test/val
            task - segmentation, surface_normal, vanishing_point
            batch_size - batch_size of the dataset
            data, label - dataset with its ground truth label
    """

    dataset = Multitask_DatasetLoader(task,  data[flag], label[flag], depth[flag], transform=multitask_transformation) 
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle=True, num_workers=4)

    return dataloader

class Multitask_DatasetLoader(Dataset):
    
    def __init__(self, task, data, label, depth, transform = None):
        self.data = data
        self.label = label
        self.depth = depth
        self.length = len(data)
        self.transform = transform
        self.task = task

    def __len__(self):
        return self.length
  
    def __getitem__(self, idx):
        
        image = Image.open(self.data[idx])
        image = image.resize(IMG_SIZE, Image.BILINEAR)
        
        if self.task == "segmentation":
            lab = np.array(Image.open(self.label[idx]))
            palette = list(MAPPING.keys())
            key = np.array(list(MAPPING.values()))
            index = np.digitize(lab.ravel(), palette, right=True)
            new_img = np.array(key[index].reshape(lab.shape)).astype(np.uint8)
            label = Image.fromarray(new_img)
            label = label.resize(IMG_SIZE, Image.NEAREST)
            # depth
            depth = Image.fromarray(np.uint8(np.array(Image.open(self.depth[idx]))/256))
            depth = depth.resize(IMG_SIZE, Image.BILINEAR)
        
        elif self.task == "surface_normal":
            label = np.load(self.label[idx])
            label = label.transpose(2, 0, 1)
            depth = np.load(self.depth[idx])
        
        elif self.task == "surface_normal_nyu":
            
            label = Image.open(self.label[idx])
            res_label = np.array(label.resize(IMG_SIZE, Image.BILINEAR), dtype=np.float32)
            label  = ((res_label*2)/255)-1
            label = label.transpose(2, 0, 1)
            depth =  Image.open(self.depth[idx])
            
        elif self.task == "vanishing_point":
            label = []
            vps = np.load(self.label[idx])
            label.append(vps['x'][0])
            label.append(vps['y'][0])
            label.append(vps['z'][0])
            label.append(vps['x'][1])
            label.append(vps['y'][1])
            label.append(vps['z'][1])
            label.append(vps['x'][2])
            label.append(vps['y'][2])
            label.append(vps['z'][2])
            depth = Image.open(self.depth[idx])
            depth = depth.resize(IMG_SIZE, Image.BILINEAR)
        
            
        if self.transform:
            image, label, depth = self.transform(self.task, image, label, depth)
            
        return image, label, depth

# -------------------------------------------------------------------------------------------------------------------------------

# transformations  

def train_transformation(task, image, label):
    """
    applies transformations on input images and their labels
    """
    p = np.random.uniform(0, 1)
    
    if task == "surface_normal":
        
        image = torchvision.transforms.ToTensor()(image)
        label = torch.from_numpy(label).type(torch.float32)       
    
    elif task == "depth":
        image = torchvision.transforms.ToTensor()(image)
        label = torch.from_numpy(np.array(label)/255.).type(torch.float32)   
      

    elif task == "vanishing_point":
        
        image = torchvision.transforms.ToTensor()(image)      
        label = torch.Tensor(label)
        label = label.type(torch.float32)
    
    elif task == "segmentation":
        if p<=0.5:
            img_out = image.transpose(Image.FLIP_LEFT_RIGHT)
            image = torchvision.transforms.ToTensor()(img_out)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
            label = torch.from_numpy(np.array(label)).type(torch.LongTensor)
            label = torch.squeeze(label,1)
        else:
            image = torchvision.transforms.ToTensor()(image)
            label = torch.tensor(np.array(label)).type(torch.LongTensor)
            label = torch.squeeze(label,1)

            
    return image, label


def test_transformation(task, image, label):
    """
    applies transformations on input images and their labels
    """


    if task == "surface_normal" or task == "surface_normal_nyu":

        image = torchvision.transforms.ToTensor()(image)
        label = torch.from_numpy(label).type(torch.float32) 
        
    elif task == "vanishing_point":
        
        image = torchvision.transforms.ToTensor()(image)  
        label = torch.Tensor(label)
        label = label.type(torch.float32)
    
    elif task == "depth":

        image = torchvision.transforms.ToTensor()(image)
        label = torch.from_numpy(np.array(label)/255.).type(torch.float32)   
      

    elif task == "segmentation":
        
        image = torchvision.transforms.ToTensor()(image)
        label = torch.from_numpy(np.array(label)).type(torch.LongTensor)
        label = torch.squeeze(label,1)
        
    return image, label

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
        if self.task != 'surface_normal':
            image = image.resize(IMG_SIZE, Image.BILINEAR)
        
        if self.task == "segmentation":
            lab = np.array(Image.open(self.label[idx]))
            palette = list(MAPPING.keys())
            key = np.array(list(MAPPING.values()))
            index = np.digitize(lab.ravel(), palette, right=True)
            new_img = np.array(key[index].reshape(lab.shape)).astype(np.uint8)
            label = Image.fromarray(new_img)
            label = label.resize(IMG_SIZE, Image.NEAREST)
                 
        elif self.task == "surface_normal":
            label = np.load(self.label[idx])
            # label = Image.fromarray(np.uint8(lab*255))
            label = label.transpose(2, 0, 1)

        elif self.task == "surface_normal_nyu":
            
            label = Image.open(self.label[idx])
            res_label = np.array(label.resize(IMG_SIZE, Image.BILINEAR), dtype=np.float32)
            label  = ((res_label*2)/255)-1
            label = label.transpose(2, 0, 1)

        elif self.task == "depth":
            
            # lab = Image.open(self.label[idx])
            # label = Image.fromarray(np.uint8((np.array(lab)/256)))
            # label = label.resize(IMG_SIZE, Image.NEAREST)
            norm = np.load(self.label[idx])
            norm = np.reshape(norm, (norm.shape[0], norm.shape[1]))
            label = Image.fromarray(norm/331.08224)
            label = label.resize(IMG_SIZE, Image.NEAREST)

        elif self.task == "vanishing_point":
            label = []
            vps = np.load(self.label[idx])
            label.append(vps['x'][0])
            label.append(vps['y'][0])
            label.append(vps['z'][0])
            label.append(vps['x'][1])
            label.append(vps['y'][1])
            label.append(vps['z'][1])
            label.append(vps['x'][2])
            label.append(vps['y'][2])
            label.append(vps['z'][2])
    
        if self.transform:
            image, label = self.transform(self.task, image, label)
            
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

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def draw_training_curves(train_losses, test_losses, curve_name, task, timestr):
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
    plt.savefig(VIS_RESULTS_PATH+task+'/'+timestr+"/{}_curves.png".format(curve_name))
    
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Load datasets

# 1. Diode Dataset for Surface Normal

def load_diode_sn_dataset():
    """
    returns train and test images with their corresponding normals as labels
    """
    
    data = {"train": [], "val":[]}
    label = {"train": [], "val":[]}
    depth = {"train": [], "val":[]}
   
    data['train'] = glob.glob(SN_TRAIN + '*.png')
    for img in data['train']:
        base = img.split('/')[-1][:-4]
        norm = base + '_normal.npy'
        dep = base + '_depth.npy'
        label['train'].append(SN_TRAIN + norm)
        depth['train'].append(SN_TRAIN + dep)


    data['val'] = glob.glob(SN_VAL + '*.png')
    for img in data['val']:
        base = img.split('/')[-1][:-4]
        norm = base + '_normal.npy'
        dep = base + '_depth.npy'
        label['val'].append(SN_VAL + norm)
        depth['val'].append(SN_VAL + dep)
    
    print("Size of train images : ", len(data['train']))
    print("Size of val images : ", len(data['val']))
    
    print("Size of train depth : ", len(depth['train']))
    print("Size of val depth : ", len(depth['val']))
    
    print("Size of train Surface normal : ", len(label['train']))
    print("Size of val Surface normal : ", len(label['val']))
    
    return data, label, depth



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

    data_image = {'train':[], 'test':[], 'val': []}
    vanish_point = {'train':[], 'test':[], 'val': []}
    vanish_depth = {'train':[], 'test':[], 'val': []}

    def sort_images(temp):
        sortedimages = sorted(temp, key=lambda x: int(re.findall(r'\d+', x)[-1]))
        return sortedimages

    def get_vp_dataset(path, dtype):

        folders = os.listdir(path)
        main_images = []
        vanishing_points = []
        vanishing_depth = []
        
        for scene in folders:
            images = glob.glob(path + scene + '/*color.png')
            vanish = glob.glob(path + scene + '/*vanish.npz')
            depth = glob.glob(path + scene + '/[!frame]*.png')
            
            main_images.extend(sort_images(images))
            vanishing_points.extend(sort_images(vanish))
            vanishing_depth.extend(sort_images(depth))
            

        data_image[dtype] = main_images
        vanish_point[dtype] = vanishing_points
        vanish_depth[dtype] = vanishing_depth

        return

    get_vp_dataset(TRAIN_VP, 'train')
    get_vp_dataset(VAL_VP, 'val')
    get_vp_dataset(TEST_VP, 'test')   


    print("Length of train data: ",len(data_image['train']))
    print("Length of val data: ",len(data_image['val']))
    print("Length of test data: ",len(data_image['test']))

    print("Length of train vp: ",len(vanish_point['train']))
    print("Length of val vp: ",len(vanish_point['val']))
    print("Length of test vp: ",len(vanish_point['test']))

    print("Length of train depth: ",len(vanish_depth['train']))
    print("Length of val depth: ",len(vanish_depth['val']))
    print("Length of test depth: ",len(vanish_depth['test']))

    return data_image, vanish_point, vanish_depth


# 4. Load SUNRGBD dataset

def load_sunrgbd_dataset():
    """
    loads sunrgbd dataset rgb images with their corresponding segmentation and depth images.
    """

    images = {'train':[], 'val':[]}
    depth = {'train':[], 'val':[]}
    seg = {'train':[], 'val':[]} 

    with open(SUNRGBD_TRAIN_IMAGES, 'r') as f:
        for line in f:
            images['train'].append(os.path.join(REL, line.strip('\n')))

    with open(SUNRGBD_TEST_IMAGES, 'r') as f:
        for line in f:
            images['val'].append(os.path.join(REL, line.strip('\n')))

    with open(SUNRGBD_TRAIN_DEPTH, 'r') as f:
        for line in f:
            depth['train'].append(os.path.join(REL, line.strip('\n')))

    with open(SUNRGBD_TEST_DEPTH, 'r') as f:
        for line in f:
            depth['val'].append(os.path.join(REL, line.strip('\n')))

    with open(SUNRGBD_TRAIN_SEG, 'r') as f:
        for line in f:
            seg['train'].append(os.path.join(REL, line.strip('\n')))

    with open(SUNRGBD_TEST_SEG, 'r') as f:
        for line in f:
            seg['val'].append(os.path.join(REL, line.strip('\n')))

    print("\nImages: ")
    print("Size of training data : ", len(images['train']))
    print("Size of validation data : ", len(images['val']))
    print("\nDepth: ")
    print("Size of training data : ", len(depth['train']))
    print("Size of validation data : ", len(depth['val']))
    print("\nSegmentation: ")
    print("Size of training data : ", len(seg['train']))
    print("Size of validation data : ", len(seg['val']), "\n")


    return images, seg, depth

def load_sn_nyuv2_dataset():
    """
    loads sunrgbd dataset rgb images with their corresponding segmentation and depth images.
    """

    images = {'test':[]}
    depth = {'test':[]}
    sn = {'test':[]} 

    images['test'] = glob.glob(NYU_RGB + '*.png')
    depth['test'] = glob.glob(NYU_DEPTH + '*.png')
    sn['test'] = glob.glob(NYU_SN + '*.png')


    print("\nImages: ")
    print("Size of test data : ", len(images['test']))
    print("\nDepth: ")
    print("Size of test data : ", len(depth['test']))
    print("\nSurface Normal: ")
    print("Size of test data : ", len(sn['test']), "\n")

    return images, sn, depth

def load_sun_nyuv2_dataset():
    """
    loads sunrgbd dataset rgb images with their corresponding segmentation and depth images.
    """

    images = {'test':[]}
    depth = {'test':[]}
    seg = {'test':[]} 

    with open(NYU_TEST_IMAGES, 'r') as f:
        for line in f:
            images['test'].append(os.path.join(REL, line.strip('\n')))

    with open(NYU_TEST_DEPTH, 'r') as f:
        for line in f:
            depth['test'].append(os.path.join(REL, line.strip('\n')))

    with open(NYU_TEST_SEG, 'r') as f:
        for line in f:
            seg['test'].append(os.path.join(REL, line.strip('\n')))

    print("\nImages: ")
    print("Size of test data : ", len(images['test']))
    print("\nDepth: ")
    print("Size of test data : ", len(depth['test']))
    print("\nSegmentation: ")
    print("Size of test data : ", len(seg['test']), "\n")

    return images, seg, depth
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

