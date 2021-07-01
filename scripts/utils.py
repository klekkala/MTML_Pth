# Libraries
import torch, torchvision
import numpy as np
import re, os, time
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
import matplotlib.pyplot as plt
import sys
import torch.nn.functional as F
eps = sys.float_info.epsilon

timestr = time.strftime("%Y%m%d-%H%M%S")
ImageFile.LOAD_TRUNCATED_IMAGES = True

# constant variables
IMG_SIZE = (384, 288)

# checkpoint path
REL = os.getcwd()
DEFAULT_CHECKPOINT_DIR = REL + '/checkpoints/' 
VIS_RESULTS_PATH = REL + '/results/'


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
    def __init__(self, patience=12, verbose=False, delta=0.0001, path=DEFAULT_CHECKPOINT_DIR+'/early_stopping_model.pth', task=None, timestamp='current'):
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
    
    def __call__(self, val_loss, model, prefix):
        
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, prefix)
            
        elif score < self.best_score + self.delta:
            self.counter += 1
            
            if self.counter >= self.patience:
                self.early_stop = True
                
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, prefix)
            self.counter = 0   
    
    def save_checkpoint(self, val_loss, model, prefix):
        """
        saves the current best version of the model if there is decrease in validation loss
        """
        self.path = DEFAULT_CHECKPOINT_DIR + self.task + '/'+ self.timestr +'/' + prefix + '_early_stopping_model.pth'
        torch.save(model.state_dict(), self.path)
        self.vall_loss_min = val_loss


class ConfMatrix(object):
    """
    This class calculates the pixel level accuracy and Pixel mIoU for Segmentation
    """
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
        h = h[1:,1:]
        acc = torch.diag(h).sum() / h.sum()
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h) + eps)
        return torch.mean(iu), acc

# class ConfMatrix(object):
#     def __init__(self, num_classes):
#         self.num_classes = num_classes
#         self.mat = None

#     def update(self, pred, target):
#         n = self.num_classes
#         if self.mat is None:
#             self.mat = torch.zeros((n, n), dtype=torch.int64, device=pred.device)
#         with torch.no_grad():
#             k = (target >= 0) & (target < n)
#             inds = n * target[k].to(torch.int64) + pred[k]
#             self.mat += torch.bincount(inds, minlength=n ** 2).reshape(n, n)

#     def get_metrics(self):
#         h = self.mat.float()
#         acc = torch.diag(h).sum() / h.sum()
#         iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
#         return torch.mean(iu), acc
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def multitask_transformation(task, image, label, depth):
    """
    applies transformations on input images and their labels
    """
    if task == "surface_normal":
        
        image = torchvision.transforms.ToTensor()(image)
        label = torch.from_numpy(label).type(torch.float32)  
        depth = torch.from_numpy(np.array(depth)).type(torch.float32)
        
    elif task == "surface_normal_nyu":
        
        image = torchvision.transforms.ToTensor()(image)
        label = torch.from_numpy(label).type(torch.float32)  
        depth = torch.from_numpy(np.array(depth)/255).type(torch.float32)

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


def get_multitask_data_loader(task, data, label, depth, batch_size):
    """
    returns train/test/val dataloaders
    params: flag - train/test/val
            task - segmentation, surface_normal, vanishing_point
            batch_size - batch_size of the dataset
            data, label - dataset with its ground truth label
    """

    dataset = Multitask_DatasetLoader(task,  data, label, depth, transform=multitask_transformation) 
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
            ch1, ch2, ch3 = label.split()
            label = Image.fromarray(np.dstack((ch1,ch3,ch2)))
            res_label = np.array(label.resize(IMG_SIZE, Image.BILINEAR), dtype=np.float32)
            label  = ((res_label*2)/255)-1
            label = label.transpose(2, 0, 1)
            depth =  np.uint8(np.array(Image.open(self.depth[idx]))/256)
            depth = Image.fromarray(depth)
            depth = depth.resize(IMG_SIZE, Image.BILINEAR)

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

def singletask_transformation(task, image, label):
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
            label = label.transpose(2, 0, 1)

        elif self.task == "depth":
            depth = Image.fromarray(np.uint8(np.array(Image.open(self.label[idx]))/256))
            label = depth.resize(IMG_SIZE, Image.BILINEAR)

        elif self.task == "surface_normal_nyu":
            label = Image.open(self.label[idx])
            label = np.array(label.resize(IMG_SIZE, Image.BILINEAR), dtype=np.float32)/255
            label = label.transpose(2, 0, 1)


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


def get_data_loader(task, data, label, batch_size):
    """
    returns train/test/val dataloader
    params: 
            task - segmentation, surface_normal, vanishing_point
            batch_size - batch_size of the dataset
            data, label - dataset with its ground truth label
    """
 
    dataset = DatasetLoader(task, data, label, transform=singletask_transformation) 
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

def sort_images(temp):
    
    sortedimages = sorted(temp, key=lambda x: int(re.findall(r'\d+', x)[-1]))
    return sortedimages

# ------------------------------------------------------------------ MTAN REPO ---------------------------------------------------------------------------------------------------------------------------------

def depth_error(x_pred, x_output):

    # x_output = x_output.unsqueeze(1)
    device = x_pred.device
    binary_mask = (torch.sum(x_output, dim=1) != 0).unsqueeze(1).to(device)
    x_pred_true = x_pred.masked_select(binary_mask)
    x_output_true = x_output.masked_select(binary_mask)

    abs_err = torch.abs(x_pred_true - x_output_true)
    rel_err = torch.abs(x_pred_true - x_output_true) / x_output_true
    return (torch.sum(abs_err) / torch.nonzero(binary_mask, as_tuple=False).size(0)).item(), \
           (torch.sum(rel_err) / torch.nonzero(binary_mask, as_tuple=False).size(0)).item()

def normal_error(x_pred, x_output):
    binary_mask = (torch.sum(x_output, dim=1) != 0)
    error = torch.acos(torch.clamp(torch.sum(x_pred * x_output, 1).masked_select(binary_mask), -1, 1)).detach().cpu().numpy()
    error = np.degrees(error)
    return np.mean(error), np.median(error), np.mean(error < 11.25), np.mean(error < 22.5), np.mean(error < 30)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_fit(x_pred, x_output, task_type):
    device = x_pred.device

    # binary mark to mask out undefined pixel space
    binary_mask = (torch.sum(x_output, dim=1) != 0).float().unsqueeze(1).to(device)

    if task_type == 'semantic':
        # semantic loss: depth-wise cross entropy
        loss = F.nll_loss(x_pred, x_output, ignore_index=-1)

    if task_type == 'depth':
        # depth loss: l1 norm
        loss = torch.sum(torch.abs(x_pred - x_output) * binary_mask) / torch.nonzero(binary_mask, as_tuple=False).size(0)

    if task_type == 'normal':
        # normal loss: dot product
        loss = 1 - torch.sum((x_pred * x_output) * binary_mask) / torch.nonzero(binary_mask, as_tuple=False).size(0)

    return loss