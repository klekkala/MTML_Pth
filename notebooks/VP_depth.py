from utils import load_scannet_vp_dataset
from utils import EarlyStopping
from utils import get_data_loader
import os
from training import train_model
from training import train_loop
from utils import draw_training_curves
from utils import get_multitask_data_loader

import torch.nn as nn
import time, torch

timestr = time.strftime("%Y%m%d-%H%M%S")
DEVICE =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
LR = 1e-3
REL = os.getcwd()
CHECKPOINT_DIR = REL + '/checkpoints/'
VIS_RESULTS_PATH = REL + '/results/'
threshold = 30
os.makedirs(VIS_RESULTS_PATH+'vanishing_point_depth'+'/'+timestr)


results_record = open(VIS_RESULTS_PATH+'vanishing_point_depth'+'/'+timestr+ "/exp_results.csv", 'w+')

data, label, depth = load_scannet_vp_dataset()

train_loader = get_multitask_data_loader('vanishing_point', data, label, depth, "train", BATCH_SIZE)
val_loader = get_multitask_data_loader('vanishing_point', data, label, depth, "val", BATCH_SIZE)

# train model
losses, accuracies, model, flag = train_model('vanishing_point_depth', train_loader, val_loader, timestr, results_record, DEVICE, LR, threshold)

# plot trained metrics 
loss_curve = "loss"
draw_training_curves(losses[0], losses[1],loss_curve, 'vanishing_point_depth', timestr)
