from utils import load_scannet_vp_dataset
from utils import EarlyStopping
from networks import SegNet
from utils import get_data_loader
import os
from training import train_model
from training import train_loop
from utils import draw_training_curves
from training import run_inference
import torch.nn as nn
import time, torch
timestr = time.strftime("%Y%m%d-%H%M%S")
DEVICE =  torch.device("cuda:1" if torch.cuda.is_available() else "cpu")



BATCH_SIZE = 64
LR = 1e-3
REL = os.getcwd()
CHECKPOINT_DIR = REL + '/checkpoints/'
VIS_RESULTS_PATH = REL + '/results/'
threshold = 40
os.makedirs(VIS_RESULTS_PATH+'vanishing_point'+'/'+timestr)


results_record = open(VIS_RESULTS_PATH+'vanishing_point'+'/'+timestr+ "/exp_results.csv", 'w+')

data, label, depth = load_scannet_vp_dataset()
train_loader = get_data_loader('vanishing_point', data, label, "train", BATCH_SIZE)
val_loader = get_data_loader('vanishing_point', data, label, "val", BATCH_SIZE)
test_loader = get_data_loader('vanishing_point', data, label, "test", BATCH_SIZE)

# train model
losses, accuracies, model, flag = train_model('vanishing_point', train_loader, val_loader, timestr, results_record, DEVICE, LR, threshold)

# plot trained metrics 
loss_curve = "loss"
draw_training_curves(losses[0], losses[1],loss_curve, 'vanishing_point', timestr)
