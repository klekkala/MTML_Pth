from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time, torch
import os
from utils import load_sunrgbd_dataset
from utils import EarlyStopping
from utils import get_multitask_data_loader
from utils import ConfMatrix
from utils import draw_training_curves
from training import run_inference
from training import train_model
from training import train_loop
from networks import SegNet

BATCH_SIZE = 16
LR = 1e-4

timestr = time.strftime("%Y%m%d-%H%M%S")
REL = os.getcwd()
VIS_RESULTS_PATH = REL + '/results/'
os.makedirs(VIS_RESULTS_PATH+'segmentation_depth'+'/'+timestr)
CHECKPOINT_DIR = REL + '/checkpoints/'
threshold = 7
DEVICE =  torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

images, seg, depth = load_sunrgbd_dataset()

train_loader = get_multitask_data_loader('segmentation', images, seg, depth,  "train", BATCH_SIZE)
val_loader = get_multitask_data_loader('segmentation', images, seg, depth, "val", BATCH_SIZE)

results_record = open(VIS_RESULTS_PATH+'segmentation_depth'+'/'+timestr+ "/exp_results.csv", 'w+')

# train model
losses, accuracies, model, flag = train_model('segmentation_depth', train_loader, val_loader, timestr, results_record, DEVICE, LR, threshold)

# plot trained metrics
loss_curve = "loss"
draw_training_curves(losses[0], losses[1],loss_curve, 'segmentation_depth', timestr)
loss_curve = "accuracy"
draw_training_curves(accuracies[0], accuracies[1],loss_curve, 'segmentation_depth', timestr)