from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time, torch
import os
from utils import load_sunrgbd_dataset
from utils import EarlyStopping
from utils import get_data_loader
from utils import ConfMatrix
from utils import draw_training_curves
from training import run_inference
from training import train_model
from training import train_loop
from networks import SegNet

BATCH_SIZE = 32


timestr = time.strftime("%Y%m%d-%H%M%S")
REL = os.getcwd()
VIS_RESULTS_PATH = REL + '/results/'
os.makedirs(VIS_RESULTS_PATH+'segmentation'+'/'+timestr)
CHECKPOINT_DIR = REL + '/checkpoints/'

DEVICE =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

images, seg, depth = load_sunrgbd_dataset()

train_loader = get_data_loader('segmentation', images, seg, "train", BATCH_SIZE)
val_loader = get_data_loader('segmentation', images, seg, "val", BATCH_SIZE)

results_record = open(VIS_RESULTS_PATH+'segmentation'+'/'+timestr+ "/exp_results.csv", 'w+')

# train model
losses, accuracies, model, flag = train_model('segmentation', train_loader, val_loader, timestr, results_record, DEVICE)

# plot trained metrics
loss_curve = "loss"
draw_training_curves(losses[0], losses[1],loss_curve, 'segmentation', timestr)
loss_curve = "accuracy"
draw_training_curves(accuracies[0], accuracies[1],loss_curve, 'segmentation', timestr)