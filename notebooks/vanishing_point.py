from utils import load_scannet_vp_dataset
from utils import EarlyStopping
from networks import VGG16Model
from utils import get_data_loader
import os
from training import train_model
from training import train_loop
from utils import draw_training_curves
from training import run_inference
import torch.nn as nn
import time

timestr = time.strftime("%Y%m%d-%H%M%S")

BATCH_SIZE = 182

CHECKPOINT_DIR = '/home4/shubham/MTML_Pth/checkpoints/'
VIS_RESULTS_PATH = '/home4/shubham/MTML_Pth/results/'
os.makedirs(VIS_RESULTS_PATH+'vanishing_point'+'/'+timestr)


data, label = load_scannet_vp_dataset()
train_loader = get_data_loader('vanishing_point', data, label, "train", BATCH_SIZE)
val_loader = get_data_loader('vanishing_point', data, label, "val", BATCH_SIZE)
test_loader = get_data_loader('vanishing_point', data, label, "test", BATCH_SIZE)

# train model
losses, accuracies, model, flag = train_model('vanishing_point', train_loader, val_loader, timestr)

# plot trained metrics 
loss_curve = "loss"
draw_training_curves(losses[0], losses[1],loss_curve, 'vanishing_point', timestr)

if flag:
    model = VGG16Model().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_DIR+'vanishing_point/'+timestr+'/early_stopping_model.pth'))
    
run_inference('vanishing_point', model, test_loader)