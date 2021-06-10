
import time, torch
import os
from datasets import  load_sun_nyuv2_dataset
from utils import get_data_loader
from utils import draw_training_curves
from singletask_training import task_inference
from singletask_training import train_model

BATCH_SIZE = 22


timestr = time.strftime("%Y%m%d-%H%M%S")
REL = os.getcwd()
VIS_RESULTS_PATH = REL + '/results/'
os.makedirs(VIS_RESULTS_PATH+'depth'+'/'+timestr)
CHECKPOINT_DIR = REL + '/checkpoints/'
LR = 1e-3
TASK = 'depth'
THRESHOLD = 12
DEVICE =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


data, seg, depth = load_sun_nyuv2_dataset()


train_loader = get_data_loader(TASK, data['train'], depth['train'], BATCH_SIZE)
val_loader = get_data_loader(TASK, data['test'], depth['test'], BATCH_SIZE)

results_record = open(VIS_RESULTS_PATH+'depth'+'/'+timestr+ "/exp_results.csv", 'w+')

# train model
losses, accuracies, model, flag = train_model(TASK, train_loader, val_loader, timestr, results_record, DEVICE, LR, THRESHOLD)

# plot trained metrics
loss_curve = "loss"
draw_training_curves(losses[0], losses[1],loss_curve, 'depth', timestr)
