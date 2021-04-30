from utils import load_diode_sn_dataset
from utils import EarlyStopping
from networks import SegNet
from utils import get_data_loader
from training import train_model
from training import train_loop
from utils import draw_training_curves
from training import run_inference
import time, torch
import os
timestr = time.strftime("%Y%m%d-%H%M%S")

REL = os.getcwd()

VIS_RESULTS_PATH = REL + '/results/'
os.makedirs(VIS_RESULTS_PATH+'surface_normal'+'/'+timestr)

CHECKPOINT_DIR = REL + '/checkpoints/'
DEVICE =  torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 16

data, labels = load_diode_sn_dataset()

train_loader =  get_data_loader('surface_normal', data, labels, "train", BATCH_SIZE)
val_loader =  get_data_loader('surface_normal', data, labels, "val", BATCH_SIZE)
test_loader =  get_data_loader('surface_normal', data, labels, "test", BATCH_SIZE)

results_record = open(VIS_RESULTS_PATH+'surface_normal'+'/'+timestr+ "/exp_results.csv", 'w+')
# train model
losses, accuracies, model, flag = train_model('surface_normal', train_loader, val_loader, timestr, results_record, DEVICE)

# plot trained metrics
loss_curve = "loss"
draw_training_curves(losses[0], losses[1],loss_curve, 'surface_normal', timestr)

