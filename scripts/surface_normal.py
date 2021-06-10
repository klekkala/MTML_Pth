## ln[]:
import time, torch
import os
from datasets import load_diode_sn_dataset, load_sn_nyuv2_dataset
from utils import get_data_loader
from utils import draw_training_curves
from singletask_training import train_model
from singletask_training import task_inference

timestr = time.strftime("%Y%m%d-%H%M%S")

# PATHS
REL = os.getcwd()
VIS_RESULTS_PATH = REL + '/results/'
CHECKPOINT_DIR = REL + '/checkpoints/'

# CONSTANTS
DEVICE =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 42
LR = 1e-3
THRESHOLD = 8
INFERENCE = False
NYU = False
TASK = 'surface_normal'

if NYU:
    data, label, depth = load_sn_nyuv2_dataset()
else:
    data, label, depth = load_diode_sn_dataset()

if INFERENCE:

    best_model = CHECKPOINT_DIR + TASK + '/20210503-175337/early_stopping_model.pth' 
    best_result = VIS_RESULTS_PATH + TASK + '/20210503-175337/' 

    if NYU:
        filename = 'nyu_test_results.txt'
        test_loader = get_data_loader('surface_normal_nyu', data, label, "test", BATCH_SIZE)
    else:
        filename = 'test_results.txt'
        test_loader = get_data_loader(TASK, data, label, "test", BATCH_SIZE)

    task_inference(TASK, test_loader, best_model, best_result, DEVICE, filename)

else:

    if NYU:
        train_loader =  get_data_loader('surface_normal_nyu', data['train'], label['train'], BATCH_SIZE)
        val_loader = get_data_loader('surface_normal_nyu', data['test'], label['test'], BATCH_SIZE)
    else:
        train_loader =  get_data_loader(TASK, data['train'], label['train'], BATCH_SIZE)
        val_loader = get_data_loader(TASK, data['val'], label['val'], BATCH_SIZE)
    # train model
    os.makedirs(VIS_RESULTS_PATH + TASK + '/' +timestr)
    results_record = open(VIS_RESULTS_PATH + TASK + '/' + timestr + "/exp_results.csv", 'w+')
    losses, accuracies, model, flag = train_model(TASK, train_loader, val_loader, timestr, results_record, DEVICE, LR, THRESHOLD)

    # plot trained metrics
    loss_curve = "loss"
    draw_training_curves(losses[0], losses[1],loss_curve, TASK, timestr)

