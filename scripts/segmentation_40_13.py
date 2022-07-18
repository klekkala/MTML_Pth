import time, os
import torch
from datasets import load_sunrgbd_dataset, load_sun_nyuv2_dataset
from utils import get_data_loader
from utils import draw_training_curves
from singletask_training import task_inference
from singletask_training import train_model

BATCH_SIZE = 16
INFERENCE = True
NYU = False

timestr = time.strftime("%Y%m%d-%H%M%S")

REL = os.getcwd()
VIS_RESULTS_PATH = REL + '/results/'
CHECKPOINT_DIR = REL + '/checkpoints/'
DEVICE =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
THRESHOLD = 20
LR = 1e-3
TASK = 'segmentation'

if NYU:
    data, label, depth = load_sun_nyuv2_dataset()
else:
    data, label, depth = load_sunrgbd_dataset()

if INFERENCE:

    best_model = CHECKPOINT_DIR + TASK + '/20210512-181847/early_stopping_model.pth'  
    best_result = VIS_RESULTS_PATH + TASK + '/20210512-181847/' 

    if NYU:
        filename = 'nyu_test_results.txt'
    else:
        filename = 'test_results.txt'

    test_loader = get_data_loader(TASK, data['val'], label['val'],  BATCH_SIZE)

    task_inference(TASK, test_loader, best_model, best_result, DEVICE, filename)

else:

    train_loader = get_data_loader(TASK, data['train'], label['train'], BATCH_SIZE)

    if NYU:
        val_loader = get_data_loader(TASK, data['test'], label['test'], BATCH_SIZE)
    else:
        val_loader = get_data_loader(TASK, data['val'], label['val'], BATCH_SIZE)

    os.makedirs(VIS_RESULTS_PATH + TASK + '/' + timestr)
    results_record = open(VIS_RESULTS_PATH + TASK + '/' + timestr+ "/exp_results.csv", 'w+')

    losses, accuracies, model, flag = train_model(TASK, train_loader, val_loader, timestr, results_record, DEVICE, LR, THRESHOLD)

    # plot trained metrics
    loss_curve = "loss"
    draw_training_curves(losses[0], losses[1],loss_curve, TASK, timestr)
    loss_curve = "accuracy"
    draw_training_curves(accuracies[0], accuracies[1],loss_curve, TASK, timestr)