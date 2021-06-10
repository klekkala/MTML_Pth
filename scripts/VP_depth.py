import time, torch, os
from datasets import load_scannet_vp_dataset
from utils import draw_training_curves
from utils import get_multitask_data_loader
from multitask_training import task_inference
from multitask_training import train_model

timestr = time.strftime("%Y%m%d-%H%M%S")

# CONSTANTS
DEVICE =  torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
LR = 1e-3
INFERENCE = False
THRESHOLD = 15
TASK = 'vanishing_point_depth'

# PATHS
REL = os.getcwd()
CHECKPOINT_DIR = REL + '/checkpoints/'
VIS_RESULTS_PATH = REL + '/results/'


data, label, depth = load_scannet_vp_dataset()


if INFERENCE:
    best_model = CHECKPOINT_DIR + TASK + '/20210430-114833/early_stopping_model.pth' 
    best_result = VIS_RESULTS_PATH + TASK + '/20210430-114833/' 

    test_loader = get_multitask_data_loader('vanishing_point', data, label, depth, "test", BATCH_SIZE)

    multitask_inference(TASK, test_loader, best_model, best_result, DEVICE)
else:
    train_loader = get_multitask_data_loader('vanishing_point', data['train'], label['train'], depth['train'], BATCH_SIZE)
    val_loader = get_multitask_data_loader('vanishing_point', data['val'], label['val'], depth['val'], BATCH_SIZE)

    # train model
    os.makedirs(VIS_RESULTS_PATH + TASK + '/' + timestr)
    results_record = open(VIS_RESULTS_PATH + TASK + '/'+ timestr + "/exp_results.csv", 'w+')

    losses, accuracies, model, flag = train_model(TASK, train_loader, val_loader, timestr, results_record, DEVICE, LR, THRESHOLD)

    # plot trained metrics 
    loss_curve = "loss"
    draw_training_curves(losses[0], losses[1], loss_curve, TASK, timestr)
