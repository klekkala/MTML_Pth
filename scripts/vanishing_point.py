import time, torch, os
from datasets import load_scannet_vp_dataset
from utils import get_data_loader
from utils import draw_training_curves
from singletask_training import train_model
from singletask_training import task_inference

timestr = time.strftime("%Y%m%d-%H%M%S")

# CONSTANTS
DEVICE =  torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
INFERENCE = False
BATCH_SIZE = 32
LR = 1e-3
THRESHOLD = 20
TASK = 'vanishing_point'

# PATHS
REL = os.getcwd()
CHECKPOINT_DIR = REL + '/checkpoints/'
VIS_RESULTS_PATH = REL + '/results/'


data, label, depth = load_scannet_vp_dataset()

if INFERENCE:

    best_model = CHECKPOINT_DIR + TASK + '/20210430-083911/early_stopping_model.pth' 
    best_result = VIS_RESULTS_PATH + TASK + '/20210430-083911/' 

    test_loader = get_data_loader(TASK, data, label, "test", BATCH_SIZE)

    singletask_inference(TASK, test_loader, best_model, best_result, DEVICE)

else:

    train_loader = get_data_loader(TASK, data['train'], label['train'], BATCH_SIZE)
    val_loader = get_data_loader(TASK, data['val'], label['val'], BATCH_SIZE)

    # train model
    os.makedirs(VIS_RESULTS_PATH + TASK + '/' + timestr)
    results_record = open(VIS_RESULTS_PATH + TASK + '/' + timestr + "/exp_results.csv", 'w+')
    losses, accuracies, model, flag = train_model(TASK, train_loader, val_loader, timestr, results_record, DEVICE, LR, THRESHOLD)

    # plot trained metrics 
    loss_curve = "loss"
    draw_training_curves(losses[0], losses[1],loss_curve, TASK, timestr)
