import time, torch, os
from datasets import load_diode_sn_dataset, load_sn_nyuv2_dataset
from utils import get_multitask_data_loader
from utils import draw_training_curves
from multitask_training import task_inference
from multitask_training import train_model
from multitask_training import retrain_depth

timestr = time.strftime("%Y%m%d-%H%M%S")

# PATHS
REL = os.getcwd()
VIS_RESULTS_PATH = REL + '/results/'
CHECKPOINT_DIR = REL + '/checkpoints/'

# CONSTANTS
THRESHOLD = 10
BATCH_SIZE = 16
LR = 1e-1
DEVICE =  torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
TASK = 'surface_normal_depth'
NYU = True
INFERENCE = True
RETRAIN = False
FINAL_TASK = True

if NYU:
    data, label, depth = load_sn_nyuv2_dataset()
else:
    data, label, depth = load_diode_sn_dataset()


if INFERENCE:
     
    best_result = VIS_RESULTS_PATH + TASK + '/20210512-184036/' 

    if NYU:
        if FINAL_TASK:
            filename = 'channel_nyu_final_results.txt'
            best_model = CHECKPOINT_DIR + TASK + '/20210512-184036/sn_early_stopping_model.pth'
        else:
            filename = 'channel_nyu_test_results.txt'
            best_model = CHECKPOINT_DIR + TASK + '/20210512-184036/early_stopping_model.pth'

        test_loader = get_multitask_data_loader('surface_normal_nyu', data['test'], label['test'], depth['test'], BATCH_SIZE)

    else:
        filename = 'test_results.txt'
        test_loader = get_multitask_data_loader('surface_normal', data['test'], label['test'], depth['test'], BATCH_SIZE)

    task_inference(TASK, test_loader, best_model, best_result, DEVICE, filename)

elif RETRAIN:

    retrain_task = "surface_normal_retrain"

    best_model = CHECKPOINT_DIR + TASK + '/20210512-184036/early_stopping_model.pth' 
    best_result = VIS_RESULTS_PATH + TASK + '/20210512-184036/' 
    filename = 'nyu_retrain_results.txt'
    timestr = '20210512-184036'

    train_loader = get_multitask_data_loader('surface_normal_nyu', data['train'], label['train'], depth['train'],   BATCH_SIZE)
    test_loader = get_multitask_data_loader('surface_normal_nyu', data['test'], label['test'], depth['test'],  BATCH_SIZE)

    losses = retrain_depth(retrain_task, best_model, train_loader, test_loader, timestr, filename, best_result, DEVICE, LR, THRESHOLD)

    # plot trained metrics
    loss_curve = "retrain_loss"
    draw_training_curves(losses[0], losses[1], loss_curve, TASK, timestr)

else:

    train_loader = get_multitask_data_loader('surface_normal', data['train'], label['train'], depth['train'], BATCH_SIZE)
    val_loader = get_multitask_data_loader('surface_normal', data['val'], label['val'], depth['val'], BATCH_SIZE)

    # train model
    os.makedirs(VIS_RESULTS_PATH + TASK + '/' + timestr)
    results_record = open(VIS_RESULTS_PATH + TASK + '/' + timestr +  "/exp_results.csv", 'w+')
    losses, accuracies, model, flag = train_model(TASK, train_loader, val_loader, timestr, results_record, DEVICE, LR, THRESHOLD)

    # plot trained metrics
    loss_curve = "loss"
    draw_training_curves(losses[0], losses[1],loss_curve, TASK, timestr)
