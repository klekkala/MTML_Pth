import time, torch, os
from datasets import load_sunrgbd_dataset
from datasets import load_sun_nyuv2_dataset
from utils import get_multitask_data_loader
from utils import draw_training_curves
from multitask_training import task_inference
from multitask_training import train_model
from multitask_training import retrain_depth, multi_task_trainer, multi_task_eval
from models import SegNet

timestr = time.strftime("%Y%m%d-%H%M%S")

# CONSTANTS
INFERENCE = False             
NYU = True
RETRAIN = False
FINAL_TASK = False
MTAN = False
MTAN_INF = True

BATCH_SIZE = 6
LR = 1e-3
DEVICE =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
THRESHOLD = 12
TASK = 'segmentation_depth'

# PATHS
REL = os.getcwd()
VIS_RESULTS_PATH = REL + '/results/'
CHECKPOINT_DIR = REL + '/checkpoints/'


if NYU:
    data, label, depth = load_sun_nyuv2_dataset()
else:
    data, label, depth = load_sunrgbd_dataset()

if MTAN:

    SegNet_MTAN = SegNet().to(DEVICE)
    if NYU:
        train_loader = get_multitask_data_loader('segmentation', data['train'], label['train'], depth['train'],   BATCH_SIZE)
        test_loader = get_multitask_data_loader('segmentation', data['test'], label['test'], depth['test'],  BATCH_SIZE)
    else:
        train_loader = get_multitask_data_loader('segmentation', data['train'], label['train'], depth['train'],   BATCH_SIZE)
        test_loader = get_multitask_data_loader('segmentation', data['val'], label['val'], depth['val'],  BATCH_SIZE)

    os.makedirs(VIS_RESULTS_PATH + TASK + '/' + timestr)
    results_record = open(VIS_RESULTS_PATH + TASK + '/' + timestr + "/mtan_exp_results.csv", 'w+')

    losses, accuracies = multi_task_trainer(TASK, timestr, train_loader, test_loader, SegNet_MTAN, DEVICE, 200, results_record)

    # plot trained metrics
    loss_curve = "loss"
    draw_training_curves(losses[0], losses[1], loss_curve, TASK, timestr)
    loss_curve = "accuracy"
    draw_training_curves(accuracies[0], accuracies[1], loss_curve, TASK, timestr)

elif MTAN_INF:

    best_result = VIS_RESULTS_PATH + TASK + '/20210621-161807/' 

    if NYU:
        filename = 'nyu_test_results.txt'
        best_model = CHECKPOINT_DIR + TASK + '/20210621-161807/mtan_early_stopping_model.pth'

    test_loader = get_multitask_data_loader('segmentation', data['test'], label['test'], depth['test'],  BATCH_SIZE)
    multi_task_eval(TASK, test_loader, best_model, best_result, DEVICE, filename)

elif INFERENCE:

    best_result = VIS_RESULTS_PATH + TASK + '/20210512-181927/' 

    if NYU:
        if FINAL_TASK:
            filename = 'nyu_final_test_results.txt'
            best_model = CHECKPOINT_DIR + TASK + '/20210512-181927/seg_early_stopping_model.pth' 
        else:
            filename = 'nyu_test_results.txt'
            best_model = CHECKPOINT_DIR + TASK + '/20210512-181927/early_stopping_model.pth'
    else:
        filename = 'test_results.txt'

    test_loader = get_multitask_data_loader('segmentation', data['test'], label['test'], depth['test'],  BATCH_SIZE)

    task_inference(TASK, test_loader, best_model, best_result, DEVICE, filename)


elif RETRAIN:

    retrain_task = "segmentation_retrain"

    best_model = CHECKPOINT_DIR + TASK + '/20210512-181927/early_stopping_model.pth' 
    best_result = VIS_RESULTS_PATH + TASK + '/20210512-181927/' 
    filename = 'nyu_retrain_results.txt'
    timestr = '20210512-181927'

    train_loader = get_multitask_data_loader('segmentation', data['train'], label['train'], depth['train'],   BATCH_SIZE)
    test_loader = get_multitask_data_loader('segmentation', data['test'], label['test'], depth['test'],  BATCH_SIZE)

    losses = retrain_depth(retrain_task, best_model, train_loader, test_loader, timestr, filename, best_result, DEVICE, LR, THRESHOLD)

    # plot trained metrics
    loss_curve = "retrain_loss"
    draw_training_curves(losses[0], losses[1], loss_curve, TASK, timestr)

else:

    train_loader = get_multitask_data_loader('segmentation', data['train'], label['train'], depth['train'],   BATCH_SIZE)
    val_loader = get_multitask_data_loader('segmentation', data['val'], label['val'], depth['val'],  BATCH_SIZE)

    # train model
    os.makedirs(VIS_RESULTS_PATH + TASK + '/' + timestr)
    results_record = open(VIS_RESULTS_PATH + TASK + '/' + timestr + "/exp_results.csv", 'w+')

    losses, accuracies, model, flag = train_model(TASK, train_loader, val_loader, timestr, results_record, DEVICE, LR, THRESHOLD)

    # plot trained metrics
    loss_curve = "loss"
    draw_training_curves(losses[0], losses[1], loss_curve, TASK, timestr)
    loss_curve = "accuracy"
    draw_training_curves(accuracies[0], accuracies[1], loss_curve, TASK, timestr)