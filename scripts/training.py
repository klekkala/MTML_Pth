import torch
import numpy as np
import os
import torchvision
import time
import torch.nn as nn
from PIL import Image, ImageFile
import matplotlib.pyplot as plt
from utils import EarlyStopping
from networks import SegNet
from utils import get_data_loader
from utils import ConfMatrix
from tempfile import TemporaryFile
from numpy import savez_compressed

EPOCHS = 500

REL = os.getcwd()
CHECKPOINT_DIR = REL + '/checkpoints/' 


def singletask_inference(task, test_loader, best_model, best_result, DEVICE, filename):
    """
    returns performance of the model on test dataset
    """
    task_test_losses = []

    f = open(best_result + filename, 'w+')

    if task == "vanishing_point":
        criterion = torch.nn.MSELoss().to(DEVICE)
        model = SegNet(3, 3).to(DEVICE)
        model.load_state_dict(torch.load(best_model))
    
    elif task == "segmentation":
        model = SegNet(3, 14).to(DEVICE)
        criterion = torch.nn.CrossEntropyLoss().to(DEVICE)
        model.load_state_dict(torch.load(best_model))
        test_cm = ConfMatrix(14)

    elif task == "surface_normal":
        criterion = torch.nn.MSELoss().to(DEVICE)
        model = SegNet(3, 3)
        model.load_state_dict(torch.load(best_model, map_location='cpu'))
        model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        for image, label in test_loader:

            image = image.to(DEVICE)
            label = label.to(DEVICE)
            
            output = model(task, False, image)
    
            loss = criterion(output, label)
            task_test_losses.append(loss.item())

            if task == "segmentation":
        
                pred_values, predicted = torch.max(output, 1)
                test_cm.update(predicted, label)
    
    test_loss = np.average(task_test_losses)

    if task == "segmentation":
        test_iou, test_accuracy = test_cm.get_metrics() 
        info = f"Segmentation Test Loss: {test_loss.item():.3f}.. "\
            +f"Segmentation mIoU: {test_iou.item():0.3f}.."\
            +f"Segmentation Pixel Accuracy: {test_accuracy.item():0.3f}"

    else:
        info = f"{task} Test Loss: {test_loss:.3f} "

    print(info)
    f.write(info)
    f.close()
    return
    

def multitask_inference(task, test_loader, best_model, best_result, DEVICE, fname):
    """
    returns performance of the model on test dataset
    """
    task_test_losses = []
    depth_test_losses = []
    total_test_losses = []

    f = open(best_result + fname, 'w+')

    if task == "vanishing_point_depth":
        criterion_task = torch.nn.MSELoss().to(DEVICE)
        criterion_depth = torch.nn.MSELoss().to(DEVICE)
        model = SegNet(3, 3).to(DEVICE)
        model.load_state_dict(torch.load(best_model))
    
    elif task == "segmentation_depth":
        model = SegNet(3, 14).to(DEVICE)
        criterion_task = torch.nn.CrossEntropyLoss().to(DEVICE)
        criterion_depth = torch.nn.MSELoss().to(DEVICE)
        model.load_state_dict(torch.load(best_model))
        test_cm = ConfMatrix(14)

    elif task == "surface_normal_depth":
        criterion_task = torch.nn.MSELoss().to(DEVICE)
        criterion_depth = torch.nn.MSELoss().to(DEVICE)
        model = SegNet(3, 3).to(DEVICE)
        model.load_state_dict(torch.load(best_model))

    model.eval()

    with torch.no_grad():
        for image, label, depth in test_loader:
     
            image = image.to(DEVICE)
            label = label.to(DEVICE)
            depth = depth.to(DEVICE)

            task_op, depth_op = model(task, True, image)
            depth_op = torch.squeeze(depth_op,1)

            task_loss = criterion_task(task_op, label)
            depth_loss = criterion_depth(depth_op, depth)
            loss = torch.mul(0.75, task_loss) + torch.mul(0.25, depth_loss)
            
            task_test_losses.append(task_loss.item())
            depth_test_losses.append(depth_loss.item())
            total_test_losses.append(loss.item())
            
            if task == "segmentation_depth":
                pred_values, predicted = torch.max(task_op, 1)
                test_cm.update(predicted, label)

    task_loss = np.average(task_test_losses)
    depth_loss = np.average(depth_test_losses)
    total_loss = np.average(total_test_losses)

    if task == "segmentation_depth":
        test_iou, test_accuracy = test_cm.get_metrics() 
        info = f"Total Test loss: {total_loss:.3f} "\
                + f"Depth Test loss: {depth_loss:.3f} "\
                + f"Segmentation Test loss: {task_loss:.3f} "\
                + f"Test mIoU: {test_iou:.3f} "\
                + f"Pixel accuracy: {test_accuracy:.3f} \n"
    else:
        info = f"Total Test loss: {total_loss:.3f} "\
                + f"Depth Test loss: {depth_loss:.3f} "\
                + f"{task} Test loss: {task_loss:.3f} "

   
    print(info)

    f.write(info)
    f.close()
    return
    
    
def multitask_train_loop(task, model, tloader, vloader, criterion, criterion_dep, optimizer, DEVICE):
    """
    returns loss and other metrics.
    params: model -  vgg16/SegNet
          tloader - train dataset
          vloader - val dataset
          criterion - loss function
          optimizer - Adam optimizer
          task - segmentation/surface_normal/vanishing_point
    """
    
    train_losses = []
    valid_losses = []
    train_accuracy, val_accuracy, t_epoch_iou, v_epoch_iou = None, None, None, None 
    ind = 0
    model.train()
    model.to(DEVICE)
    cnt = 1

    task_train_losses = []
    task_valid_losses = []
    dep_train_losses = []
    dep_valid_losses = []
    multiloss = {}
    train_cm = ConfMatrix(14)
    test_cm = ConfMatrix(14)
 
    for image, label, depth in tloader:
        
        image = image.to(DEVICE)
        label = label.to(DEVICE)
        depth = depth.to(DEVICE)

        optimizer.zero_grad()

        task_op, depth_op = model(task, True, image)
        
        depth_op = torch.squeeze(depth_op,1)
        
        task_loss = criterion(task_op, label)
        depth_loss = criterion_dep(depth_op, depth)

        task_train_losses.append(task_loss.item())
        dep_train_losses.append(depth_loss.item())

        loss = torch.mul(0.75, task_loss) + torch.mul(0.25, depth_loss)

        train_losses.append(loss.item())
        loss.backward()
        optimizer.step()

        if task == "segmentation_depth":
            pred_values, predicted = torch.max(task_op, 1)
            train_cm.update(predicted, label)
       
    if task == "segmentation_depth":
        t_epoch_iou, train_accuracy = train_cm.get_metrics()    
    
    t_epoch_loss = np.average(train_losses)

    model.eval()
    
    with torch.no_grad():
        for image, label, depth in vloader:
     
            image = image.to(DEVICE)
            label = label.to(DEVICE)
            depth = depth.to(DEVICE)

            task_op, depth_op = model(task, True, image)
            depth_op = torch.squeeze(depth_op,1)

            task_loss = criterion(task_op, label)
            depth_loss = criterion_dep(depth_op, depth)
            loss = torch.mul(0.75, task_loss) + torch.mul(0.25, depth_loss)
            
            task_valid_losses.append(task_loss.item())
            dep_valid_losses.append(depth_loss.item())

            valid_losses.append(loss.item())
  
            if task == "segmentation_depth":
        
                pred_values, predicted = torch.max(task_op, 1)
                test_cm.update(predicted, label)
    
    v_epoch_loss = np.average(valid_losses)
    
    multiloss['task_train'] = np.average(task_train_losses)
    multiloss['task_val'] = np.average(task_valid_losses)
    multiloss['dep_train'] = np.average(dep_train_losses)
    multiloss['dep_val'] = np.average(dep_valid_losses)

    if task == "segmentation_depth":
        v_epoch_iou, val_accuracy = test_cm.get_metrics() 
    
    return model, t_epoch_loss, v_epoch_loss, train_accuracy, val_accuracy, t_epoch_iou, v_epoch_iou, multiloss


def train_loop(task, model, tloader, vloader, criterion, optimizer, DEVICE):
    """
    returns loss and other metrics.
    params: model -  vgg16/SegNet
          tloader - train dataset
          vloader - val dataset
          criterion - loss function
          optimizer - Adam optimizer
          task - segmentation/surface_normal/vanishing_point
    """
    
    train_losses = []
    valid_losses = []
    train_accuracy, val_accuracy, t_epoch_iou, v_epoch_iou = None, None, None, None 
    model.train()
    model.to(DEVICE)
    train_cm = ConfMatrix(14)
    test_cm = ConfMatrix(14)
 
    for image, label in tloader:
        
        image = image.to(DEVICE)
        label = label.to(DEVICE)

        optimizer.zero_grad()

        output = model(task, False, image)

        if task=='depth':
            output = torch.squeeze(output, 1)
        loss = criterion(output, label)
       
        train_losses.append(loss.item())
        loss.backward()
        optimizer.step()

        if task == "segmentation":
            pred_values, predicted = torch.max(output, 1)
            train_cm.update(predicted, label)
       
    if task == "segmentation":
        t_epoch_iou, train_accuracy = train_cm.get_metrics()    
    
    t_epoch_loss = np.average(train_losses)

    
    model.eval()
    
    with torch.no_grad():
        for image, label in vloader:
        
            image = image.to(DEVICE)
            label = label.to(DEVICE)
            
            output = model(task, False, image)
            
            if task=="depth":
                output = torch.squeeze(output,1)
            loss = criterion(output, label)
            valid_losses.append(loss.item())
            
            
            if task == "segmentation":
        
                pred_values, predicted = torch.max(output, 1)
                test_cm.update(predicted, label)
    
    v_epoch_loss = np.average(valid_losses)
    
    if task == "segmentation":
        v_epoch_iou, val_accuracy = test_cm.get_metrics() 
    
    return model, t_epoch_loss, v_epoch_loss, train_accuracy, val_accuracy, t_epoch_iou, v_epoch_iou



def train_model(task, trainloader, valloader, timestr, results_record, DEVICE, LR, threshold=30):
    """
    returns losses (train and val), accuracies (train and val), trained_model
    params: trainloader - train dataset
            valloader - validation dataset
            task - segmentation/surface_normal/vanishing_point
    """
    os.makedirs(CHECKPOINT_DIR + task +"/"+timestr)
    flag = False
    multitask = False
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    results_record.write("learning rate: "+str(LR))
    if task == "segmentation":
        model = SegNet(3, 14).to(DEVICE)
        # model = nn.DataParallel(model)
        criterion = torch.nn.CrossEntropyLoss().to(DEVICE)
        early_stop = EarlyStopping(patience=10, task='segmentation', timestamp=timestr)
        
    elif task == "depth":
        model = SegNet(3, 1).to(DEVICE)
        criterion = torch.nn.MSELoss().to(DEVICE)
        early_stop = EarlyStopping(patience=7, task='depth', timestamp=timestr)

    elif task == "segmentation_depth":
        multitask = True
        model = SegNet(3, 14).to(DEVICE)
        # model = nn.DataParallel(model)
        criterion = torch.nn.CrossEntropyLoss().to(DEVICE)
        criterion_dep = torch.nn.MSELoss().to(DEVICE)
        early_stop = EarlyStopping(patience=8, task='segmentation_depth', timestamp=timestr)
    
    elif task == "surface_normal_depth":
        multitask = True
        model = SegNet(3, 3).to(DEVICE)
        criterion = torch.nn.MSELoss().to(DEVICE)
        criterion_dep = torch.nn.MSELoss().to(DEVICE)
        early_stop = EarlyStopping(patience=8, task='surface_normal_depth', timestamp=timestr)

    elif task == "vanishing_point_depth":
        multitask = True
        model = SegNet(3, 3).to(DEVICE)
        model = nn.DataParallel(model)
        criterion = torch.nn.MSELoss().to(DEVICE)
        criterion_dep = torch.nn.MSELoss().to(DEVICE)
        early_stop = EarlyStopping(patience=10, task='vanishing_point_depth', timestamp=timestr)
        
    elif task == "vanishing_point":
        
        model = SegNet(3, 3).to(DEVICE)
        # model = nn.DataParallel(model)
        criterion = nn.MSELoss().to(DEVICE)
        early_stop = EarlyStopping(patience=15, task='vanishing_point',  timestamp=timestr)
        
    elif task == "surface_normal":
        model = SegNet(3, 3).to(DEVICE)
        criterion = torch.nn.MSELoss().to(DEVICE)
        early_stop = EarlyStopping(patience=10, task='surface_normal', timestamp=timestr)
    
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, threshold, gamma=0.1, last_epoch=-1, verbose=True)

    for epoch in range(EPOCHS):
        
        start = time.time()
        if multitask:
            model, epoch_train_loss,  epoch_val_loss, train_ac, val_ac, train_iou, val_iou, multiloss = multitask_train_loop(task, model, trainloader, valloader, criterion, criterion_dep, optimizer, DEVICE)
        else:
            model, epoch_train_loss,  epoch_val_loss, train_ac, val_ac, train_iou, val_iou = train_loop(task, model, trainloader, valloader, criterion, optimizer, DEVICE)

        train_loss.append(epoch_train_loss)   
        val_loss.append(epoch_val_loss)

        print("\nTime required: ",(time.time()-start)/60.)
        if task == "segmentation":
            train_acc.append(train_ac)
            val_acc.append(val_ac)
            info = f"Epoch {epoch+1}/{EPOCHS} \n"\
                + f"Train loss: {epoch_train_loss:.3f} "\
                + f"Train mIoU: {train_iou:.3f} "\
                + f"Train accuracy: {train_ac:.3f} \n"\
                + f"Val loss: {epoch_val_loss:.3f} "\
                + f"Val mIoU: {val_iou:.3f} "\
                + f"Val accuracy: {val_ac:.3f} \n"

            results_record.write(info)
            print(info)
            print("----------------------------------------------------------------------")
       
        elif task =='segmentation_depth':

            train_acc.append(train_ac)
            val_acc.append(val_ac)

            info = f"Epoch {epoch+1}/{EPOCHS} \n"\
                + f"Total Train loss: {epoch_train_loss:.3f} "\
                + f"Depth Train loss: {multiloss['dep_train']:.3f} "\
                + f"Seg Train loss: {multiloss['task_train']:.3f} "\
                + f"Train mIoU: {train_iou:.3f} "\
                + f"Train accuracy: {train_ac:.3f} \n"\
                + f"Total Val loss: {epoch_val_loss:.3f} "\
                + f"Depth Val loss: {multiloss['dep_val']:.3f} "\
                + f"Seg Val loss: {multiloss['task_val']:.3f} "\
                + f"Val mIoU: {val_iou:.3f} "\
                + f"Val accuracy: {val_ac:.3f} \n"

            results_record.write(info)
            print(info)
            print("----------------------------------------------------------------------")
        
        elif task =='surface_normal_depth':

            info = f"Epoch {epoch+1}/{EPOCHS} \n"\
                + f"Total Train loss: {epoch_train_loss:.3f} "\
                + f"Depth Train loss: {multiloss['dep_train']:.3f} "\
                + f"SN Train loss: {multiloss['task_train']:.3f} "\
                + f"Total Val loss: {epoch_val_loss:.3f} "\
                + f"Depth Val loss: {multiloss['dep_val']:.3f} "\
                + f"SN Val loss: {multiloss['task_val']:.3f} \n"

            results_record.write(info)
            print(info)
            print("----------------------------------------------------------------------")
        
        elif task =='vanishing_point_depth':

            info = f"Epoch {epoch+1}/{EPOCHS} \n"\
                + f"Total Train loss: {epoch_train_loss:.3f} "\
                + f"Depth Train loss: {multiloss['dep_train']:.3f} "\
                + f"VP Train loss: {multiloss['task_train']:.3f} "\
                + f"Total Val loss: {epoch_val_loss:.3f} "\
                + f"Depth Val loss: {multiloss['dep_val']:.3f} "\
                + f"VP Val loss: {multiloss['task_val']:.3f} \n"

            results_record.write(info)
            print(info)
            print("----------------------------------------------------------------------")

        else:
            info = f"Epoch {epoch+1}/{EPOCHS}.. "\
                + f"Train loss: {epoch_train_loss:.3f}.. "\
                + f"Val loss: {epoch_val_loss:.3f} \n"

            results_record.write(info)
            print(info)
            print("----------------------------------------------------------------------")
        
        scheduler.step()
        early_stop(epoch_val_loss, model)
    
        if early_stop.early_stop:
            print("Early stopping")
            flag = True
            break 

        if (epoch+1)%25 == 0:
            torch.save(model.state_dict(), CHECKPOINT_DIR + task +"/"+timestr+"/model_epoch_" + str(epoch+1) + ".pth")

    print("Training completed!")
    
    losses = [train_loss, val_loss]
    accuracies = [train_acc, val_acc]
        
    return losses, accuracies, model, flag



def retrain_depth(task, trainloader, valloader, timestr, results_record, DEVICE, LR, threshold=30):

    if task == "segmentation_retrain":
        
        model = SegNet(3, 14).to(DEVICE)
        # model = nn.DataParallel(model)
        criterion_dep = torch.nn.MSELoss().to(DEVICE)
        early_stop = EarlyStopping(patience=10, task=task, timestamp=timestr)
        # YET TO DO PLEASE IGNORE
