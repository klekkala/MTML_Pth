import torch
import numpy as np
import os
import time
from models import Segmentation, Surface_Normal, VP, Task_Depth
from utils import EarlyStopping
from utils import ConfMatrix

EPOCHS = 200

REL = os.getcwd()
CHECKPOINT_DIR = REL + '/checkpoints/' 


def task_inference(task, test_loader, best_model, best_result, DEVICE, filename):
    """
    returns performance of the model on test dataset
    """
    task_test_losses = []

    f = open(best_result + filename, 'w+')

    if task == "vanishing_point":
        criterion = torch.nn.MSELoss().to(DEVICE)
        model = VP(input_channels=3)
        model.load_state_dict(torch.load(best_model))

    elif task == "segmentation":
        model = Segmentation(input_channels=3, output_channels=14, task="seg")
        criterion = torch.nn.CrossEntropyLoss().to(DEVICE)
        model.load_state_dict(torch.load(best_model))
        test_cm = ConfMatrix(14)

    elif task == "surface_normal":
        criterion = torch.nn.MSELoss().to(DEVICE)
        model = Surfac_Normal(input_channels=3, output_channels=3, task="sn")
        model.load_state_dict(torch.load(best_model, map_location='cpu'))

    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        for image, label in test_loader:

            image = image.to(DEVICE)
            label = label.to(DEVICE)
            
            output = model(image)
    
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

        output = model(image)

        if task=="depth":
            output = torch.squeeze(output,1)

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
            
            output = model(image)
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


    IMPORTANT INSTRUCTION:

    PARAMETERS TO THE OBJECTS OF THE DERIVED CLASSES i.e. Segmentation, Surface_Normal, Segmentation_Depth, Surface_Normal_Depth
    should be:

    model =  {Model_name}(input_channels=x, output_channels=y, task="z")

        where,
                Model_nane = Segmentation, Surface_Normal, Segmentation_Depth, Surface_Normal_Depth
                x = 3 for all tasks
                y = 1 for depth, 14 for segmentation, 3 for Surface Normal
                z = "seg" for Segmentation and Segmentation Depth, "sn" for Surface Normal and Surface Normal Depth

    For VP and VP_DEPTH:

    model = {Model_name}(input_channels=x)

    """


    os.makedirs(CHECKPOINT_DIR + task +"/"+timestr)
    early_stop_ = False
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    results_record.write("learning rate: "+str(LR))

    if task == "segmentation":
        key = "seg"
        model = Segmentation(input_channels=3, output_channels=14, task=key).to(DEVICE)
        criterion = torch.nn.CrossEntropyLoss().to(DEVICE)
        early_stop = EarlyStopping(patience=10, task=task, timestamp=timestr)
     
    elif task == "vanishing_point":  
        key = "vp"
        model = VP(input_channels=3).to(DEVICE)
        criterion = torch.nn.MSELoss().to(DEVICE)
        early_stop = EarlyStopping(patience=10, task=task,  timestamp=timestr)
        
    elif task == "surface_normal":
        key = "sn"
        model = Surface_Normal(input_channels=3, output_channels=3, task="sn").to(DEVICE)
        criterion = torch.nn.MSELoss().to(DEVICE)
        early_stop = EarlyStopping(patience=10, task=task, timestamp=timestr)
       
    elif task == "depth":
        key = "dep"
        model = Task_Depth(input_channels=3, output_channels=1, task=key)
        criterion = torch.nn.MSELoss().to(DEVICE)
        early_stop = EarlyStopping(patience=10, task=task,  timestamp=timestr)

    results_record.write("\n\nModel Architecture:\n"+str(model))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, threshold, gamma=0.1, last_epoch=-1, verbose=True)

    for epoch in range(EPOCHS):
        
        start = time.time()
        
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

        else:
            info = f"Epoch {epoch+1}/{EPOCHS}.. "\
                + f"Train loss: {epoch_train_loss:.3f}.. "\
                + f"Val loss: {epoch_val_loss:.3f} \n"


        results_record.write(info)
        print(info)
        print("----------------------------------------------------------------------")
        scheduler.step()
        early_stop(epoch_val_loss, model, key)
    
        if early_stop.early_stop:
            print("Early stopping")
            early_stop_ = True
            break 

        if (epoch+1)%20 == 0:
            torch.save(model.state_dict(), CHECKPOINT_DIR + task +"/"+timestr+"/model_epoch_" + str(epoch+1) + ".pth")

    print("Training completed!")
    
    losses = [train_loss, val_loss]
    accuracies = [train_acc, val_acc]
        
    return losses, accuracies, model, early_stop_