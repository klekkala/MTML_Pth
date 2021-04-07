import torch
import numpy as np
import os
import torchvision
import time
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
from utils import load_sunrgbd_dataset
from utils import EarlyStopping
from networks import SegNet
from utils import get_data_loader
from utils import ConfMatrix
from networks import VGG16Model
from tempfile import TemporaryFile
from numpy import savez_compressed
LR = 1e-3
DEVICE =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 500
CHECKPOINT_DIR = '/home4/shubham/MTML_Pth/checkpoints/'


def run_inference(task, model, testloader):
    """
    returns performance of the model on test dataset
    """
    valid_losses = []
    model.eval()
    
    if task == "segmentation":
        criterion = torch.nn.CrossEntropyLoss().to(DEVICE)
        test_cm = ConfMatrix(14)
        
    elif task == "vanishing_point" or task == "surface_normal" or task == "depth":
        criterion = nn.MSELoss().to(DEVICE)

    with torch.no_grad():
        for image, label in testloader:
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
        iou, accuracy = test_cm.get_metrics()
        print("Mean Pixel IoU: ",iou.item())
        print("Accuracy: ",accuracy.item())  
    
    print("Final loss on test images: ",v_epoch_loss)    
    
    

def train_loop(task, model, tloader, vloader, criterion, optimizer):
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
    train_cm = ConfMatrix(14)
    test_cm = ConfMatrix(14)
 
    for image, label in tloader:
        
        image = image.to(DEVICE)
        label = label.to(DEVICE)
        
        
        optimizer.zero_grad()

        output = model(image)
  
        if task == "depth":
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
            cnt += 1
            image = image.to(DEVICE)
            label = label.to(DEVICE)
            
            output = model(image)
            if task=="depth":
                output = torch.squeeze(output,1)
            loss = criterion(output, label)
            valid_losses.append(loss.item())
            
#             if task == "depth" and cnt==20:
                
#                 fig = plt.figure(figsize=(11,11))
#                 ax = plt.subplot(1, 3, 1)
#                 plt.imshow(image[0].cpu().numpy().transpose((1, 2, 0)))
#                 ax = plt.subplot(1, 3, 2)
#                 plt.imshow((label[0].cpu().numpy().transpose((1, 2, 0))*255).astype(np.uint8))
#                 ax = plt.subplot(1, 3, 3)
#                 plt.imshow((output[0].cpu().numpy().transpose((1, 2, 0))*255).astype(np.uint8))
#                 plt.show()
#                 fig2 = plt.figure(figsize=(11,11))
#                 ax = plt.subplot(1, 3, 1)
#                 plt.imshow(image[1].cpu().numpy().transpose((1, 2, 0)))
#                 ax = plt.subplot(1, 3, 2)
#                 plt.imshow((label[1].cpu().numpy().transpose((1, 2, 0))*255).astype(np.uint8))
#                 ax = plt.subplot(1, 3, 3)
#                 plt.imshow(output[1].cpu().numpy().transpose((1, 2, 0)))
#                 plt.show()
            
            if task == "segmentation":
        
                pred_values, predicted = torch.max(output, 1)
                test_cm.update(predicted, label)
#                 if ind == 0:
#                     fig = plt.figure(figsize=(11,11))
#                     ax = plt.subplot(1, 3, 1)
#                     plt.imshow(image[0].cpu().numpy().transpose((1, 2, 0)))
#                     ax = plt.subplot(1, 3, 2)
#                     plt.imshow(torch.squeeze(label[0].cpu(),0))
#                     ax = plt.subplot(1, 3, 3)
#                     plt.imshow(predicted[0].cpu())
#                     plt.show()
#                     fig2 = plt.figure(figsize=(11,11))
#                     ax = plt.subplot(1, 3, 1)
#                     plt.imshow(image[1].cpu().numpy().transpose((1, 2, 0)))
#                     ax = plt.subplot(1, 3, 2)
#                     plt.imshow(torch.squeeze(label[1].cpu(),0))
#                     ax = plt.subplot(1, 3, 3) 
#                     plt.imshow(predicted[1].cpu())
#                     plt.show()
#                     ind+=1
    
    v_epoch_loss = np.average(valid_losses)
    
    if task == "segmentation":
        v_epoch_iou, val_accuracy = test_cm.get_metrics() 
    
    return model, t_epoch_loss, v_epoch_loss, train_accuracy, val_accuracy, t_epoch_iou, v_epoch_iou



def train_model(task, trainloader, valloader, timestr):
    """
    returns losses (train and val), accuracies (train and val), trained_model
    params: trainloader - train dataset
            valloader - validation dataset
            task - segmentation/surface_normal/vanishing_point
    """
    os.makedirs(CHECKPOINT_DIR + task +"/"+timestr)
    flag = False
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    
    if task == "segmentation":
        model = SegNet(3, 14).to(DEVICE)
        criterion = torch.nn.CrossEntropyLoss().to(DEVICE)
        early_stop = EarlyStopping(patience=4, task='segmentation', timestamp=timestr)
    
    elif task == "depth":
        model = SegNet(3, 1).to(DEVICE)
        criterion = torch.nn.MSELoss().to(DEVICE)
        
    elif task == "vanishing_point":
        model = VGG16Model().to(DEVICE)
        criterion = nn.MSELoss().to(DEVICE)
        early_stop = EarlyStopping(patience=4, task='vanishing_point',  timestamp=timestr)
        
    elif task == "surface_normal":
        model = SegNet(3, 3).to(DEVICE)
        criterion = torch.nn.MSELoss().to(DEVICE)
        early_stop = EarlyStopping(patience=4, task='surface_normal', timestamp=timestr)
    
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    
    for epoch in range(EPOCHS):
        print("Running Epoch {}".format(epoch+1))
        
        start = time.time()
        model, epoch_train_loss,  epoch_val_loss, train_ac, val_ac, train_iou, val_iou = train_loop(task, model, trainloader, valloader, criterion, optimizer)
        
        train_loss.append(epoch_train_loss)   
        val_loss.append(epoch_val_loss)
        print("Time required: ",(time.time()-start)/60.)
        if task == "segmentation":
            train_acc.append(train_ac)
            val_acc.append(val_ac)

            print("Training loss: {0:.4f}   Training accuracy: {1:.4f}   Training mIoU: {2:.4f}".format(epoch_train_loss, train_ac, train_iou))
            print("Validation loss: {0:.4f} Validation accuracy: {1:.4f} Validation mIoU: {2:.4f}".format(epoch_val_loss, val_ac, val_iou))
            print("--------------------------------------------------------")
       
        elif task == "vanishing_point":
            
            print("Training loss: {0:.4f}  Testing loss: {1:0.4f}".format(epoch_train_loss, epoch_val_loss))
            print("--------------------------------------------------------")
        
        elif task == "surface_normal":
            
            print("Training loss: {0:.4f}  Testing loss: {1:0.4f}".format(epoch_train_loss, epoch_val_loss))
            print("--------------------------------------------------------") 
        
        elif task == "depth":
            
            print("Training loss: {0:.4f}  Testing loss: {1:0.4f}".format(epoch_train_loss, epoch_val_loss))
            print("--------------------------------------------------------") 
        
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







