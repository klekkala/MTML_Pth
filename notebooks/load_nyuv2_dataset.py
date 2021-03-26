import glob
import numpy as np

TRAIN_RGB_PATH = "/home4/shubham/MTML_Pth/datasets/nyuv2/train_rgb/"
TRAIN_SEG_PATH = "/home4/shubham/MTML_Pth/datasets/nyuv2/train_seg13/"

TEST_RGB_PATH = "/home4/shubham/MTML_Pth/datasets/nyuv2/test_rgb/"
TEST_SEG_PATH = "/home4/shubham/MTML_Pth/datasets/nyuv2/test_seg13/"

TRAIN_SN_PATH = "/home4/shubham/MTML_Pth/datasets/nyuv2/train_sn/"
TEST_SN_PATH = "/home4/shubham/MTML_Pth/datasets/nyuv2/test_sn/"

def load_dataset(flag):
    """
    returns dictionary of images and their corresponding annotations split into train, val and test
    :params: flag - (task) segmentation, depth or surface normal
    """ 
    data = {}
    label = {}
    
    TRAIN_PATH_IMG = None
    TRAIN_PATH_LAB = None
    TEST_PATH_IMG = None
    TEST_PATH_LAB = None
    
    if flag == "segmentation":
        TRAIN_PATH_IMG =  TRAIN_RGB_PATH
        TRAIN_PATH_LAB = TRAIN_SEG_PATH
        TEST_PATH_IMG = TEST_RGB_PATH
        TEST_PATH_LAB = TEST_SEG_PATH
        
    elif flag == "normal":
        TRAIN_PATH_IMG =  TRAIN_RGB_PATH
        TRAIN_PATH_LAB = TRAIN_SN_PATH
        TEST_PATH_IMG = TEST_RGB_PATH
        TEST_PATH_LAB = TEST_SN_PATH
        
    train_images = glob.glob(TRAIN_PATH_IMG + "*.png")
    train_labels = glob.glob(TRAIN_PATH_LAB + "*.png")
    
    index = np.random.permutation(len(train_images))
    images = np.array(train_images)[index]
    labels = np.array(train_labels)[index]
    
    length = int(len(images)*0.85)
   
    data["train"], data["val"] = images[:length], images[length:]
    label["train"], label["val"] = labels[:length], labels[length:]
    data["test"] = glob.glob(TEST_PATH_IMG + "*.png")
    label["test"] = glob.glob(TEST_PATH_LAB + "*.png")
    
    return data, label