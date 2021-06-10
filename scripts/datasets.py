import glob, os, re
import numpy as np

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# PATHS
REL = os.getcwd()

# DIODE
SN_TRAIN = REL + '/datasets/diode_dataset/train/'
SN_VAL = REL + '/datasets/diode_dataset/val/'

# NYUv2
TRAIN_NYU_RGB_PATH = REL + "/datasets/nyuv2/train_rgb/"
TRAIN_NYU_SEG_PATH = REL + "/datasets/nyuv2/train_seg13/"
TEST_NYU_RGB_PATH = REL + "/datasets/nyuv2/test_rgb/"
TEST_NYU_SEG_PATH = REL + "/datasets/nyuv2/test_seg13/"
TRAIN_NYU_SN_PATH = REL + "/datasets/nyuv2/train_sn/"
TEST_NYU_SN_PATH = REL + "/datasets/nyuv2/test_sn/"

# Scannet VP
TRAIN_VP = REL + '/datasets/scannet_vp/train/'
VAL_VP = REL + '/datasets/scannet_vp/val/'
TEST_VP = REL + '/datasets/scannet_vp/test/'

# SUNRGBD
SUNRGBD_TRAIN_IMAGES = REL + '/datasets/sunrgbd_dataset/sunrgbd_images_train_final.txt'
SUNRGBD_TEST_IMAGES = REL + '/datasets/sunrgbd_dataset/sunrgbd_images_test_final.txt'
SUNRGBD_TRAIN_DEPTH = REL + '/datasets/sunrgbd_dataset/sunrgbd_depth_train_final.txt'
SUNRGBD_TEST_DEPTH = REL + '/datasets/sunrgbd_dataset/sunrgbd_depth_test_final.txt'
SUNRGBD_TRAIN_SEG = REL + '/datasets/sunrgbd_dataset/sunrgbd_seg_train_final.txt'
SUNRGBD_TEST_SEG = REL + '/datasets/sunrgbd_dataset/sunrgbd_seg_test_final.txt'

# SUNRGBD NYUv2
NYU_TEST_IMAGES = REL + '/datasets/sun_nyu2/nyu_test_images.txt'
NYU_TEST_DEPTH = REL + '/datasets/sun_nyu2/nyu_test_depth.txt'
NYU_TEST_SEG = REL + '/datasets/sun_nyu2/nyu_test_seg.txt'

NYU_TRAIN_IMAGES = REL + '/datasets/sun_nyu2/nyu_train_images.txt'
NYU_TRAIN_DEPTH = REL + '/datasets/sun_nyu2/nyu_train_depth.txt'
NYU_TRAIN_SEG = REL + '/datasets/sun_nyu2/nyu_train_seg.txt'

# SN NYUv2
NYU_TRAIN_SN_RGB = REL + '/datasets/nyuv2/train_rgb/'
NYU_TRAIN_SN_DEPTH = REL + '/datasets/nyuv2/train_depth/'
NYU_TRAIN_SN = REL + '/datasets/nyuv2/train_sn/'

NYU_TEST_SN_RGB = REL + '/datasets/nyuv2/test_rgb/'
NYU_TEST_SN_DEPTH = REL + '/datasets/nyuv2/test_depth/'
NYU_TEST_SN = REL + '/datasets/nyuv2/test_sn/'

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Load datasets

# 1. Diode Dataset for Surface Normal

def load_diode_sn_dataset():
    """
    returns train and test images with their corresponding normals as labels
    """
    
    data = {"train": [], "val":[]}
    label = {"train": [], "val":[]}
    depth = {"train": [], "val":[]}
   
    data['train'] = glob.glob(SN_TRAIN + '*.png')
    for img in data['train']:
        base = img.split('/')[-1][:-4]
        norm = base + '_normal.npy'
        dep = base + '_depth.npy'
        label['train'].append(SN_TRAIN + norm)
        depth['train'].append(SN_TRAIN + dep)


    data['val'] = glob.glob(SN_VAL + '*.png')
    for img in data['val']:
        base = img.split('/')[-1][:-4]
        norm = base + '_normal.npy'
        dep = base + '_depth.npy'
        label['val'].append(SN_VAL + norm)
        depth['val'].append(SN_VAL + dep)
    
    print("Size of train images : ", len(data['train']))
    print("Size of val images : ", len(data['val']))
    
    print("Size of train depth : ", len(depth['train']))
    print("Size of val depth : ", len(depth['val']))
    
    print("Size of train Surface normal : ", len(label['train']))
    print("Size of val Surface normal : ", len(label['val']))
    
    return data, label, depth



# 2. Load nyuv2 dataset for segmentation and surface normal

def load_nyuv2_dataset(flag):
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
        TRAIN_PATH_IMG =  TRAIN_NYU_RGB_PATH
        TRAIN_PATH_LAB = TRAIN_NYU_SEG_PATH
        TEST_PATH_IMG = TEST_NYU_RGB_PATH
        TEST_PATH_LAB = TEST_NYU_SEG_PATH
        
    elif flag == "surface_normal":
        TRAIN_PATH_IMG =  TRAIN_NYU_RGB_PATH
        TRAIN_PATH_LAB = TRAIN_NYU_SN_PATH
        TEST_PATH_IMG = TEST_NYU_RGB_PATH
        TEST_PATH_LAB = TEST_NYU_SN_PATH
        
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


# 3. Load scannet vanishing points dataset


def load_scannet_vp_dataset():
    """
    returns training and validation images and their corresponding vanishing points labels.
    """

    data_image = {'train':[], 'test':[], 'val': []}
    vanish_point = {'train':[], 'test':[], 'val': []}
    vanish_depth = {'train':[], 'test':[], 'val': []}

    def sort_images(temp):
        sortedimages = sorted(temp, key=lambda x: int(re.findall(r'\d+', x)[-1]))
        return sortedimages

    def get_vp_dataset(path, dtype):

        folders = os.listdir(path)
        main_images = []
        vanishing_points = []
        vanishing_depth = []
        
        for scene in folders:
            images = glob.glob(path + scene + '/*color.png')
            vanish = glob.glob(path + scene + '/*vanish.npz')
            depth = glob.glob(path + scene + '/[!frame]*.png')
            
            main_images.extend(sort_images(images))
            vanishing_points.extend(sort_images(vanish))
            vanishing_depth.extend(sort_images(depth))
            

        data_image[dtype] = main_images
        vanish_point[dtype] = vanishing_points
        vanish_depth[dtype] = vanishing_depth

        return

    get_vp_dataset(TRAIN_VP, 'train')
    get_vp_dataset(VAL_VP, 'val')
    get_vp_dataset(TEST_VP, 'test')   


    print("Length of train data: ",len(data_image['train']))
    print("Length of val data: ",len(data_image['val']))
    print("Length of test data: ",len(data_image['test']))

    print("Length of train vp: ",len(vanish_point['train']))
    print("Length of val vp: ",len(vanish_point['val']))
    print("Length of test vp: ",len(vanish_point['test']))

    print("Length of train depth: ",len(vanish_depth['train']))
    print("Length of val depth: ",len(vanish_depth['val']))
    print("Length of test depth: ",len(vanish_depth['test']))

    return data_image, vanish_point, vanish_depth


# 4. Load SUNRGBD dataset

def load_sunrgbd_dataset():
    """
    loads sunrgbd dataset rgb images with their corresponding segmentation and depth images.
    """

    images = {'train':[], 'val':[]}
    depth = {'train':[], 'val':[]}
    seg = {'train':[], 'val':[]} 

    with open(SUNRGBD_TRAIN_IMAGES, 'r') as f:
        for line in f:
            images['train'].append(os.path.join(REL, line.strip('\n')))

    with open(SUNRGBD_TEST_IMAGES, 'r') as f:
        for line in f:
            images['val'].append(os.path.join(REL, line.strip('\n')))

    with open(SUNRGBD_TRAIN_DEPTH, 'r') as f:
        for line in f:
            depth['train'].append(os.path.join(REL, line.strip('\n')))

    with open(SUNRGBD_TEST_DEPTH, 'r') as f:
        for line in f:
            depth['val'].append(os.path.join(REL, line.strip('\n')))

    with open(SUNRGBD_TRAIN_SEG, 'r') as f:
        for line in f:
            seg['train'].append(os.path.join(REL, line.strip('\n')))

    with open(SUNRGBD_TEST_SEG, 'r') as f:
        for line in f:
            seg['val'].append(os.path.join(REL, line.strip('\n')))

    print("\nImages: ")
    print("Size of training data : ", len(images['train']))
    print("Size of validation data : ", len(images['val']))
    print("\nDepth: ")
    print("Size of training data : ", len(depth['train']))
    print("Size of validation data : ", len(depth['val']))
    print("\nSegmentation: ")
    print("Size of training data : ", len(seg['train']))
    print("Size of validation data : ", len(seg['val']), "\n")


    return images, seg, depth



# load SN NYUv2

def load_sn_nyuv2_dataset():
    """
    loads sunrgbd dataset rgb images with their corresponding segmentation and depth images.
    """

    images = {'train':[], 'test':[]}
    depth = {'train':[], 'test':[]}
    sn = {'train':[], 'test':[]} 

    images['train'] = glob.glob(NYU_TRAIN_SN_RGB + '*.png')
    depth['train'] = glob.glob(NYU_TRAIN_SN_DEPTH + '*.png')
    sn['train'] = glob.glob(NYU_TRAIN_SN + '*.png')
    
    images['test'] = glob.glob(NYU_TEST_SN_RGB + '*.png')
    depth['test'] = glob.glob(NYU_TEST_SN_DEPTH + '*.png')
    sn['test'] = glob.glob(NYU_TEST_SN + '*.png')

    
    print("\nImages: ")
    print("Size of test data : ", len(images['test']))
    print("\nDepth: ")
    print("Size of test data : ", len(depth['test']))
    print("\nSurface Normal: ")
    print("Size of test data : ", len(sn['test']), "\n")
    
    print("\nImages: ")
    print("Size of train data : ", len(images['train']))
    print("\nDepth: ")
    print("Size of train data : ", len(depth['train']))
    print("\nSurface Normal: ")
    print("Size of train data : ", len(sn['train']), "\n")
    return images, sn, depth


# load Segmentation NYUv2

def load_sun_nyuv2_dataset():
    """
    loads sunrgbd dataset rgb images with their corresponding segmentation and depth images.
    """

    images = {'test':[], 'train':[]}
    depth = {'test':[], 'train':[]}
    seg = {'test':[], 'train':[]} 

    with open(NYU_TEST_IMAGES, 'r') as f:
        for line in f:
            images['test'].append(os.path.join(REL, line.strip('\n')))

    with open(NYU_TEST_DEPTH, 'r') as f:
        for line in f:
            depth['test'].append(os.path.join(REL, line.strip('\n')))

    with open(NYU_TEST_SEG, 'r') as f:
        for line in f:
            seg['test'].append(os.path.join(REL, line.strip('\n')))

    with open(NYU_TRAIN_IMAGES, 'r') as f:
        for line in f:
            images['train'].append(os.path.join(REL, line.strip('\n')))

    with open(NYU_TRAIN_DEPTH, 'r') as f:
        for line in f:
            depth['train'].append(os.path.join(REL, line.strip('\n')))

    with open(NYU_TRAIN_SEG, 'r') as f:
        for line in f:
            seg['train'].append(os.path.join(REL, line.strip('\n')))

    print("\nImages: ")
    print("Size of train data : ", len(images['train']))
    print("Size of test data : ", len(images['test']))
    print("\nDepth: ")
    print("Size of train data : ", len(depth['train']))
    print("Size of test data : ", len(depth['test']))
    print("\nSegmentation: ")
    print("Size of train data : ", len(seg['train']))
    print("Size of test data : ", len(seg['test']), "\n")

    return images, seg, depth
