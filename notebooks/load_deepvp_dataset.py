def extract_labels(images):
    """
    returns vanishing point labels from the file name.
    :params: images - path to the images
    """
    
    labels = []
    for vp in images:
        vps = vp.split('/')[-1].split('_')[3:5]
        labels.append(vps)
        
    return labels

def get_data():
    """
    returns train val and test labels and their corresponding images.
    """
    data = {}
    label = {}
    
    with open('train.txt', 'r') as f1:
        train_images = f1.readlines()

    with open('val.txt', 'r') as f2:
        val_images = f2.readlines()

    with open('test.txt', 'r') as f3:
        test_images = f3.readlines()

    for i in range(len(train_images)):
        train_images[i] = train_images[i].strip('\n')
        
    for i in range(len(val_images)):
        val_images[i] = val_images[i].strip('\n')

    for i in range(len(test_images)):
        test_images[i] = test_images[i].strip('\n')
        
        
    train_labels = extract_labels(train_images)
    val_labels = extract_labels(val_images)
    test_labels = extract_labels(test_images)
    
    data['train'], data['val'], data['test'] = train_images, val_images, test_images
    label['train'], label['val'], label['test'] = train_labels, val_labels, test_labels
    
    return data, label
