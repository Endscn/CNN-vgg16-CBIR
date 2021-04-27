# # Creating Train / Val / Test folders (One time use)
# kaggle data 참조

import os
import numpy as np
import shutil
import random
root_dir = 'D:/Jupyter/gitCBIR/flask-keras-cnn-image-retrieval-master/label_img/train_data/' # data root path
classes_dir = ['character_list', 'flower_list','widepatt_list',
               'dreamystical_list','densepatt_list','person_list',
               'coloredwall_list','scenery_list','calligraphy_list','pendrawing_list'] #total labels




val_ratio = 0.25
test_ratio = 0.00

for cls in classes_dir:
    os.makedirs(root_dir +'train/' + cls)
    os.makedirs(root_dir +'val/' + cls)
    os.makedirs(root_dir +'test/' + cls)


# Creating partitions of the data after shuffeling
src = root_dir + cls # Folder to copy images from

allFileNames = os.listdir(src)
np.random.shuffle(allFileNames)
train_FileNames, val_FileNames, test_FileNames = np.split(np.array(allFileNames), [int(len(allFileNames)*(1 - (val_ratio + test_ratio))), int(len(allFileNames)* (1 - test_ratio))])


train_FileNames = [src+'/'+ name for name in train_FileNames.tolist()]
val_FileNames = [src+'/' + name for name in val_FileNames.tolist()]
test_FileNames = [src+'/' + name for name in test_FileNames.tolist()]

print('Total images: ', len(allFileNames))
print('Training: ', len(train_FileNames))
print('Validation: ', len(val_FileNames))
print('Testing: ', len(test_FileNames))

for cls in classes_dir:
       # Copy-pasting images
       for name in train_FileNames:
              shutil.copy(name, root_dir +'train/' + cls)

       for name in val_FileNames:
              shutil.copy(name, root_dir +'val/' + cls)

       for name in test_FileNames:
               shutil.copy(name, root_dir +'test/' + cls)

#val test까지 한번에 나눌 때 사용한다. 3갈래로 나눠진다는 특징이 있다.
