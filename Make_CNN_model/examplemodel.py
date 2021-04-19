# coding: utf-8
# Author : Shin

"""
CNN을 이용한 이미지 태그 분류
작성 신용헌
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import utils
from tensorflow.python.keras import layers
from tensorflow.python.keras import datasets
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D


#버전에 맞는 텐서플로우 설치하기
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import utils
from tensorflow.keras import layers
from tensorflow.keras import datasets
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import cast

# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
random.seed(3)


# In[8]:


import cv2
from PIL import Image
from pathlib import Path
import os
import glob

######정규화 및 변경
from sklearn.preprocessing import MinMaxScaler

#df = pd.read_csv("totaldata.csv")
#df = df.drop(["Unnamed: 0"],axis=1)
#y_data = df['ID']
#df = df.drop(["ID"],axis=1)

df_test = pd.read_csv("D:/Jupyter/gitCBIR/totaldata_test.csv")
df_test = df_test.drop(["Unnamed: 0"],axis=1)
y_data_test = df_test['ID']
df_test = df_test.drop(["ID"],axis=1)

#정규화
scaler = MinMaxScaler()

#df[:] = scaler.fit_transform(df[:])
#df= pd.concat([y_data,df],axis=1)

df_test[:] = scaler.fit_transform(df_test[:])
df_test= pd.concat([y_data_test,df_test],axis=1)

#mindf 는 Min 값만 찾아둔것.
df_sorted_by_values = df_test.sort_values(by='ID',ascending=True)
mindf = df_sorted_by_values.min(axis=1)

dfvalmin = pd.concat([df_sorted_by_values,mindf],axis=1)
dfvalmin = dfvalmin.rename(columns={0:"MIN"})

import time
#시간 측정
start = time.time()

namelist =[]

for keys in dfvalmin.keys():
    namelist.append(keys)
print(namelist)

special_list = []
for k in range(len(mindf)):         #실제 실행할때는 전체 사용len(mindf)
    for i in range(1,13):
        if dfvalmin.iloc[k][i]==dfvalmin.iloc[k][13]:
            print(k+1,"번째 사진은",i,"번째 특성")
            print(dfvalmin.iloc[k][i],dfvalmin.iloc[k][13])
            print(namelist[i])
            special_list.append(namelist[i])
    print("=============")
print(special_list)

print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간
#######################################################################

for keynames in dfvalmin.keys():
    print(keynames)
    kkkkk = special_list.count(str(keynames))
    print(kkkkk)


#ln[]
group_name = ["Agroup","Bgroup","Cgroup","Dgroup","Egroup","Fgroup"
             ,"Ggroup","Hgroup","Igroup","Jgroup","Kgroup","Lgroup"]

## train data 이름 변경하기
folder_path = "D:/Jupyter/labeldata/train_data/"

for i in range(len(group_name)):
    #os.rename(folder_path + group_name[i],folder_path + namelist[i])
    os.rename(folder_path + namelist[i],folder_path + group_name[i]) #반대로 하는경우
    #print(group_name[i])
    file_list = glob.glob(folder_path+group_name[i]+"/*.png")
    count=1
    
    for files in file_list:
        os.rename(files,folder_path+group_name[i] + "/"+ group_name[i] + str(count) + ".png")
        count += 1
    
#시간은 5초정도 걸림

# In[9]:


# 미리 정해진 이미지들을 알기쉽게 텍스트로.
class_names = group_name

label_to_index = dict((name, index) for index,name in enumerate(class_names))

print(label_to_index)

# In[10]:


##라벨링 정의

def getPic(img_path):
    return np.array(Image.open(img_path).convert('RGB').resize((256,256),Image.ANTIALIAS))

# returns the Label of the image based on its first 3 characters
def get_label(img_path):
    return Path(img_path).absolute().name[0:6]

# Return the images and corresponding labels as numpy arrays
def get_ds(data_path):
    img_paths = list()
    # Recursively find all the image files from the path data_path
    for img_path in glob.glob(data_path+"/**/*"):
        img_paths.append(img_path)
    images = np.zeros((len(img_paths),256,256,3))
    labels = np.zeros(len(img_paths))
      
    # Read and resize the images
    # Get the encoded labels
    for i, img_path in enumerate(img_paths):
        images[i] = getPic(img_path)
        labels[i] = label_to_index[get_label(img_path)]
        
    return images,labels

def get_final_test(img_path):
    #images = np.zeros(1,256,256,3)
    images = np.array(Image.open(requests.get(img_path,stream=True).raw).convert('RGB').resize((256,256),Image.ANTIALIAS))
    images = images.astype('float32') /255
    images = (np.expand_dims(images,0))
    return images

# In[11]:


import time
start = time.time()  # 시작 시간 저장

train_images, train_labels = get_ds("D:/Jupyter/labeldata/train_data1111/")
test_images, test_labels = get_ds("D:/Jupyter/labeldata/test_data1111/")


print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간

# In[12]:


plt.figure(figsize=(10,10))

for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow((train_images[i]).astype(np.uint8) , cmap=plt.cm.binary)
    plt.xlabel(class_names[int(train_labels[i])])
plt.show()

# In[14]:


#어떻게 생겼는지 알아보자
print(type(train_images))
print(type(test_images))
print(train_images.shape[1:])
print(train_images.shape)
print(train_labels.shape)

# In[17]:


num_classes = len(class_names)
batch_size = 64
start = time.time()  # 시작 시간 저장

train_images = train_images.astype('float32') /255
test_images = test_images.astype('float32') /255
 
train_labels = utils.to_categorical(train_labels, num_classes)
test_labels = utils.to_categorical(test_labels, num_classes)


print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간
print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)

"""
#gpu테스트해보기
from tensorflow.python.client import device_lib
device_lib.list_local_devices()
#
"""
# In[1]:
model = keras.Sequential([
    keras.layers.Conv2D(filters=64, kernel_size=3, padding="same",  activation='relu', input_shape=(256,256,3)),
    keras.layers.Conv2D(filters=64, kernel_size=3, padding="same",  activation='relu'),
    keras.layers.MaxPooling2D(pool_size=2),
    keras.layers.Conv2D(filters=128, kernel_size=3,padding="same",  activation='relu'),
    keras.layers.Conv2D(filters=128, kernel_size=3,padding="same",  activation='relu'),
    keras.layers.MaxPooling2D(pool_size=2),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(num_classes, activation='softmax')
])

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 과대적합을 막기위해서 설정
early_stopping = EarlyStopping(monitor='val_loss', patience=15)

batch_size = 32
epochs = 5

history = model.fit(
    train_images, train_labels,
    epochs=epochs,
    batch_size=batch_size,
    #validation_split=0.2, #이걸 사용하면 정확도가 매우떨어진다. 이유는 label을 인식을 못하는것 같다.
    validation_data=(test_images,test_labels),
    shuffle=True,
    callbacks=[early_stopping]
)

model.save('test_model_no1.h5')

# In[42]:


#모델평가
loss, acc = model.evaluate(test_images,test_labels)
 
print("\nLoss: {}, Acc: {}".format(loss, acc))

# In[43]:


def plt_show_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc=0)

def plt_show_acc(history):
    plt.plot(history.history['acc']) #tensorflow 낮은버전에선 acc 쓰고 2.0 이후버전에선 accuracy 다써주기
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc=0)
    
plt_show_loss(history)
plt.show()
 
plt_show_acc(history)
plt.show()

# 예측
predictions = model.predict(train_images)

# In[1]:

def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
 
    plt.imshow(img, cmap=plt.cm.binary)
 
    predicted_label = np.argmax(predictions_array)
    if predicted_label == np.argmax(true_label):
        color = 'blue'
    else:
        color = 'red'
 
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[np.argmax(true_label)]),
                                color=color)



def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(num_classes), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    
    thisplot[predicted_label].set_color('red')
    thisplot[np.argmax(true_label)].set_color('blue')
  
  
num_rows = 10
num_cols = 8
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))

for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions, test_labels)
plt.show()

# ABCDE = "주소받아오기"
# 아래 주소부분 = ABCDE

My_images = get_final_test("https://we2d-app.s3.ap-northeast-2.amazonaws.com/designedImg/1600272696423.png")
My_predictions = model.predict(My_images)
print(My_predictions)
