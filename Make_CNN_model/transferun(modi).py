# -*- github Endscn -*-
# Author: Shin

"""
CNN을 이용한 이미지 태그 분류
작성 신용헌
"""

# In[1]:

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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import cv2
from PIL import Image
from pathlib import Path
import os
import glob

"""

df = pd.read_csv("totaldata.csv")
df = df.drop(["Unnamed: 0"],axis=1)

#maxdf 는 Max값만 찾아둔것. 21년 4월 26일 작성
df_sorted_by_values = df.sort_values(by='ID',ascending=True)
y_data = df_sorted_by_values['ID']

df_sorted_by_values = df_sorted_by_values.drop(["ID"],axis=1)
print(df_sorted_by_values)

maxdf = df_sorted_by_values.max(axis=1)

dfvalmax = pd.concat([df_sorted_by_values,maxdf],axis=1)
dfvalmax = pd.concat([y_data,dfvalmax],axis=1)
dfvalmax = dfvalmax.rename(columns={0:"MAX"})

#y_data는 리스트화 하여 다시쓰자
y_data = pd.Series.tolist(y_data)

# In[4]:


import time
import pandas as pd
#시간 측정
start = time.time()

namelist =[]
special_list = []

for keys in dfvalmax.keys():
    namelist.append(keys)
print(namelist)
print(maxdf)
print(len(maxdf))
print(dfvalmax)

for k in range(len(maxdf)):
    for i in range(1,59):  #namecntlist는 특성의 갯수 현재 총 60여가지의 특성 맨앞과 맨뒤만 빼고 같은 것을 찾아준다고 생각. (ID값, MAX 이부분을 버려준다.)
        if dfvalmax.iloc[k][i]==dfvalmax.iloc[k][-1]:
            print("DataFrame 내의",k+1,"번째 데이터는",i,"번째 특성")
            print("사진 폴더 내의",y_data[k],"번째 사진")
            print(dfvalmax.iloc[k][i],dfvalmax.iloc[k][-1])
            print(namelist[i])
            print("------------------------------------------")
            special_list.append(namelist[i])
    print("=============")

#print(namelist)
print(len(namelist))
#print(special_list)
print(len(special_list))

print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간


#############각 특성 항목이 몇개를 보유했는지 알아보기

for keynames in dfvalmax.keys():
    print(keynames)
    key_cnt = special_list.count(str(keynames))
    print(key_cnt)

# In[17]:


#namelist.remove('ID')
#namelist.remove('MAX')


# In[3]: #
#remove ID and MAX line
""" #이건 폴더명이 group_name 과 다르게 b'*'.jpgSCORE 이렇게 되어있을때 필요한 코드임.


# In[2]:

group_name = ['character_list', 'flower_list','widepatt_list',
               'dreamystical_list','densepatt_list','person_list',
               'coloredwall_list','scenery_list','calligraphy_list','pendrawing_list']

group_name1 = ['char', 'flow','wide',
               'drea','dens','pers',
               'colo','scen','call','pend']


#첫 한번만 실행하면 됨
## train data 이름 변경하기
folder_path = "D:/Jupyter/gitCBIR/flask-keras-cnn-image-retrieval-master/label_img/train_data/"

for i in range(len(group_name)):
    #os.rename(folder_path + group_name[i],folder_path + namelist[i])
    #os.rename(folder_path + namelist[i],folder_path + group_name[i]) #반대로 하는경우
    #print(group_name[i])
    file_list = glob.glob(folder_path+group_name[i]+"/*.png")
    count=1
    
    for files in file_list:
        os.rename(files,folder_path+group_name[i] + "/"+ group_name[i] + str(count) + ".png")
        count += 1

    # 시간은 5초정도 소요

## validation_data 이름 변경하기
folder_path = "D:/Jupyter/gitCBIR/flask-keras-cnn-image-retrieval-master/label_img/validation_data/"

for i in range(len(group_name)):
    #os.rename(folder_path + group_name[i],folder_path + namelist[i])
    #os.rename(folder_path + namelist[i],folder_path + group_name[i]) #반대로 하는경우
    #print(group_name[i])
    file_list = glob.glob(folder_path+group_name[i]+"/*.png")
    count=1
    
    for files in file_list:
        os.rename(files,folder_path+group_name[i] + "/"+ group_name[i] + str(count) + ".png")
        count += 1
    
    #use os.rename


# In[3]:

# 미리 정해진 이미지들을 알기쉽게 텍스트로.
class_names = group_name1

label_to_index = dict((name, index) for index,name in enumerate(class_names))

print(label_to_index)



# In[4]:
##라벨링 정의

def getPic(img_path):
    return np.array(Image.open(img_path).convert('RGB').resize((300,300),Image.ANTIALIAS))

# returns the Label of the image based on its first 4 characters
def get_label(img_path):
    return Path(img_path).absolute().name[0:4]


# Return the images and corresponding labels as numpy arrays
def get_ds(data_path):
    img_paths = list()
    # Recursively find all the image files from the path data_path
    for img_path in glob.glob(data_path+"/**/*"):
        img_paths.append(img_path)
    images = np.zeros((len(img_paths),300,300,3))
    labels = np.zeros(len(img_paths))
      
    # Read and resize the images
    # Get the encoded labels
    for i, img_path in enumerate(img_paths):
        images[i] = getPic(img_path)
        labels[i] = label_to_index[get_label(img_path)]
        
    return images,labels

#테스트 정의
def get_final_test(img_path):
    #images = np.zeros(1,256,256,3)
    images = np.array(Image.open(requests.get(img_path,stream=True).raw).convert('RGB').resize((300,300),Image.ANTIALIAS))
    images = images.astype('float32') /255
    images = (np.expand_dims(images,0))
    return images


# In[5]:
# 라벨링 하기

import time
start = time.time()  # 시작 시간 저장

train_images, train_labels = get_ds("D:/Jupyter/gitCBIR/flask-keras-cnn-image-retrieval-master/label_img/train_data/")
print("train image labeling done")
test_images, test_labels = get_ds("D:/Jupyter/gitCBIR/flask-keras-cnn-image-retrieval-master/label_img/validation_data/")
print("validation image labeling done")

print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간
#1. 1440초 소요 #2. 2389초 소요 #3. 1575초 소요 #4. 1866초 소요


# In[6]:
#이미지 프리뷰

plt.figure(figsize=(10,10))

for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow((train_images[i]).astype(np.uint8) , cmap=plt.cm.binary)
    plt.xlabel(class_names[int(train_labels[i])])
plt.show()



# In[7]:
#이미지 shape 알아보기
print(type(train_images))
print(type(test_images))
print(train_images.shape[1:])
print(train_images.shape)
print(train_labels.shape)


# In[8]:

num_classes = len(class_names)

start = time.time()  # 시작 시간 저장

train_images = train_images.astype('float32') /255
test_images = test_images.astype('float32') /255
 
train_labels = utils.to_categorical(train_labels, num_classes)
test_labels = utils.to_categorical(test_labels, num_classes)


print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간
#1. 240초 소요 #3. 155초 소요

print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)

# In[8-1]:
# train_images 로 저장한 nparray를 npy파일로 저장해서 나중에 쓰기 # 3분내외
"""
start = time.time()  # 시작 시간 저장
np.save('D:/Jupyter/gitCBIR/flask-keras-cnn-image-retrieval-master/label_img/train_img_save', train_images) # train_save.npy
np.save('D:/Jupyter/gitCBIR/flask-keras-cnn-image-retrieval-master/label_img/train_lab_save', train_labels) # train_save.npy
np.save('D:/Jupyter/gitCBIR/flask-keras-cnn-image-retrieval-master/label_img/test_img_save', test_images) # train_save.npy
np.save('D:/Jupyter/gitCBIR/flask-keras-cnn-image-retrieval-master/label_img/test_lab_save', test_labels) # train_save.npy
print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간

start = time.time()  # 시작 시간 저장
train_i = np.load('D:/Jupyter/gitCBIR/flask-keras-cnn-image-retrieval-master/label_img/train_img_save.npy')
train_l = np.load('D:/Jupyter/gitCBIR/flask-keras-cnn-image-retrieval-master/label_img/train_lab_save.npy')
test_i = np.load('D:/Jupyter/gitCBIR/flask-keras-cnn-image-retrieval-master/label_img/test_img_save.npy')
test_l =np.load('D:/Jupyter/gitCBIR/flask-keras-cnn-image-retrieval-master/label_img/test_lab_save.npy')
print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간 
#저장하는데 train_image 빼고 90초 전체다 불러오는데 72초 걸림
#print(train_i)
"""

# In[9]:
# 모델만들기
import requests
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3

from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Adam

"""
local_weights_file = 'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

pre_trained_model = InceptionV3(input_shape=(300, 300, 3),
                              include_top=False,
                              weights=None)

#전이학습에 필요한 모델 로드하기.
pre_trained_model.load_weights(local_weights_file)

#미리 학습된 모델이 학습 불가능하게 설정(변경되지않도록 고정)
for layer in pre_trained_model.layers:
  layer.trainable = False

#모델 요약
pre_trained_model.summary()

#--------------
last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output.shape)
last_output = last_layer.output

#--------------
model_x = layers.Flatten()(last_output)
#model_x = layers.Dense(16, activation='relu')(model_x)
model_x = layers.Dense(num_classes, activation='softmax')(model_x)

model = Model(pre_trained_model.input, model_x)

#opt1 = keras.optimizers.Adam(lr=0.001)
#opt2 = keras.optimizers.RMSprop(lr=0.0001) #RMSprop = 0.0001에서 적용됨 0.001에선 적용안됨

model.compile(optimizer=Adam(lr=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy'])

# 과대적합을 막기위해서 설정
#early_stopping = EarlyStopping(monitor='val_loss', patience=10)

batch_size = 32
#batch_size = 128 #변경해서 한번 적용해보자.
epochs = 15

history = model.fit(
    train_images, train_labels,
    epochs=epochs,
    batch_size=batch_size,
    #validation_split=0.2, #이걸 사용하면 정확도가 매우떨어진다. 이유는 정확히 모르지만 label을 인식을 못하는것 같다.
    validation_data=(test_images,test_labels),
    shuffle=True,
    #callbacks=[early_stopping]
)

#새로운 모델 저장 및 사진 저장
model_name = "test_model" + "_bat" + str(batch_size) + "_epoch" + str(epochs) + "_mixed7"+"_Adam"+"0_0005"
model.save(model_name+".h5")

plt_show_acc(history)
plt.savefig(model_name+".png")

#직전에 한것 아담0.001

#1트 bat32/epoch50 no1
#2트 bat32/epoch5 no2
#3트 bat32/epoch10 no3
#4트 bat128/epoch5
#5트 bat128/epoch10
#6트 dense 1024->16으로 변경 or 삭제

#7트 bat32 / epoch10 / mixed7번 라인에서 끊음 / dense 1024 뺌 / Adam(lr=0.0005) 유의미한 느낌?
""" #반복없이 한번만 모델만드는 코드

batch_list = [16, 32, 64, 128]
lr_list = [0.005, 0.001, 0.0005, 0.0001]
predict_list1 = []
predict_list2 = []

for lr in lr_list:
    for bat in batch_list:

        local_weights_file = 'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

        pre_trained_model = InceptionV3(input_shape=(300, 300, 3),
                                        include_top=False,
                                        weights=None)

        # 전이학습에 필요한 모델 로드하기.
        pre_trained_model.load_weights(local_weights_file)

        # 미리 학습된 모델이 학습 불가능하게 설정(변경되지않도록 고정)
        for layer in pre_trained_model.layers:
            layer.trainable = False

        # 모델 요약
        pre_trained_model.summary()

        # --------------
        last_layer = pre_trained_model.get_layer('mixed7')
        print('last layer output shape: ', last_layer.output.shape)
        last_output = last_layer.output

        # --------------
        model_x = layers.Flatten()(last_output)
        # model_x = layers.Dense(16, activation='relu')(model_x) #모델 추가없이 제작
        model_x = layers.Dense(num_classes, activation='softmax')(model_x)

        model = Model(pre_trained_model.input, model_x)

        # opt1 = keras.optimizers.Adam(lr=0.001)
        # opt2 = keras.optimizers.RMSprop(lr=0.0001) #RMSprop = 0.0001에서 적용됨 0.001에선 적용안됨

        model.compile(optimizer=Adam(lr=lr),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        # 과대적합을 막기위해서 설정
        # early_stopping = EarlyStopping(monitor='val_loss', patience=10)

        batch_size = bat
        # batch_size = 128 #변경해서 한번 적용해보자.
        epochs = 20

        history = model.fit(
            train_images, train_labels,
            epochs=epochs,
            batch_size=batch_size,
            # validation_split=0.2, #이걸 사용하면 정확도가 매우떨어진다. 이유는 정확히 모르지만 label을 인식을 못하는것 같다.
            validation_data=(test_images, test_labels),
            shuffle=True,
            # callbacks=[early_stopping]
        )

        # 새로운 모델 저장 및 사진 저장
        model_name = "test_model" + "_bat" + str(batch_size) + "_epoch" + str(epochs) + "_mixed7" + "_Adam" + str(lr)
        model.save(model_name + ".h5")

        plt_show_acc(history)
        plt.savefig(model_name + ".png")

        My_images = get_final_test(
            "https://we2d-app.s3.ap-northeast-2.amazonaws.com/designedImg/1600272696423.png")  # 글씨 + 캐릭터
        My_predictions = model.predict(My_images)
        print(My_predictions)
        print(group_name)
        predict_list1.append(My_predictions)


for lr in lr_list:
    for bat in batch_list:

        local_weights_file = 'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

        pre_trained_model = InceptionV3(input_shape=(300, 300, 3),
                                        include_top=False,
                                        weights=None)

        # 전이학습에 필요한 모델 로드하기.
        pre_trained_model.load_weights(local_weights_file)

        # 미리 학습된 모델이 학습 불가능하게 설정(변경되지않도록 고정)
        for layer in pre_trained_model.layers:
            layer.trainable = False

        # 모델 요약
        pre_trained_model.summary()

        # --------------
        last_layer = pre_trained_model.get_layer('mixed7')
        print('last layer output shape: ', last_layer.output.shape)
        last_output = last_layer.output

        # --------------
        model_x = layers.Flatten()(last_output)
        model_x = layers.Dense(16, activation='relu')(model_x)
        model_x = layers.Dense(num_classes, activation='softmax')(model_x)

        model = Model(pre_trained_model.input, model_x)

        # opt1 = keras.optimizers.Adam(lr=0.001)
        # opt2 = keras.optimizers.RMSprop(lr=0.0001) #RMSprop = 0.0001에서 적용됨 0.001에선 적용안됨

        model.compile(optimizer=Adam(lr=lr),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        # 과대적합을 막기위해서 설정
        # early_stopping = EarlyStopping(monitor='val_loss', patience=10)

        batch_size = bat
        # batch_size = 128 #변경해서 한번 적용해보자.
        epochs = 20

        history = model.fit(
            train_images, train_labels,
            epochs=epochs,
            batch_size=batch_size,
            # validation_split=0.2, #이걸 사용하면 정확도가 매우떨어진다. 이유는 정확히 모르지만 label을 인식을 못하는것 같다.
            validation_data=(test_images, test_labels),
            shuffle=True,
            # callbacks=[early_stopping]
        )

        # 새로운 모델 저장 및 사진 저장
        model_name = "test_model" + "_bat" + str(batch_size) + "_epoch" + str(epochs) + "_mixed7" + "_Adam" + str(lr) + "_denseplus"
        model.save(model_name + ".h5")

        plt_show_acc(history)
        plt.savefig(model_name + ".png")

        My_images = get_final_test(
            "https://we2d-app.s3.ap-northeast-2.amazonaws.com/designedImg/1600272696423.png")  # 글씨 + 캐릭터
        My_predictions = model.predict(My_images)
        print(My_predictions)
        print(group_name)
        predict_list2.append(My_predictions)



# In[10]:
# 모델평가
loss, acc = model.evaluate(test_images,test_labels)
 
print("\nLoss: {}, Acc: {}".format(loss, acc))


# In[11]:
# 보여주기
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
    
#plt_show_loss(history)
#plt.show()
 
plt_show_acc(history)
plt.show()


# In[12]:
# 전체 사진을 예측해서 "predictions" 에 저장하기
start = time.time()  # 시작 시간 저장
predictions = model.predict(train_images)
print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간
#70초 소요

# In[13]:
# 전체 예측한 사진을 그림으로 나타내기
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i+1100], true_label[i+1100], img[i+1100]
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
    predictions_array, true_label = predictions_array[i+1100], true_label[i+1100]
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


# In[14]:
# 새로운 하나의 이미지 불러와서 퍼센트로 나타내기
"""
# ABCDE = "주소받아오기"
# 아래 주소부분 = ABCDE

My_images = get_final_test("https://we2d-app.s3.ap-northeast-2.amazonaws.com/designedImg/1600272696423.png")
My_images = My_images.astype('float32') /255
print(type(My_images))
print(My_images.shape)

# 이미지 하나만 사용할 때도 배치에 추가, 차원을 늘려줘야함
My_images = (np.expand_dims(My_images,0))
print(My_images.shape)
"""


# ABCDE = "주소받아오기"
# 아래 주소부분 = ABCDE

import requests
My_images = get_final_test("https://we2d-app.s3.ap-northeast-2.amazonaws.com/designedImg/1600272696423.png") #글씨 + 캐릭터
My_images = get_final_test("https://we2d-app.s3.ap-northeast-2.amazonaws.com/designedImg/1608227644728.png") #사과나무
My_images = get_final_test("https://we2d-app.s3.ap-northeast-2.amazonaws.com/designedImg/1614180544168.png")
My_predictions = model.predict(My_images)
print(My_predictions)
#predict_name =max(My_predictions)
#print(predict_name)
print(group_name)


# In[15]:


#get_ipython().system('python --version')

