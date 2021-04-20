# coding: utf-8
# Author : Shin

#import tensorflow as tf
#from tensorflow import cast
from tensorflow import keras
import numpy as np
from PIL import Image
import requests
import time

from keras.models import load_model

start = time.time()
model = keras.models.load_model('model_no1.h5')
end = time.time()
print("time :",end - start)

def get_final_test(img_path):
    #To make images = np.zeros(1,256,256,3)
    images = np.array(Image.open(requests.get(img_path,stream=True).raw).convert('RGB').resize((256,256),Image.ANTIALIAS))
    images = images.astype('float32') /255
    images = (np.expand_dims(images,0))
    return images

# ABCDE = "주소받아오기"
# 아래 주소부분 = ABCDE

My_images = get_final_test("https://we2d-app.s3.ap-northeast-2.amazonaws.com/designedImg/1600272696423.png")
My_predictions = model.predict(My_images)
print(My_predictions)
