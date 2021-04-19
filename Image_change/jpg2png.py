from PIL import Image
import cv2
import numpy as np
import requests
import csv
import os
from glob import glob #file list 받아오기

## jpeg to png 하는 코드

catego =  ['kni', #knife painting
        'pat', #pattern
        'dra', #drawing
        ]

for taip in catego:
    ##jpeg,jpg ->png 로 바꿀 폴더 위치를 설정해준다. taip 부분이 ani 폴더가 됨
        jpegs = glob('D:/training_images/usable/'+taip+'/*.jpeg')
        jpgs = glob('D:/training_images/usable/'+taip+'/*.jpg')

        for j in jpegs:
            img = cv2.imread(j, cv2.IMREAD_COLOR)
            cv2.imwrite(j[:-4] + 'png', img)

        for k in jpgs:
            img = cv2.imread(k, cv2.IMREAD_COLOR)
            cv2.imwrite(k[:-3] + 'png', img)
        
        print(taip," end")
