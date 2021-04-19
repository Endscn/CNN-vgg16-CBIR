# -*- coding: utf-8 -*-
# Author: Shin

from PIL import Image
import cv2
import numpy as np
import requests
import csv
import os
from glob import glob #file list 받아오기        

##cate의 각각 폴더에 사진을 넣어주어야 한다.

cate =  ['ani', #동물
        'chi', #시크
        'cul', #B급감성
        'cut', #귀여운
        'dar', #어두운
        'emo', #감성적인
        'hip', #힙한
        'mod', #모던한
        'mys', #신비로운
        'per', #인물 
        'ret', #레트로
        'sce', #풍경
        'sim', #심플한
        'tra', #전통적인
        'uni'  #유니크한
        ]


for tip in cate:
    ##이미지 사이즈 조정이 필요한 사진들이 있는 폴더를 지정해준다
    imglist = glob("/Projects/keras_talk/imagedown/image-Downloader/download_images/"+tip+"/*.png")
    a=0
    for img_path in imglist:
        a= a+1
        print(a)
        canvas_width = 1500
        canvas_height = 1500
        canvas = np.zeros((canvas_width,canvas_height,4), np.uint8)    

        image = Image.open(img_path)
        image = cv2.imread(img_path , cv2.IMREAD_COLOR) #ANYCOLOR에서 COLOR로 바꾸니까 됨

        h,w = image.shape[:2]
        
        if(h<=400):
            image = cv2.resize(image, dsize=(int(w*5), int(h*5)), interpolation=cv2.INTER_AREA)
            h, w = image.shape[:2]

        if(w<=400):
            image = cv2.resize(image, dsize=(int(w*5), int(h*5)), interpolation=cv2.INTER_AREA)
            h, w = image.shape[:2]

        if(h<=1000):
            image = cv2.resize(image, dsize=(int(w*3), int(h*3)), interpolation=cv2.INTER_AREA)
            h, w = image.shape[:2]

        if(w<=1000):
            image = cv2.resize(image, dsize=(int(w*3), int(h*3)), interpolation=cv2.INTER_AREA)
            h, w = image.shape[:2]

        if(h>=1500):
            image = cv2.resize(image, dsize=(int(w/h*(canvas_height-10)), int(canvas_height-10)), interpolation=cv2.INTER_AREA)
            h, w = image.shape[:2]

        if(w>=1500):
            image = cv2.resize(image, dsize=(int(canvas_width-10), int(h/w*(canvas_width-10))), interpolation=cv2.INTER_AREA)
            h, w = image.shape[:2]

        print(h,w)        
        
        x_offset = int((canvas_width-w)/2)
        y_offset = int((canvas_height-h)/2)
        canvas[y_offset: y_offset+h, x_offset: x_offset+ w] = image
        
        ##여기 따옴표 사이엔 내가 새로 이미지를 저장할 위치를 입력
        ##tip에 ani,cut 등이 들어가는것

        fileDir = '/Projects/keras_talk/productimages/'+tip
        print(tip)

        if os.path.exists(fileDir):
            cv2.imwrite('/Projects/keras_talk/productimages/' + tip + '/' + tip + str(a) + '.png', canvas)
        else :
            os.makedirs(fileDir)
            cv2.imwrite('/Projects/keras_talk/productimages/' + tip + '/'+ tip + str(a) +'.png', canvas)


print('-----------------End------------------')
