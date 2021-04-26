# -*- coding: utf-8 -*-
# Author: Shin
from extract_cnn_vgg16_keras import VGGNet

import numpy as np
import h5py
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-query", required = True,
	help = "Path to query which contains image to be queried")
ap.add_argument("-index", required = True,
	help = "Path to index")
ap.add_argument("-result", required = True,
	help = "Path for output retrieved images")
args = vars(ap.parse_args())


# read in indexed images' feature vectors and corresponding image names
h5f = h5py.File(args["index"],'r')
# feats = h5f['dataset_1'][:]
feats = h5f['dataset_1'][:]
print(feats)
imgNames = h5f['dataset_2'][:]
print(imgNames)
h5f.close()
        
print("--------------------------------------------------")
print("               searching starts")
print("--------------------------------------------------")
    
# read and show query image
queryDir = args["query"]
queryImg = mpimg.imread(queryDir)
#plt.title("Query Image")
#plt.imshow(queryImg)
#plt.show()

# init VGGNet16 model
model = VGGNet()

# extract query image's feature, compute similarity score and sort
queryVec = model.extract_feat(queryDir)
scores = np.dot(queryVec, feats.T)
rank_ID = np.argsort(scores)[::-1]
rank_score = scores[rank_ID]
#print rank_ID
#print rank_score


# number of top retrieved images to show
maxres = 100
imlist = [imgNames[index] for i,index in enumerate(rank_ID[0:maxres])]
#print("top %d images in order are: " %maxres, imlist)

maxres_fake = 14491
imlist_fake = [imgNames[index] for i,index in enumerate(rank_ID[0:maxres_fake])]
#print("top %d images in order are: " %maxres_fake, imlist_fake)

"""
#show top #maxres images entirely---------------------------
plt.figure(figsize=(10,10))

for i,im in enumerate(imlist):
    image = mpimg.imread(args["result"] + "/" + str(im, 'utf-8'))
    plt.subplot(10,10,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image.astype(np.uint8) , cmap=plt.cm.binary)
    plt.xlabel(round(scores[rank_ID][i],4))
    print(i , im)
plt.show()

#print(imlist)
#print (scores[rank_ID])
"""
#save as a csv file and combine later--------------------------

plt.figure(figsize=(10,10))

D_F = pd.DataFrame()

Font_size = 6

for i,im in enumerate(imlist):
    image = mpimg.imread(args["result"] + "/" + str(im, 'utf-8'))
    plt.subplot(10,10,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.rc('font', size=Font_size)
    plt.imshow(image.astype(np.uint8) , cmap=plt.cm.binary)
    plt.xlabel(round(scores[rank_ID][i],5))
    print(i , im)

imlist_name = imlist[0]
imlist_name = imlist_name.decode('UTF-8')
imlist_name = imlist_name.replace(".jpg", "")

plt.savefig("D:/Jupyter/gitCBIR/flask-keras-cnn-image-retrieval-master/resultcsv/result" + imlist_name + ".png")
#plt.show()

""" 
imlist_name = imlist_fake[0]
tran_tbl = str.maketrans('b\'.jpg','      ')
imlist_name = imlist_name.decode('UTF-8')
imlist_name = imlist_name.translate(tran_tbl)
""" #table 이용한 제거방법 (띄어쓰기가 유지가 됨. results1    .csv

imlist_name4csv = imlist_fake[0]
imlist_name4csv = imlist_name4csv.decode('UTF-8')
imlist_name4csv = imlist_name4csv.replace(".jpg", "")
#replace 이용한 제거방법

for i,im in enumerate(imlist_fake):

    im = im.decode('UTF-8')
    im = im.replace(".jpg", "")
    print(im)
    print(imlist_name,"번째 사진으로 CSV 생성")
    D_F = D_F.append({"ID": im, imlist_name4csv + "SCORE": round(scores[rank_ID][i], 6)}, ignore_index=True)



D_F.to_csv("D:/Jupyter/gitCBIR/flask-keras-cnn-image-retrieval-master/resultcsv/result" + imlist_name4csv + ".csv", encoding='utf-8')

print("--------------------------------------------------")
print("            Successfully save csv file")
print("--------------------------------------------------")

"""
# show top #maxres retrieved result one by one
for i,im in enumerate(imlist):
    image = mpimg.imread(args["result"]+"/"+str(im, 'utf-8'))
    plt.title("search output %d" %(i+1))
    plt.imshow(image)
    plt.show()
"""

#python query_online.py -query database/1.jpg -index featureCNN.h5 -result database

#26 1622 8917 13891 13481 13429 13356 13314
