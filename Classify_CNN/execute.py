# -*- coding: utf-8 -*-
# Author: Shin

import random
random.seed(1)
import time

"""
temp = list( range(1, 14491, 1) )
ran_list = random.sample(temp, k=14490)
print(ran_list)


for i in ran_list:
    print(i)
    !python query_online.py -query database/{i}.jpg -index featureCNN.h5 -result database

""" #random list 랜덤으로 고른 사진으로 사진&CSV 파일 생성하는 코드

#------------------------------------------------------
pick_list = []

# 필요한 사진 번호 입력하기
while True:
    print("종료하려면 0을 입력하세요")
    A = int(input())
    if A==0:
        break
    pick_list.append(A)
    
for i in pick_list:
    print(i)
    !python query_online.py -query database/{i}.jpg -index featureCNN.h5 -result database


# pick_list에 내가고른 번호 받아서 그거만 뽑아오는 코드 보통 "리스트길이*1분" 걸린다.
