# -*- github Endscn -*-
# Author: Shin

#이 코드는 처음 이미지 테스트할 때 어떤 이미지를 누르는지에 따라 퍼센트를 구해서

import pandas as pd
import random

#기준 설정
group_name = ['character_list', 'flower_list', 'widepatt_list',
              'dreamystical_list',
              'person_list',
              'scenery_list', 'calligraphy_list', 'pendrawing_list']

"""
#랜덤으로 시작하기
random.seed(1)
tasty_data = random.sample(full_list_pred,6)
tasty_data.append(full_list_pred[0])
print(tasty_data)

# 내가 원하는것을 하나씩 붙이는 방법
tasty_data = []
for i in range(48):
    tasty_data.append((full_list_pred[13]))
    
for i in range(0,55,7):
    tasty_data.append(full_list_pred[i])
    print(i)
print(tasty_data)
""" #실험용

#여기에 실제 데이터 받아오면됨
tasty_data = []

#데이터 프레임 형태로 전환하기.

train_df = pd.DataFrame(tasty_data, columns=group_name) # columes = group_name

hap = 0
mean_df_list =[]
for i in group_name:
    meangap = train_df.describe()[i]["mean"]
    hap = hap + meangap
    mean_df_list.append(meangap)
    #print(meangap)
#print(hap)
print(mean_df_list)
