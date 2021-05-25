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
#함수 정의하기
def first_max(input_predictions):
    My_predict_max = sorted(input_predictions[0], reverse=True)
    My_predict_max = My_predict_max[0]

    for i in range(len(group_name)):  #group name 돌면서
        if input_predictions[0][i] == My_predict_max: #input_prediction이 My_predict_max 랑 겹치면 그때의 위치를 가져온다.
            #print(input_predictions[0][i], "= max 값")
            # print(group_name[i],"= max 값의 카테고리")
            #print(i + 1, "= max값이 존재하는 위치")
            max_cate = group_name[i]  # df.columns 에서 i번째 있는 퍼센트를 maxcate에 넣는다.
    #print(max_cate)
    first_list.append(max_cate)
    return max_cate

def second_max(input_predictions):
    My_predict_sec = sorted(input_predictions[0], reverse=True) # 두번째 최댓값 찾을때부턴 정렬해서 2번째 이런식으로 찾음
    My_predict_sec = My_predict_sec[1]
    # print(My_predict_sec)

    for i in range(len(group_name)):
        if input_predictions[0][i] == My_predict_sec:
            # print(input_predictions[0][i], "= 두번째로 큰 값")
            # print(group_name[i],"= 두번째로 큰 값의 카테고리")
            # print(i + 1, "= 두번째로 큰 값이 존재하는 위치")
            sec_max_cate = group_name[i]
    # print(sec_max_cate)
    second_list.append(sec_max_cate)
    return sec_max_cate

def third_max(input_predictions):
    My_predict_trd = sorted(input_predictions[0], reverse=True)
    My_predict_trd = My_predict_trd[2]
    # print(My_predict_trd)

    for i in range(len(group_name)):
        if input_predictions[0][i] == My_predict_trd:
            # print(input_predictions[0][i], "= 세번째로 큰 값")
            # print(group_name[i],"= 세번째로 큰 값의 카테고리")
            # print(i + 1, "= 세번째로 큰 값이 존재하는 위치")
            trd_max_cate = group_name[i]
    # print(trd_max_cate)
    third_list.append(trd_max_cate)
    return trd_max_cate

def find_Outlier(input_most_common):
    # quartile_0 = train_df[input_most_common].quantile(0)  # percentile (같은 데이터 수로 100 등분), decile (같은 데이터 수로 10 등분), quartile (같은 데이터 수로 4 등분) #최소값
    quartile_1 = train_df[input_most_common].quantile(0.25)  # 1 사분위수 (25% 지점수)
    # quartile_2 = train_df[input_most_common].quantile(0.5)  # 2 사분위수 (50% 지점수, 중앙값 (median))
    quartile_3 = train_df[input_most_common].quantile(0.75)  # 3 사분위수 (70% 지점수)
    # quartile_4 = train_df[input_most_common].quantile(1)  # 최대값

    IQR = quartile_3 - quartile_1

    search_df = train_df[(train_df[input_most_common] < (quartile_1 - IQR)) | (
                train_df[input_most_common] > (quartile_3 + IQR))]
    # print(search_df)
    # print(quartile_1 - IQR,"보다 작거나", quartile_3 + IQR ,"보다 큰 데이터 제거하기")
    #print(input_most_common)
    return search_df



#랜덤으로 시작하기
random.seed(1)
recent_data = random.sample(full_list_pred,6)
recent_data.append(full_list_pred[0])
print(recent_data)



# 내가 원하는것을 하나씩 붙이는 방법
recent_data = []
for i in range(48):
    recent_data.append((full_list_pred[13]))
    
for i in range(0,55,7):
    recent_data.append(full_list_pred[i])
    print(i)
print(recent_data)
#데이터 프레임 형태로 전환하기.

train_df = pd.DataFrame(recent_data, columns=group_name) # columes = group_name

hap = 0
trimed_mean_df_list =[]
for i in group_name:
    meangap = train_df.describe()[i]["mean"]
    hap = hap + meangap
    trimed_mean_df_list.append(meangap)
    print(meangap)
print(hap)
print(trimed_mean_df_list)

