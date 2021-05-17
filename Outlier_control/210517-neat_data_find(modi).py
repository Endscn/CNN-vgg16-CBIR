# -*- github Endscn -*-
# Author: Shin

#이 코드는 오브젝트에 저장된 어레이 가져오는것 부터 시작된다.

import pandas as pd
import numpy as np

#함수 정의하기
def first_max(input_predictions):
    My_predict_max = sorted(input_predictions[0], reverse=True)
    My_predict_max = My_predict_max[0]

    for i in range(len(group_name)):  #group name 돌면서
        if input_predictions[0][i] == My_predict_max: #input_prediction이 My_predict_max 랑 겹치면 그때의 위치를 가져온다.
            print(input_predictions[0][i], "= max 값")
            # print(group_name[i],"= max 값의 카테고리")
            print(i + 1, "= max값이 존재하는 위치")
            max_cate = group_name[i]  # df.columns 에서 i번째 있는 퍼센트를 maxcate에 넣는다.
    print(max_cate)
    first_list.append(max_cate)
    return max_cate

def second_max(input_predictions):
    My_predict_sec = sorted(input_predictions[0], reverse=True) # 두번째 최댓값 찾을때부턴 정렬해서 2번째 이런식으로 찾음
    My_predict_sec = My_predict_sec[1]
    # print(My_predict_sec)

    for i in range(len(group_name)):
        if input_predictions[0][i] == My_predict_sec:
            print(input_predictions[0][i], "= 두번째로 큰 값")
            # print(group_name[i],"= 두번째로 큰 값의 카테고리")
            print(i + 1, "= 두번째로 큰 값이 존재하는 위치")
            sec_max_cate = group_name[i]
    print(sec_max_cate)
    second_list.append(sec_max_cate)
    return sec_max_cate

trimed_df_list = [] # trimed_df_list에다가 오브젝트에 저장된 유저의 평균값 받아오기

trimed_df_list_arr = [np.array(trimed_df_list)]
print(trimed_df_list_arr)
trim_class = first_max(trimed_df_list_arr)
print(trim_class)


# 받아온 데이터 프레임 가져오면 된다.[[1,2],[3,4],[1,3]] 이런 형태의 데이터 그대로 넣어도 됨.
# df = pd.DataFrame() #형식맞춰서 우리 모든 이미지 데이터프레임으로 가져오기
df.columns = group_name #컬럼명은 group_name 순서대로 가져옴

#----------------------------------------------------------
# 여기는 임의로 선택한 행에 대해서 근처를 보여주는 함수
# 랜덤row 생성  여기서 row 대신 우리가 필요한 데이터 가져오면 될듯.
print('저장된 평균값:')
row = pd.DataFrame(trimed_df_list).T
row.columns = group_name
print(row)

# 임의로 생성한 유사한 행에 대해서 마스크 생성,적용
mask = np.logical_and.reduce([
    df[trim_class] == row[trim_class],
    abs(df['b'] - row['b']) <= 1,
    df['h'] == (3 - row['h'])
])


print('"Similar" Results:')
df_filtered = df.loc[mask, :]
print(df_filtered)

