# -*- github Endscn -*-
# Author: Shin

#이 코드는 모든 유저아이디-클릭한이미지 데이터베이스에서 어떤 유저의 recent_data를 받아오면서 시작된다.

import pandas as pd
from collections import Counter

#기준 설정
group_name = ['character_list', 'flower_list', 'widepatt_list',
              'dreamystical_list',
              #'densepatt_list',
              'person_list',
              #'coloredwall_list',
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


##########데이터 로드 (최근 본 이미지 데이터) 임시로 설정해둠
recent_data = [ #여기에 리스트 형식으로 데이터 프레임 받아오기.
    [0.03107312321662903, 0.017883457243442535, 0.1646692305803299, 0.0024019479751586914, 0.02002149447798729, 0.001050083665177226, 0.029615627601742744, 0.7332850098609924],
    [0.0023397139739245176, 0.038091711699962616, 0.019511958584189415, 0.0019318792037665844, 0.012131009250879288, 0.0010019338224083185, 0.03776083514094353, 0.8872308731079102],
    [0.017946403473615646, 0.004906929098069668, 0.7139138579368591, 0.0009331188048236072, 0.0006144311046227813, 0.0009287179564125836, 0.07472322881221771, 0.18603338301181793],
    [0.0041369227692484856, 0.009776410646736622, 0.007823455147445202, 0.001007462851703167, 0.015998296439647675,
     0.00038706723717041314, 0.004343136213719845, 0.9565272331237793],
    [0.0008081936975941062, 0.002653488889336586, 0.0011466504074633121, 0.00013330242654774338, 0.22045817971229553,
     5.3979532822268084e-05, 0.0003383098228368908, 0.7744078636169434],
    [0.005169415380805731, 0.01192859373986721, 0.04010774940252304, 0.001123810070566833, 0.007715275045484304,
     0.0009067727369256318, 0.033374637365341187, 0.8996738195419312],
    [0.003388200653716922, 0.005364085081964731, 0.1015043705701828, 0.0006920119631104171, 0.0031869763042777777,
    0.000756317691411823, 0.17224770784378052, 0.7128604054450989],
    [0.014726607128977776, 0.006001607980579138, 0.005559870041906834, 0.013029847294092178, 0.13689394295215607,
     0.004910221789032221, 0.05718179792165756, 0.7616961002349854],
    [0.03475189581513405, 0.002613181946799159, 0.006318417377769947, 0.00044884736416861415, 0.013405590318143368, 8.432743197772652e-05, 0.0031350210774689913, 0.9392426609992981], # 이 위로는 전부다 펜드로잉
    [0.0006242059171199799, 0.9807424545288086, 0.01650126837193966, 0.0004215415974613279, 0.0009066245402209461, 4.334335244493559e-05, 0.0006144722574390471, 0.0001460998028051108], #꽃
    [2.607266651466489e-05, 0.9927816987037659, 0.0065936618484556675, 1.404024806106463e-05, 7.301350706256926e-06, 1.2628735930775292e-05, 0.00040004574111662805, 0.0001646520395297557] # 꽃
]

#데이터 프레임 형태로 전환하기.

train_df = pd.DataFrame(recent_data, columns=group_name) # columes = group_name

first_list =[]
second_list =[]
third_list =[]
for i in range(len(recent_data)):
    recent_data_list = [np.array(recent_data[i])]
    #print(recent_data_list[0])
    #제일 많이 들어있는걸 기준으로 만들어 보자.
    #print(i+1,"번째 이미지는")
    first_max(recent_data_list) #리스트에 계속 높게 나온 정보를 저장한다. (counter 이용해서 저장하기위함)
    second_max(recent_data_list)
    third_max(recent_data_list)
    #print("--------------------------------------")

first_most_com = Counter(first_list).most_common(1)[0][0]
second_most_com = Counter(second_list).most_common(1)[0][0]
third_most_com = Counter(third_list).most_common(1)[0][0]


# 함수 실행해서 아웃라이어 찾아내기
Detected_Out = find_Outlier(first_most_com)

# 아웃라이어 행 제거 하기
Outlier_index = Detected_Out.index
trimed_df = train_df.drop(index=Outlier_index)

# 정돈된 데이터프레임의 평균 내서 유저데이터베이스에 저장하기.

hap = 0
trimed_mean_df_list =[]
for i in group_name:
    meangap = trimed_df.describe()[i]["mean"]
    hap = hap + meangap
    trimed_mean_df_list.append(meangap)
    #print(meangap)
#print(hap)
print(trimed_mean_df_list)

# 여기까지 진행하면 이상치 제거하고 평균낸 리스트가 나온다. 이것을 유저데이터에 함께 저장해서 나중에 불러온다.
# trimed_df_list = 이상치 제거하고 평균낸 리스트 (어레이상태)
