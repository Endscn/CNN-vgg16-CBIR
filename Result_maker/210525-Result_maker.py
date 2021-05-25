# -*- coding: utf-8 -*-
# Author: Shin

#이 코드는 이미지를 읽어서 수치로 변환하는 코드이다. 마지막에 단하나의 print만 결과로 낸다.

# In[1]:
# 모듈 불러오기

import pandas as pd
from tensorflow import keras
import numpy as np
from PIL import Image
import requests
# from keras.models import load_model

from glob import glob
import time

# In[2]:
# 모델 불러오기

Test_model = keras.models.load_model(
    'D:/Jupyter/gitCBIR/flask-keras-cnn-image-retrieval-master/testmodels/210523_real_model_bat16_epoch5_mixed7_Adam5e-05_true_1.h5') #모델 위치 넣어주기



# In[3]:
# 함수 정의하기

# 이미지 주소받아와서 테스트에 적합한 형태로 만들어 준다.
# 1. 이미지 shape 따라서 mask 다르게 해서 기존 이미지를 흰색 배경에 합쳐준다.
# 2. (1,300,300,3) 모델에 넣으러면 이런 형태로 만들어 줘야 한다.
def get_final_test(img_path):
    # images = np.zeros(1,300,300,3)
    # images = np.array(Image.open(requests.get(img_path, stream=True).raw).convert('RGB').resize((300, 300), Image.ANTIALIAS))

    # 직접가져올때
    png = Image.open(img_path).convert().resize((300,300),Image.ANTIALIAS)

    # png에 이미지 주소로 불러오기
    #png = Image.open(requests.get(img_path, stream=True).raw)
    png_nparray = np.array(png)
    try:
        if png_nparray.shape[2] == 3:
            #print(png_nparray.shape)
            #print("3채널 이미지일때 결과")
            background = Image.new("RGB", png.size, (255, 255, 255))

            # background에 png에서 투명을 뺀 채널 가져와서 합성하고 300,300으로 맞춰서 테스트 가능하게 만들기.
            background.paste(png)  # 3 is the alpha channel
            background = background.resize((300, 300), Image.ANTIALIAS)

            # 합쳐진 이미지를 모델에 넣을 수 있도록 (1,300,300,3) 형식으로 차원을 하나 늘려준다.
            images = np.array(background)
            images = images[:, :, ::-1].copy()
            # print(images.shape)
            #cv2.imwrite("foo.png",images) #imwrite 이용하면 실제로 어떻게 보이는지 볼 수 있다.
            images = images.astype('float32') / 255
            images = (np.expand_dims(images, 0))


        elif png_nparray.shape[2] == 4:
            #print(png_nparray.shape)
            #print("4채널 이미지일때 결과")

            background = Image.new("RGB", png.size, (255, 255, 255))

            # background에 png에서 투명을 뺀 채널 가져와서 합성하고 300,300으로 맞춰서 테스트 가능하게 만들기.
            background.paste(png, mask=png.split()[3])  # 3 is the alpha channel
            background = background.resize((300, 300), Image.ANTIALIAS)

            # 합쳐진 이미지를 모델에 넣을 수 있도록 (1,300,300,3) 형식으로 차원을 하나 늘려준다.
            images = np.array(background)
            images = images[:, :, ::-1].copy()
            #cv2.imwrite("foo.png",images) #imwrite 이용하면 실제로 어떻게 보이는지 볼 수 있다.
            # print(images.shape)
            images = images.astype('float32') / 255
            images = (np.expand_dims(images, 0))

        elif png_nparray.shape[2] == 2:
            #print(png_nparray.shape)
            #print("투명 2채널 이미지일때 결과")

            background = Image.new("RGB", png.size, (255, 255, 255))

            # background에 png에서 투명을 뺀 채널 가져와서 합성하고 300,300으로 맞춰서 테스트 가능하게 만들기.
            background.paste(png, mask=png.split()[1])  # 3 is the alpha channel
            background = background.resize((300, 300), Image.ANTIALIAS)

            # 합쳐진 이미지를 모델에 넣을 수 있도록 (1,300,300,3) 형식으로 차원을 하나 늘려준다.
            images = np.array(background)
            #cv2.imwrite("foo.png",images) #imwrite 이용하면 실제로 어떻게 보이는지 볼 수 있다.
            # print(images.shape)
            images = images.astype('float32') / 255
            images = (np.expand_dims(images, 0))
        else:
            # png.shape()를 써서 앞 상황에서 적용안되면 고의로 except 상황으로 보내기. png.shape()는 없는 함수라서 바로 오류 뜸
            png.shape()
    except:
        #print(png_nparray.shape)
        #print("2채널 이미지일때 결과")
        background = Image.new("RGB", png.size, (255, 255, 255))

        # background에 png에서 투명을 뺀 채널 가져와서 합성하고 300,300으로 맞춰서 테스트 가능하게 만들기.
        background.paste(png)  #
        background = background.resize((300, 300), Image.ANTIALIAS)

        # 합쳐진 이미지를 모델에 넣을 수 있도록 (1,256,256,3) 형식으로 차원을 하나 늘려준다.
        images = np.array(background)
        # print(images.shape)
        #cv2.imwrite("foo.png",images) #imwrite 이용하면 실제로 어떻게 보이는지 볼 수 있다.
        images = images.astype('float32') / 255
        images = (np.expand_dims(images, 0))

    return images

# 최댓값 카테고리 정의
def first_max(input_predictions):
    My_predict_max = sorted(input_predictions[0], reverse=True) # 두번째 최댓값 찾을때부턴 정렬해서 2번째 이런식으로 찾음
    My_predict_max = My_predict_max[0]

    for i in range(len(group_name)):  #group name 돌면서
        if input_predictions[0][i] == My_predict_max: #input_prediction이 My_predict_max 랑 겹치면 그때의 위치를 가져온다.
            #print(input_predictions[0][i], "= max 값")
            # print(group_name[i],"= max 값의 카테고리")
            #print(i + 1, "= max값이 존재하는 위치")
            max_cate = group_name[i]  # df.columns 에서 i번째 있는 퍼센트를 maxcate에 넣는다.
    #print(max_cate)
    return max_cate

def second_max(input_predictions):
    My_predict_sec = sorted(input_predictions[0], reverse=True) # 두번째 최댓값 찾을때부턴 정렬해서 2번째 이런식으로 찾음
    My_predict_sec = My_predict_sec[1]
    # print(My_predict_sec)

    for i in range(len(group_name)):
        if input_predictions[0][i] == My_predict_sec:
            #print(input_predictions[0][i], "= 두번째로 큰 값")
            # print(group_name[i],"= 두번째로 큰 값의 카테고리")
            #print(i + 1, "= 두번째로 큰 값이 존재하는 위치")
            sec_max_cate = group_name[i]
    #print(sec_max_cate)
    return sec_max_cate

def third_max(input_predictions):
    My_predict_trd = sorted(input_predictions[0], reverse=True)
    My_predict_trd = My_predict_trd[2]
    # print(My_predict_trd)

    for i in range(len(group_name)):
        if input_predictions[0][i] == My_predict_trd:
            #print(input_predictions[0][i], "= 세번째로 큰 값")
            # print(group_name[i],"= 세번째로 큰 값의 카테고리")
            #print(i + 1, "= 세번째로 큰 값이 존재하는 위치")
            trd_max_cate = group_name[i]
    #print(trd_max_cate)
    return trd_max_cate

# ln[4]:
# 전체다 가져와서 DataFrame 만들기
# 실제로 진행할 때는 서버에서 받아와서 읽는다.
start = time.time()
data_pool = glob('D:/Jupyter/gitCBIR/flask-keras-cnn-image-retrieval-master/database/*.jpg')

database_pred = []
cnt=1
for i in data_pool:
    My_images = get_final_test(i)
    My_predictions = Test_model.predict(My_images)
    database_pred.append(My_predictions[0].tolist())
    print(cnt)
    cnt+=1
end = time.time()
print("time :", end - start)

print(database_pred)
#------------------------------------------------------------------------------------------------database_pred -----No.1

# In[5]:
#기준 설정 후 실험 
group_name = ['character_list', 'flower_list', 'widepatt_list',
              'dreamystical_list',
              'person_list',
              'scenery_list', 'calligraphy_list', 'pendrawing_list']


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
    print(meangap)
print(hap)
print(mean_df_list)

# ------------------------------------------------------------------------------------------------mean_df_list -----No.2


# use No.2 mean_df_list
trimed_df_list = mean_df_list 
trimed_df_list_arr = [np.array(trimed_df_list)]
print(trimed_df_list_arr)
trim_class = first_max(trimed_df_list_arr)
print(trim_class)

print(type(trim_class))
# 받아온 데이터 프레임 가져오면 된다.[[1,2],[3,4],[1,3]] 이런 형태의 데이터 그대로 넣어도 됨.

# use No.1 database_pred 
# df = pd.DataFrame() #형식맞춰서 우리 모든 이미지 데이터프레임으로 가져오기
df = pd.DataFrame(database_pred, columns=group_name)
print(df)
df.columns = group_name #컬럼명은 group_name 순서대로 가져옴

#----------------------------------------------------------
# 여기는 임의로 선택한 행에 대해서 근처를 보여주는 함수
# 랜덤row 생성  여기서 row 대신 우리가 필요한 데이터 가져오면 될듯.
print('저장된 평균값:')
row = pd.DataFrame(trimed_df_list).T
row.columns = group_name
row = row.loc[0,:]
print(row)
print(type(row))
print(df)
df = df.astype(dtype=float, copy=True, errors='raise')

# 임의로 생성한 유사한 행에 대해서 마스크 생성,적용

# 여기서 오류가 발생할 때가 있다. 왜냐하면 df의 dtype과 row의 dtype이 다르면 비교가 안됨.
# 지금같은 경우 df의 dtype을 변경해서 쓰는걸 해보자.

mask = np.logical_and.reduce([
    abs(df[trim_class] - row[trim_class]) <= 0.1
])

print(df[trim_class]-row[trim_class])
print(df[trim_class])
print(row[trim_class])

print('"Similar" Results:')
df_filtered = df.loc[mask, :]
print(df_filtered)
