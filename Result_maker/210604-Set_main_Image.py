# -*- coding: utf-8 -*-
# Author: Shin

# 취향테스트를 위한 메인 이미지를 설정하는 코드이다.

# In[1]:
# 모듈 불러오기

from tensorflow import keras
import numpy as np
from PIL import Image
import requests
# from keras.models import load_model
# import tensorflowjs as tfjs
# In[2]:
# 모델 불러오기

model = keras.models.load_model(
    'D:/Jupyter/gitCBIR/flask-keras-cnn-image-retrieval-master/testmodels/210604_real_model_bat16_epoch1_mixed7_Adam2.5e-05_30point0.25second_verygood.h5') #모델 위치 넣어주기
    #'D:/Jupyter/gitCBIR/flask-keras-cnn-image-retrieval-master/testmodels/210523_real_model_bat16_epoch5_mixed7_Adam5e-05_true_1.h5') # 원본파일 이 모델로 사용했었음



# In[2]:
# 함수 정의하기

# 이미지 주소받아와서 테스트에 적합한 형태로 만들어 준다.
# 1. 이미지 shape 따라서 mask 다르게 해서 기존 이미지를 흰색 배경에 합쳐준다.
# 2. (1,300,300,3) 모델에 넣으러면 이런 형태로 만들어 줘야 한다.
def get_final_test_png(img_path):
    # images = np.zeros(1,300,300,3)
    # images = np.array(Image.open(requests.get(img_path, stream=True).raw).convert('RGB').resize((300, 300), Image.ANTIALIAS))

    # 직접가져올때
    png = Image.open(img_path)

    # png에 이미지 불러오기
    #png = Image.open(requests.get(img_path, stream=True).raw)
    png_nparray = np.array(png)
    #print(png.size)
    try:
        if png_nparray.shape[2] == 3:
            #print(png_nparray.shape)
            #print("3채널 이미지일때 결과")
            if png.size[1]>=png.size[0]:
                background = Image.new("RGB", (png.size[1]+10,png.size[1]+10), (255, 255, 255))
                # background에 png에서 투명을 뺀 채널 가져와서 합성하고 300,300으로 맞춰서 테스트 가능하게 만들기.
                background.paste(png, box=(int((png.size[1]+10-png.size[0])/2),5))  # 3 is the alpha channel
            else:
                background = Image.new("RGB", (png.size[0]+10,png.size[0]+10), (255, 255, 255))
                # background에 png에서 투명을 뺀 채널 가져와서 합성하고 300,300으로 맞춰서 테스트 가능하게 만들기.
                background.paste(png, box=(5,int((png.size[0]+10-png.size[1])/2)))  # 3 is the alpha channel
            background = background.resize((500, 500), Image.ANTIALIAS)

            # 합쳐진 이미지를 모델에 넣을 수 있도록 (1,300,300,3) 형식으로 차원을 하나 늘려준다.
            images = np.array(background)
            images = images[:, :, ::-1].copy()
            # print(images.shape)
            cv2.imwrite("foo.png",images) #imwrite 이용하면 실제로 어떻게 보이는지 볼 수 있다.
            images = images.astype('float32') / 255
            images = (np.expand_dims(images, 0))


        elif png_nparray.shape[2] == 4:
            #print(png_nparray.shape)
            #print("4채널 이미지일때 결과")
            if png.size[1]>=png.size[0]:
                background = Image.new("RGB", (png.size[1]+10,png.size[1]+10), (255, 255, 255))
                # background에 png에서 투명을 뺀 채널 가져와서 합성하고 300,300으로 맞춰서 테스트 가능하게 만들기.
                background.paste(png, box=(int((png.size[1]+10-png.size[0])/2),5), mask=png.split()[3])  # 3 is the alpha channel
            else:
                background = Image.new("RGB", (png.size[0]+10,png.size[0]+10), (255, 255, 255))
                # background에 png에서 투명을 뺀 채널 가져와서 합성하고 300,300으로 맞춰서 테스트 가능하게 만들기.
                background.paste(png, box=(5,int((png.size[0]+10-png.size[1])/2)), mask=png.split()[3])  # 3 is the alpha channel
            background = background.resize((500, 500), Image.ANTIALIAS)

            # 합쳐진 이미지를 모델에 넣을 수 있도록 (1,300,300,3) 형식으로 차원을 하나 늘려준다.
            images = np.array(background)
            images = images[:, :, ::-1].copy()
            cv2.imwrite("foo.png",images) #imwrite 이용하면 실제로 어떻게 보이는지 볼 수 있다.
            # print(images.shape)
            images = images.astype('float32') / 255
            images = (np.expand_dims(images, 0))

        elif png_nparray.shape[2] == 2:
            #print(png_nparray.shape)
            #print("투명 2채널 이미지일때 결과")
            if png.size[1]>=png.size[0]:
                background = Image.new("RGB", (png.size[1]+10,png.size[1]+10), (255, 255, 255))
                # background에 png에서 투명을 뺀 채널 가져와서 합성하고 300,300으로 맞춰서 테스트 가능하게 만들기.
                background.paste(png, box=(int((png.size[1]+10-png.size[0])/2,5)), mask=png.split()[1])  # 3 is the alpha channel
            else:
                background = Image.new("RGB", (png.size[0]+10,png.size[0]+10), (255, 255, 255))
                # background에 png에서 투명을 뺀 채널 가져와서 합성하고 300,300으로 맞춰서 테스트 가능하게 만들기.
                background.paste(png, box=(5,int((png.size[0]+10-png.size[1])/2)), mask=png.split()[1])  # 3 is the alpha channel
            background = background.resize((500, 500), Image.ANTIALIAS)

            # 합쳐진 이미지를 모델에 넣을 수 있도록 (1,300,300,3) 형식으로 차원을 하나 늘려준다.
            images = np.array(background)
            cv2.imwrite("foo.png",images) #imwrite 이용하면 실제로 어떻게 보이는지 볼 수 있다.
            # print(images.shape)
            images = images.astype('float32') / 255
            images = (np.expand_dims(images, 0))
        else:
            # png.shape()를 써서 앞 상황에서 적용안되면 고의로 except 상황으로 보내기. png.shape()는 없는 함수라서 바로 오류 뜸
            png.shape()
    except:
        #print(png_nparray.shape)
        #print("2채널 이미지일때 결과")
            if png.size[1]>=png.size[0]:
                background = Image.new("RGB", (png.size[1]+10,png.size[1]+10), (255, 255, 255))
                # background에 png에서 투명을 뺀 채널 가져와서 합성하고 300,300으로 맞춰서 테스트 가능하게 만들기.
                background.paste(png, box=(int((png.size[1]+10-png.size[0])/2),5))  # 3 is the alpha channel
            else:
                background = Image.new("RGB", (png.size[0]+10,png.size[0]+10), (255, 255, 255))
                # background에 png에서 투명을 뺀 채널 가져와서 합성하고 300,300으로 맞춰서 테스트 가능하게 만들기.
                background.paste(png, box=(5,int((png.size[0]+10-png.size[1])/2)))  # 3 is the alpha channel
            background = background.resize((500, 500), Image.ANTIALIAS)

            # 합쳐진 이미지를 모델에 넣을 수 있도록 (1,256,256,3) 형식으로 차원을 하나 늘려준다.
            images = np.array(background)
            # print(images.shape)
            cv2.imwrite("foo.png",images) #imwrite 이용하면 실제로 어떻게 보이는지 볼 수 있다.
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


# In[3]:
# 이미지 받아오기
# 인물 이미지 위치 https://www.gettyimagesbank.com/s/?lv=&st=union&mi=2&q=%25EC%259D%25B8%25EB%25AC%25BC%2520%25EB%2593%259C%25EB%25A1%259C%25EC%259E%2589&ssi=go
# ABCDE = "이미지주소받아오기" #형식은 이런식으로
# My_images = get_final_test("ABCDE")

#My_images = get_final_test("https://imagescdn.gettyimagesbank.com/500/21/424/260/0/1316235109.jpg") #예시 이미지 (주소로 바로 넣어주면 테스트 가능)
#My_images = get_final_test("https://imagescdn.gettyimagesbank.com/500/201906/jv11367123.jpg")
#https://imagescdn.gettyimagesbank.com/500/202102/jv12240601.jpg
# 이 이미지가 사람에 맞는듯

correct_list=[]
ai_name_list = []
ai_percent_list = []
for i in range(1,42):
    got_path = "D:/Jupyter/gitCBIR/flask-keras-cnn-image-retrieval-master/eesang/chiyang/그림"+str(i)+".png"
    My_images = get_final_test_png(got_path)
    My_predictions = model.predict(My_images)
    My_predictions_result = My_predictions[0].tolist()
    print(i,"번째 이미지 : 최댓값",first_max(My_predictions), max(My_predictions_result)," 두번째값 :",second_max(My_predictions))

    if first_max(My_predictions)==answer_list[i-1]:
        correct_list.append("correct")
    else:
        pass
    #프론트엔드에 보낼 정보들 모으기(테스트로 보여줄 용도)


    #백엔드에 보낼 정보들 모으기(순서대로)
    ai_name_list.append(got_path)
    ai_percent_list.append(My_predictions_result)
    # print(My_predictions_result)
    # print(group_name)
    if i==6 or i==11 or i==16 or i==21 or i==26 or i==31 or i==36:
        print("-=-=-=-=-=-=-=-=-=-")
    else:
        pass
print(correct_list.count("correct") ,"/ 41",format(int(correct_list.count("correct"))/41,".3f"),"점")


####-----최종 정리된 이미지로 테스트

correct_list=[]
ai_name_list = []
ai_percent_list = []
for i in range(1,25):
    got_path = "D:/Jupyter/gitCBIR/flask-keras-cnn-image-retrieval-master/eesang/chiyang/pick/"+str(i)+".png"
    My_images = get_final_test_png(got_path)
    My_predictions = model.predict(My_images)
    My_predictions_result = My_predictions[0].tolist()
    print(i,"번째 이미지 : 최댓값",first_max(My_predictions), max(My_predictions_result)," 두번째값 :",second_max(My_predictions))

    if first_max(My_predictions)==answer_list[i-1]:
        correct_list.append("correct")
    else:
        pass
    #프론트엔드에 보낼 정보들 모으기(테스트로 보여줄 용도)


    #백엔드에 보낼 정보들 모으기(순서대로)
    got_name= str(i)+".png"
    ai_name_list.append(got_name)
    ai_percent_list.append(My_predictions_result)
    # print(My_predictions_result)
    # print(group_name)
    if (i%3)==0:
        print("-=-=-=-=-=-=-=-=-=-")
    else:
        pass
print(correct_list.count("correct") ,"/ 24",format(int(correct_list.count("correct"))/24,".3f"),"점")
# >>> 18/24 0.75점 

#프론트엔드에 보낼 정보들 순서대로 넣기
lililist=[]
for i in range(1,25):
    if i%3==1:
        lililist.append(ai_name_list[i-1])
for i in range(1,25):
    if i%3==2:
        lililist.append(ai_name_list[i-1])
for i in range(1,25):
    if i%3==0:
        lililist.append(ai_name_list[i-1])
print(lililist)
    # print(ai_name_list[i-1])


#백엔드에 보낼 정보들 정리하기
print(ai_name_list)
print(ai_percent_list)


# In[4]:
# model.predict() 사용해서 점수 구하기. 이걸 쓰면 0.8 이런 퍼센트 나옴
My_predictions = Test_model.predict(My_images)
My_predictions_result = My_predictions[0].tolist()
print(My_predictions_result)

#group_name은 위치 비교하라고 넣어둠
print(group_name)

#-----------첫번째 큰값
print(first_max(My_predictions))

#-----------두번째 큰값
print(second_max(My_predictions))

#-----------세번째 큰값
# third_max(My_predictions)
