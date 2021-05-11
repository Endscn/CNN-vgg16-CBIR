# -*- github Endscn -*-
# Author: Shin

import cv2
import numpy as np


img_path = "https://cdn.pixabay.com/photo/2021/04/23/00/39/search-png-clipart-6200457_960_720.png" # 2채널이면서 흑백사진인것
img_path = "https://we2d-app.s3.ap-northeast-2.amazonaws.com/designedImg/1613140072231.png" # 4캘리그래피
img_path = "https://we2d-app.s3.ap-northeast-2.amazonaws.com/designedImg/1620271223431.png" #보라색 풍경같은거
#img = Image.open(requests.get(img_path, stream=True).raw)

img_path = "https://cdn.pixabay.com/photo/2020/11/01/13/03/lemon-5703655_960_720.jpg" #레몬패턴

img_path = "https://we2d-app.s3.ap-northeast-2.amazonaws.com/designedImg/1620315690958.png"  #다이아몬드같으거

img_path = "https://qquing.net/data/upload/manga/2021_05_17940bc719b4da2fa.jpg" #3채널 공룡캐릭터


del img_path
del png
# png에 이미지 불러오기
png = Image.open(requests.get(img_path, stream=True).raw)
print(png.split())
png_nparray = np.array(png)
print(png_nparray.shape)
try:
    if png_nparray.shape[2] == 3:
        background = Image.new("RGB", png.size, (255, 255, 255))

        # background에 png에서 투명을 뺀 채널 가져와서 합성하고 300,300으로 맞춰서 테스트 가능하게 만들기.
        background.paste(png)  # 3 is the alpha channel
        background = background.resize((300, 300), Image.ANTIALIAS)

        # 합쳐진 이미지를 모델에 넣을 수 있도록 (1,256,256,3) 형식으로 차원을 하나 늘려준다.
        images = np.array(background)
        images = images[:, :, ::-1].copy()
        cv2.imwrite("foo.png",images)
        images = images.astype('float32') / 255
        images = (np.expand_dims(images, 0))
        print("3채널 이미지입니다.")


    elif png_nparray.shape[2] == 4:

        background = Image.new("RGB", png.size, (255, 255, 255))

        # background에 png에서 투명을 뺀 채널 가져와서 합성하고 300,300으로 맞춰서 테스트 가능하게 만들기.
        background.paste(png, mask=png.split()[3])  # 3 is the alpha channel
        background = background.resize((300, 300), Image.ANTIALIAS)

        # 합쳐진 이미지를 모델에 넣을 수 있도록 (1,256,256,3) 형식으로 차원을 하나 늘려준다.
        images = np.array(background)
        images = images[:, :, ::-1].copy()
        cv2.imwrite("foo.png",images)
        images = images.astype('float32') / 255
        images = (np.expand_dims(images, 0))
        print("4채널 이미지입니다.")
    else:
        png.shape()
except:
    print("2채널 이미지 입니다.")

    background = Image.new("RGB", png.size, (255, 255, 255))

    # background에 png에서 투명을 뺀 채널 가져와서 합성하고 300,300으로 맞춰서 테스트 가능하게 만들기.
    background.paste(png, mask=png.split()[1])  # 3 is the alpha channel
    background = background.resize((300, 300), Image.ANTIALIAS)

    # 합쳐진 이미지를 모델에 넣을 수 있도록 (1,256,256,3) 형식으로 차원을 하나 늘려준다.
    images = np.array(background)
    cv2.imwrite("foo.png",images)
    images = images.astype('float32') / 255
    images = (np.expand_dims(images, 0))
    
#시간 오래안걸리는지 확인하기.
