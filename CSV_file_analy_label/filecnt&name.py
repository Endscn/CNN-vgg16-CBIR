# -*- coding: utf-8 -*-
# Author: Shin

import time
import pandas as pd
#시간 측정
start = time.time()

namelist =[]
trimlist = []
special_list = []

for keys in dfvalmax.keys():
    namelist.append(keys)
print(namelist)

for k in range(len(maxdf)):         #실제 실행할때는 전체 사용len(maxdf) 이부분은 정확한 파일명이 아니라 순서를 찾아서 해주는것 (~~.png 파일을 의미하는것이 아니라 df상 순서를 의미하는것)
    for i in range(len(namecntlist)):
        if dfvalmax.iloc[k][i]==dfvalmax.iloc[k][-1]:
            print("DataFrame 내의",k+1,"번째 사진은",i,"번째 특성")
            print(dfvalmax.iloc[k][i],dfvalmax.iloc[k][-1])
            print(namelist[i])
            special_list.append(namelist[i])
    print("=============")
print(special_list)

print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간


for i in special_list:
  #처음부터 b''제거하고 오게 수정하기
    i = i.replace("b\'","")
    i = i.replace(".jpg\'SCORE","")
    if int(i) in character_list:
        trimlist.append("character_list")
    elif int(i) in flower_list:
        trimlist.append("flower_list")
    elif int(i) in widepatt_list:
        trimlist.append("widepatt_list")
    elif int(i) in dreamystical_list:
        trimlist.append("dreamystical_list")
    elif int(i) in densepatt_list:
        trimlist.append("densepatt_list")
    elif int(i) in person_list:
        trimlist.append("person_list")
    elif int(i) in coloredwall_list:
        trimlist.append("coloredwall_list")
    elif int(i) in scenery_list:
        trimlist.append("scenery_list")
    elif int(i) in calligraphy_list:
        trimlist.append("calligraphy_list")
    elif int(i) in pendrawing_list:
        trimlist.append("pendrawing_list")
print(trimlist)


dfvalmax = pd.concat([,maxdf],axis=1)
dfvalmax = dfvalmax.rename(columns={0:"MAX"})

#-------------각 항목이 몇개를 보유했는지 알아보기

for keynames in dfvalmax.keys():
    print(keynames)
    key_cnt = special_list.count(str(keynames))
    print(key_cnt)


from collections import Counter
print ("---Counter()---")
result = Counter(trimlist)
print (result)

for key in result:
    print (key, result[key])
