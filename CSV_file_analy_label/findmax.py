# -*- coding: utf-8 -*-
# Author: Shin
import pandas as pd
from glob import glob


##############파일 합치기만
namecntlist =[]
standards = glob('D:/Jupyter/gitCBIR/flask-keras-cnn-image-retrieval-master/resultcsv/allresult/*.csv')

for i in standards:
    i = i.replace("D:/Jupyter/gitCBIR/flask-keras-cnn-image-retrieval-master/resultcsv/allresult\\result","")
    i = i.replace(".csv","")
    print(i)
    namecntlist.append(i)
print(namecntlist)

totaldf = pd.DataFrame({'ID': 1}, index=[1])

for i in namecntlist:
    dfmerge1 = pd.read_csv("D:/Jupyter/gitCBIR/flask-keras-cnn-image-retrieval-master/resultcsv/allresult/result" + str(i) + ".csv")
    #print(dfmerge1.head())
    dfmerge1 = dfmerge1.drop(columns='Unnamed: 0')
    totaldf= pd.merge(totaldf,dfmerge1, on='ID', how='outer')
    print(i)

print(totaldf.head())

"""
for i in range(9816,9999):
    start = time.time
    dfmerge1 = pd.read_csv("D:/Jupyter/gitCBIR/flask-keras-cnn-image-retrieval-master/resultcsv/resultb\'" + str(i) + ".jpg\'.csv")
    dfmerge1 = dfmerge1.drop(columns='Unnamed: 0')

    totaldf= pd.merge(totaldf,dfmerge1, on='ID', how='outer')
    print(i)

print(totaldf.head())
""" #숫자로 번호 합치기

"""
for i in range(1,13):
    dfmerge1 = pd.read_csv('D:/Jupyter/gitCBIR/flask-keras-cnn-image-retrieval-master/resultcsv/results'+str(i)+'.csv')
    dfmerge1 = dfmerge1.drop(columns='Unnamed: 0')

    totaldf= pd.merge(totaldf,dfmerge1, on='ID', how='outer')
"""



totaldf.to_csv("totaldata.csv", encoding='utf-8')
print(totaldf)

######정규화 및 변경
from sklearn.preprocessing import MinMaxScaler

#df = pd.read_csv("totaldata.csv")
#df = df.drop(["Unnamed: 0"],axis=1)
#y_data = df['ID']
#df = df.drop(["ID"],axis=1)

df_test = pd.read_csv("totaldata.csv")
df_test = df_test.drop(["Unnamed: 0"],axis=1)
y_data_test = df_test['ID']
#df = df.drop(["ID"],axis=1)

"""
#정규화
scaler = MinMaxScaler()
#df[:] = scaler.fit_transform(df[:])
#df= pd.concat([y_data,df],axis=1)
df_test[:] = scaler.fit_transform(df_test[:])
df_test= pd.concat([y_data_test,df_test],axis=1)
"""#정규화 하기

"""
#mindf 는 Min 값만 찾아둔것.
df_sorted_by_values = df_test.sort_values(by='ID',ascending=True)
mindf = df_sorted_by_values.min(axis=1)

dfvalmin = pd.concat([df_sorted_by_values,mindf],axis=1)
dfvalmin = dfvalmin.rename(columns={0:"MIN"})
""" # min값 찾기

#maxdf 는 Max값만 찾아둔것. 21년 4월 26일 작성
df_sorted_by_values = df.sort_values(by='ID',ascending=True)
y_data = df_sorted_by_values['ID']

df_sorted_by_values = df_sorted_by_values.drop(["ID"],axis=1)
print(df_sorted_by_values)

#df_sorted_by_values_4_findmax = df_sorted_by_values[df_sorted_by_values.between(0,1, inclusive=False)]
maxdf = df_sorted_by_values.max(axis=1)

dfvalmax = pd.concat([df_sorted_by_values,maxdf],axis=1)
dfvalmax = pd.concat([y_data,dfvalmax],axis=1)
dfvalmax = dfvalmax.rename(columns={0:"MAX"})

#y_data는 리스트화 하여 다시쓰자
y_data = pd.Series.tolist(y_data)
print(df)
print(maxdf)
print(dfvalmax)

print(y_data)
print(len(y_data))

#------------------------------------------------------------------------------------------
import time
import pandas as pd
#시간 측정
start = time.time()

namelist =[]
special_list = []

for keys in dfvalmax.keys():
    namelist.append(keys)
print(namelist)
print(maxdf)
print(len(maxdf))
print(dfvalmax)

"""
for k in range(len(maxdf)):
    for i in range(len(namecntlist)):  #namecntlist는 특성의 갯수 현재 총 60여가지의 특성
        if dfvalmax.iloc[k][i]==dfvalmax.iloc[k][-1]:
            print("DataFrame 내의",k+1,"번째 데이터는",i,"번째 특성")
            print("사진 폴더 내의",y_data[k],"번째 사진")
            print(dfvalmax.iloc[k][i],dfvalmax.iloc[k][-1])
            print(namelist[i])
            print("------------------------------------------")
            special_list.append(namelist[i])
    print("=============")
"""# len(namecntlist)를 가져오면 안되는 이유는 앞에서부터 가져오는데 ID값을 포함한 dfvalmax에서 찾기때문에 맨 뒤에 하나를 못찾게 된다. 따라서 직접입력한다.

for k in range(len(maxdf)):
    for i in range(1,59):  #namecntlist는 특성의 갯수 현재 총 60여가지의 특성 맨앞과 맨뒤만 빼고 같은 것을 찾아준다고 생각. (ID값, MAX 이부분을 버려준다.)
        if dfvalmax.iloc[k][i]==dfvalmax.iloc[k][-1]:
            print("DataFrame 내의",k+1,"번째 데이터는",i,"번째 특성")
            print("사진 폴더 내의",y_data[k],"번째 사진")
            print(dfvalmax.iloc[k][i],dfvalmax.iloc[k][-1])
            print(namelist[i])
            print("------------------------------------------")
            special_list.append(namelist[i])
    print("=============")

#print(namelist)
print(len(namelist))
#print(special_list)
print(len(special_list))

print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간

#######################################################################

for keynames in dfvalmax.keys():
    print(keynames)
    kkkkk = special_list.count(str(keynames))
    print(kkkkk)




