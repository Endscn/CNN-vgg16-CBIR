# -*- coding: utf-8 -*-
# Author: Shin

import pandas as pd
from glob import glob


##############파일 합치기만
totaldf = pd.DataFrame({'ID': b'1.jpg'}, index=[1])
namelist =[]
standards = glob('D:/Jupyter/gitCBIR/flask-keras-cnn-image-retrieval-master/resultcsv/*.csv')

for i in standards:
    i = i.replace("D:/Jupyter/gitCBIR/flask-keras-cnn-image-retrieval-master/resultcsv\\resultb'","")
    i = i.replace("\'.csv","")
    print(i)
    namelist.append(i)
print(namelist)

for i in namelist:
    dfmerge1 = pd.read_csv("D:/Jupyter/gitCBIR/flask-keras-cnn-image-retrieval-master/resultcsv/resultb\'" + str(i) + "\'.csv")
    dfmerge1 = dfmerge1.drop(columns='Unnamed: 0')

    totaldf= pd.merge(totaldf,dfmerge1, on='ID', how='outer')
print(totaldf)


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

df_test = pd.read_csv("totaldata_test.csv")
df_test = df_test.drop(["Unnamed: 0"],axis=1)
y_data_test = df_test['ID']
df_test = df_test.drop(["ID"],axis=1)

"""
#정규화
scaler = MinMaxScaler()
#df[:] = scaler.fit_transform(df[:])
#df= pd.concat([y_data,df],axis=1)
df_test[:] = scaler.fit_transform(df_test[:])
df_test= pd.concat([y_data_test,df_test],axis=1)
"""

#mindf 는 Min 값만 찾아둔것.
df_sorted_by_values = df_test.sort_values(by='ID',ascending=True)
mindf = df_sorted_by_values.min(axis=1)

dfvalmin = pd.concat([df_sorted_by_values,mindf],axis=1)
dfvalmin = dfvalmin.rename(columns={0:"MIN"})

#print(df)
#print(mindf)
#print(dfvalmin)
