# -*- github Endscn -*-
# Author: Shin
# 참조 : https://www.python2.net/questions-591096.htm
# 데이터 프레임 내부에서 유사값을 찾는 방법 내 데이터에 맞게 수정하기--

import pandas as pd
import numpy as np

# 랜덤시드 고정
np.random.seed(0)


# 더미데이터 만들기
N = 100
df = pd.DataFrame(data=np.random.choice(range(5), size=(N, 8)))
df.columns = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

#유사행 찾는 함수 정의
def similar_rows(idx, row, df):
    mask = np.logical_and.reduce([
        df['a'] == row['a'],
        abs(df['b'] - row['b']) <= 1,
        df['h'] == (3 - row['h'])
    ])
    df_tmp = df.loc[mask, :]
    df_tmp.insert(0, 'original_index', idx)
    return df_tmp


# 결과 생성
df_new = pd.concat([similar_rows(idx, row, df) for idx, row in df.iterrows()])
df_new.reset_index(inplace=True)
df_new.rename({'index': 'similar_index'}, axis=1, inplace=True)
print(df_new.head(10))

#----------------------------------------------------------
# 여기는 임의로 선택한 행에 대해서 근처를 보여주는 함수
# 랜덤row 생성
row = df.loc[np.random.choice(N), :]
print('Randomly Selected Row:')
print(pd.DataFrame(row).T)

# 임의로 생성한 유사한 행에 대해서 마스크 생성,적용
mask = np.logical_and.reduce([
    df['a'] == row['a'],
    abs(df['b'] - row['b']) <= 1,
    df['h'] == (3 - row['h'])
])

print('"Similar" Results:')
df_filtered = df.loc[mask, :]
print(df_filtered)

