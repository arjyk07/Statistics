#####################################################################

# Chapter 2 파이썬과 주피터 노트북 기초

#####################################################################

  
# Chapter 2 파이썬과 주피터 노트북 기초
# p92   2.3.8   if문을 사용한 분기
data = 1
if(data < 2):
    print("2보다 작은 데이터입니다.")
else:
    print("2 이상인 데이터입니다.")

# p93   2.3.9   for 문을 사용한 반복 실행
range(0, 3)
for i in range(0, 3):
    print(i)
for i in range(0, 30):
    print("hello")


# p95   2.4 numpy와 pandas 기본
import numpy as np
import pandas as pd

# p97   2.4.5   배열
sample_array = np.array([1,2,3,4,5])
sample_array
sample_array_2 = np.array([[1,2,3,4,5], [6,7,8,9,10]])
sample_array_2
sample_array_2.shape    # 2행 5열

# p98   2.4.6   등차수열을 만드는 방법
np.arange(start = 1, stop = 6, step = 1)
np.arange(start = 0.1, stop = 0.8, step = 0.2)
np.arange(0.1, 0.8, 0.2)

# p99   2.4.7   여러가지 배열을 만드는 방법
np.tile("A", 5)
np.tile(0, 4)
np.zeros(4)
np.zeros([2,3])
np.ones(3)

# p102  2.4.9   데이터프레임
sample_df = pd.DataFrame({
    'col1'  : sample_array,
    'col2'  : sample_array*2,
    'col3'  : ["A", "B", "C", "D", "E"]
})
print(sample_df)
sample_df

# p104  2.4.11  데이터프레임 병함
df_1 = pd.DataFrame({
    'col1'  : np.array([1,2,3]),
    'col2'  :np.array(["A", "B", "C"])
})
df_2 = pd.DataFrame({
    'col1'  : np.array([4,5,6]),
    'col2'  : np.array(["D", "E", "F"])
})

print(pd.concat([df_1, df_2]))
print(pd.concat([df_1, df_2], axis=1))

# p105  2.4.12  데이터프레임 열에 대해 작업하기
print(sample_df)
print(sample_df.col2)
print(sample_df["col2"])
print(sample_df[["col2", "col3"]])
print(sample_df.drop("col1", axis=1))

# p107  2.4.13  데이터프레임 행에 대해 작업하기
print(sample_df.head(n = 3))
print(sample_df.query('index == 0'))
print(sample_df.query('col3 == "A"'))
print(sample_df.query('col3 == "A" | col3 == "D"'))
print(sample_df.query('col3 == "A" & col1 == 3'))
print(sample_df.query('col3 == "A"')[["col2", "col3"]])