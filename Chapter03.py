#####################################################################

# Chapter 3 파이썬을 이용한 데이터 분석

#####################################################################

import numpy as np
import scipy as sp
from scipy import stats     # 4분위수
import pandas as pd         # 수치 계산에 사용하는 라이브러리
import seaborn as sns
sns.set()       # sns.set()을 하면 그래프 디자인이 바뀜
from matplotlib import pyplot as plt        # 그래프를 그리기 위한 라이브러리

pd.set_option("display.precision", 3)       # 표시 자릿수 지정 : 소수점 이하 자릿수 3
# %precision 3                                # 표시자릿수 지정

# %matplotlib inline                          # 그래프를 주피터 노트북에 그리기 위한 설정



#####################################################################

fish_data = np.array([2,3,3,4,4,4,4,5,5,6])
fish_data_2 = np.array([2,3,3,4,4,4,4,5,5,100])
fish_data_3 = np.array([1,2,3,4,5,6,7,8,9])

fish_multi = pd.read_csv("source/sample/3-2-1-fish_multi.csv")
shoes = pd.read_csv("source/sample/3-2-2-shoes.csv")
cov_data = pd.read_csv("source/sample/3-2-3-cov.csv")
fish_multi = pd.read_csv("source/sample/3-3-2-fish_multi_2.csv")

#####################################################################

# p112  3.1.2   1변량 데이터의 관리 : 1가지 종류의 데이터
fish_data

# p112  3.1.3   합계와 샘플사이즈
sp.sum(fish_data)
len(fish_data)

# p113  3.1.4   평균값(기댓값)
N = len(fish_data)
sum_value = sp.sum(fish_data)
mu = sum_value / N
mu

sp.mean(fish_data)

# p113  3.1.5   표본분산
sigma_2_sample = sp.sum((fish_data - mu) ** 2) / N
sigma_2_sample

sp.var(fish_data, ddof = 0)

# p115  3.1.6   불편분산
# 표본분산은 표본의 평균을 사용해서 분산을 계산한 값
# 이 값은 분산을 과소추정하는 경향이 있어 그것을 없애기 위한 것이 불편분산
sigma_2 = sp.sum((fish_data - mu) ** 2) / (N - 1)
sigma_2

sp.var(fish_data, ddof = 1)

# p116  3.1.7   표준편차
# 분산은 데이터를 제곱하여 계산
# 때문에 단위 등도 제곱이 되는데 이 상태로는 계산이 불편함
# 그래서 분산에 루트를 취한 표준편차를 사용
sigma = sp.sqrt(sigma_2)
sigma

sp.std(fish_data, ddof = 1) # 불편분산에 루트값을 취해서 표준편차를 구하는 경우
sp.std(fish_data, ddof = 0) # 표본분산에 루트값을 취함

# p117  3.1.8   표준화
# 데이터의 평균을 0으로, 표준편차(분산)를 1로 하는 변환을 표준화라고 함
# 여러 변수를 다룰 때 큰 변수와 작은 변수가 섞여 있으면 다루기 어려우므로 표준화로 데이터를 비교하기 쉽게 함
fish_data - mu
sp.mean(fish_data - mu)

# 데이터의 표준편차(분산)을 1로 만드는 방법은 데이터를 일률적으로 표준편차로 나누는 것
fish_data / sigma
sp.std(fish_data / sigma, ddof = 1)

standard = (fish_data - mu) / sigma
standard

sp.mean(standard)       # 2.220446049250313e-16
np.sum((fish_data - mu) / sigma)
np.mean(standard)
sp.std(standard, ddof = 1)


# p118  3.1.9   그 외의 통계량
sp.amax(fish_data)  # 최대값
np.amax(fish_data)

sp.amin(fish_data)  # 최소값
np.amin(fish_data)

sp.median(fish_data)    # 중앙값
np.median(fish_data)

# 평균값, 중앙값 차이가 날 때(극단적으로 큰 물고기 한 마리 포함)
sp.mean(fish_data_2)
np.mean(fish_data_2)
sp.median(fish_data_2)
np.median(fish_data_2)


# p120  3.1.10  scipy.stats와 사분위수
# 데이터를 순서대로 늘어놓았을 때 아래에서부터 25%, 75%에 해당하는 값
len(fish_data_3)
stats.scoreatpercentile(fish_data_3, 25)    # 25%에 해당하는 값
stats.scoreatpercentile(fish_data_3, 75)    # 75%에 해당하는 값




##### 3.2 파이썬을 이용한 기술통계: 다변량 데이터
# 여러 개의 변수를 조합한 데이터: 다변량 데이터
# 깔끔한 데이터(Tidy data) 형식으로 정리하는 것이 중요함

# p121  3.2.1   깔끔한 데이터
"""
    깔끔한 데이터의 특징
        1. 개별값이 하나의 셀을 이룬다.
        2. 개별 변수가 하나의 열을 이룬다.
        3. 개별 관측이 하나의 행을 이룬다.
        4. 개별 관측 유닛 유형이 하나의 표를 이룬다.
        
    Tidy data를 사용할 때의 장점
        복잡한 데이터를 모을 때 통일성 있는 처리를 할 수 있다.
    
    Tidy data는 소프트웨어로 다루기 쉬운 형식
        사람이 눈으로 보고 바로 판단할 수 있는 데이터 <> 소프트웨어로 다루기 쉬운 데이터 형식

"""

# p122  3.2.2   지저분한 데이터

# p122  3.2.3   교차분석표
# 깔끔한 데이터는 '행 하나에 1개의 결과'가 있도록 정리
# 교차분석표는 '행이 변수의 의미를 갖는' 경향이 있음
"""
    디지털 환경에서 데이터 분석할 때는 Tidy data로
    (필요할 때) 파이썬 코드 등으로 교차분석표로 변환하는 것이 좋음
    
    그렇지 않으면 나중에 데이터를 별도로 정리할 때 예상보다
    많은 시간과 노력이 들어간다는 사실
"""

# p123  3.2.4   다변량 데이터 관리하기
fish_multi = pd.read_csv("source/sample/3-2-1-fish_multi.csv")
print(fish_multi)


# p124  3.2.5   그룹별 통계량 계산하기
group = fish_multi.groupby("species")
print(group.mean())
print(group.std(ddof = 1))
group.describe()


# p125  3.2.6   교차분석표 구현하기
shoes = pd.read_csv("source/sample/3-2-2-shoes.csv")
print(shoes)
cross = pd.pivot_table(
    data        = shoes     ,       # 데이터를 지정
    values      = "sales"   ,       # 데이터를 모을 열 지정
    aggfunc     = "sum"     ,       # 데이터를 모을 함수 지정
    index       = "store"   ,       # 교차분석표의 행 지정
    columns     = "color"           # 교차분석표의 열 지정
)
print(cross)


# p126  3.2.7   공분산
# p127  3.2.8   분산-공분산 행렬
# p128  3.2.9   공분산(실습)
cov_data = pd.read_csv("source/sample/3-2-3-cov.csv")
print(cov_data)

# 데이터 분리
x = cov_data["x"]
y = cov_data["y"]
# 표본 크기
N = len(cov_data)
# 평균값 계산
mu_x = sp.mean(x)
mu_y = sp.mean(y)
# 공분산 계산
cov_sample = sum((x - mu_x) * (y - mu_y)) / N
cov_sample
cov = sum((x - mu_x) * (y - mu_y)) / (N - 1)
cov


# p129  3.2.10  분산-공분산 행렬(실습)
sp.cov(x, y, ddof = 0)
sp.cov(x, y, ddof = 1)


# p129  3.2.11  피어슨 상관계수
# 공분산을 -1 ~ 1 사이가 되도록 표준화
# 공분산은 편리한 지표지만 최대값이나 최소값이 얼마가 될 지 알 수가 없음
#  예) 단위가 cm에서 m으로 변하면 공분산 값도 변함


# p130  3.2.12  상관행렬
# p130  3.2.13  피어슨 상관계수(실습)
# 분산 계산
sigma_2_x = sp.var(x, ddof = 1)
sigma_2_y = sp.var(y, ddof = 1)
# 표준편차 계산
sigma_x = sp.std(x, ddof = 1)
sigma_y = sp.std(y, ddof = 1)
# 상관계수
rho = cov / sp.sqrt(sigma_2_x * sigma_2_y)
rho
rho_ = cov / (sigma_x * sigma_y)
rho_
# 상관행렬 계산
sp.corrcoef(x, y)




### p132    3.3     matplotlib과 seaborn을 이용한 데이터 시각화
"""
    matplotlib  : 그래프를 그리는 표준 라이브러리
    seaborn     : matplotlib의 그래프를 더 예쁘게 그리기 위한 라이브러리
"""

# p132  3.3.2   시각화를 위한 준비
# 수치 계산에 사용하는 라이브러리
import numpy as np
import pandas as pd
# 표시 자리수 지정
pd.set_option("display.precision", 3)
# 그래프를 그리기 위한 라이브러리
from matplotlib import pyplot as plt
# 그래프를 주피터 노트북에 그리기 위한 설정
# %matplotlib inline


# p133  3.3.3   pyplot을 이용한 꺾은선 그래프
x = np.array([0,1,2,3,4,5,6,7,8,9])
y = np.array([2,3,4,3,5,4,6,7,4,8])

plt.plot(x, y, color = 'black')
plt.title("lineplot matplotlib")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
plt.ion()
plt.pause(1)

# p134  3.3.4   seaborn과 pyplot을 이용한 꺾은선 그래프
import seaborn as sns
sns.set()       # sns.set()을 하면 그래프 디자인이 바뀜
# %matplotlib inline
plt.plot(x, y, color = 'black')
plt.title("lineplot seaborn")
plt.xlabel("x")
plt.ylabel("y")
plt.show()


# p135  3.3.5   seaborn을 이용한 히스토그램
fish_data = np.array([2,3,3,4,4,4,4,5,5,6])
fish_data

sns.distplot(fish_data, bins = 5, color = 'black', kde = False)


# p136  3.3.6   커널밀도추정에 따른 히스토그램 평활화
sns.distplot(fish_data, bins = 1, color = 'black', kde = False)
sns.distplot(fish_data, color = 'black')
sns.distplot(fish_data, color = 'black', norm_hist = True)


# p137  3.3.7   2변량 데이터에 대한 히스토그램
fish_multi = pd.read_csv("source/sample/3-3-2-fish_multi_2.csv")
fish_multi
fish_multi.groupby("species").describe()
length_a = fish_multi.query("species == 'A'")["length"]
length_b = fish_multi.query("species == 'B'")["length"]

sns.distplot(length_a, bins = 5, color = 'black', kde = False)
sns.distplot(length_b, bins = 5, color = 'gray', kde = False)


# p139  3.3.8   다변량 데이터를 시각화하는 코드 작성
"""
   sns.함수명(
    x = "x축의 열 이름",
    y = "y축의 열 이름",
    data = 데이터프레임,
    그_외의_인수
    ) 
"""


# p140  3.3.9   상자그림
# 종류별 물고기 몸길이 등 '카테고리 변수 * 수치형 변수' 조합의 데이터를 표시해야할 경우
# 상자그림 = boxplot
sns.boxplot(x = "species", y = "length", data = fish_multi, color = "gray")
fish_multi.groupby("species").describe()



# p141  3.3.10  바이올린플롯
# 바이얼린플롯은 상자그림의 상자 대신 커널밀도추정의 결과를 사용한 것
# 상자 대신 히스토그램을 세로로 세워서 배치한 상자그림
# 어느 부분에 데이터가 집중되어 있는지(도수가 어떤지) 정보가 추가되어 있음
sns.violinplot(x = "species", y = "length",
               data = fish_multi, color = "gray")


# p141  3.3.11  막대그래프
# 막대그래프는 seaborn의 barplot 함수 사용
sns.barplot(x = "species", y = "length",
            data = fish_multi, color = "gray")


# p142  3.3.12  산포도
# '수치형 변수 * 수치형 변수' 조합 그래프
# sns.jointplot : 산포도와 함께 히스토그램도 붙어 있는 그래프
cov_data = pd.read_csv("source/sample/3-2-3-cov.csv")
print(cov_data)
sns.jointplot(x = "x", y = "y", data = cov_data, color = "black")


x1 = cov_data["x"].astype(float)
y1 = cov_data["y"].astype(float)

df = pd.DataFrame(data = {"x": x1, "y":y1})
print(df.dtypes)
print(cov_data.dtypes)

sns.jointplot(x = "x", y = "y", data = df, color = "black")

x1 = cov_data["x"].astype(float)
y1 = cov_data["y"].astype(float)

df1 = pd.DataFrame()
df1['x'] = x1
df1['y'] = y1
sns.jointplot(data=df1, x='x', y='y')
print(df1.dtypes)

#################################################
# 위 에러로 인한 참고
# https://github.com/mwaskom/seaborn/issues/2677
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


df = pd.DataFrame()
df['x'] = np.random.normal(2, 0.5, 200)
df['y'] = np.linspace(0, 1, 200)
df['group'] = np.random.choice(3, size=200)


g = sns.jointplot(data=df, x='x', y='y', hue='group')
print(g.ax_joint.get_legend_handles_labels())

g = sns.jointplot(data=df, x='x', y='y', hue='group', kind='kde')
print(g.ax_joint.get_legend_handles_labels())

#################################################


# p144  3.3.13  페어플롯
# 많은 양의 변수를 가지고 있는 데이터를 대상으로 그래프를 그리는 방법
# 2개 이상의 변수를 모아서 정리하여 표시하는 방법

# 데이터 읽기
iris = sns.load_dataset("iris")
iris.head(n = 3)
# 붓꽃의 종류별, 특징별 평균값
iris.groupby("species").mean()
sns.pairplot(iris, hue = "species", palette = "gray")


### p146    3.4 모집단에서 표본 추출 시뮬레이션

# p146  3.4.1   라이브러리 임포트
# 수치 계산에 사용하는 라이브러리
import numpy as np
import pandas as pd
import scipy as sp
from scipy import stats
# 그래프를 그리기 위한 라이브러리
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()
# 표시자릿수 지정
# %precision 3
# 그래프를 주피터 노트북에 그리기 위한 설정
# %matplotlib inline


# p146  3.4.2   표본을 얻는 프로세스


# p147  3.4.3   5마리 물고기의 예
fish_5 = np.array([2,3,4,5,6])
fish_5

np.random.choice(fish_5, size = 1, replace = False)     # 1마리 샘플링, 2번 이상 선택x
np.random.choice(fish_5, size = 3, replace = False)     # 3마리 샘플링, 2번 이상 선택x

# 난수 시드(seed, 씨앗)를 이용한 시뮬레이션
# 난수 시드를 지정하면 매번 같은 데이터가 랜덤하게 선택되게 할 수 있음
np.random.seed(1)
np.random.choice(fish_5, size = 3, replace = False)

np.random.seed(1)
sp.mean(
    np.random.choice(fish_5, size = 3, replace = False)
)


# p149  3.4.4   난수
# 난수: 랜덤으로 골라낸 값
# 옛날에는 주사위를 굴리거나 동전을 던져서 난수 생성
# 표본추출 시뮬레이션은 일종의 난수 생성 작업이라고도 할 수 있음


# p149  3.4.5   복원추출과 비복원추출
# 복원추출: 추출된 표본을 다시 모집단에 되돌려놓음으로써 다시 추출될 수 있게 하는 샘플링 방법
# 비복원추출: 추출된 표본을 모집단에 다시 돌려놓지 않음(replace = False)


# p149  3.4.6   더 많은 물고기가 있는 호수에서 표본추출
fish_100000 = pd.read_csv("source/sample/3-4-1-fish_length_100000.csv")["length"]
fish_100000

len(fish_100000)

sampling_result = np.random.choice(
    fish_100000, size = 10, replace = False)        # 10마리 샘플링
sampling_result

sp.mean(sampling_result)    # 표본평균


# p151  3.4.7   모집단분포
sp.mean(fish_100000)        # 모평균
sp.std(fish_100000)         # 모표준편차
sp.var(fish_100000)         # 모분산
sns.distplot(fish_100000, kde = False, color = 'black')


# p152  3.4.8   모집단분포와 정규분포 간 확률밀도함수 비교
# 모집단의 히스토그램과 '평균 4, 분산 0.64인 정규분포'의 확률밀도 비교
x = np.arange(start = 1, stop = 7.1, step = 0.1)
x

# 확률밀도는 stats.norm.pdf 함수를 사용해서 계산 가능
stats.norm.pdf(x = x, loc = 4, scale = 0.8)     # 인수 loc: 평균값, scale: 표준편차

# 평균 4, 분산 0.64(표준편차 0.8)인 정규분포의 확률밀도
plt.plot(x,
         stats.norm.pdf(x = x, loc = 4, scale = 0.8),
         color = 'black')

# 위의 정규분포와 모집단의 히스토그램의 그래프 겹쳐서 그리기
sns.distplot(fish_100000, kde = False, norm_hist = True, color = 'black')
# sns.distplot에 norm_hist = True를 지정하면 면적이 1이 되는 히스토그램이 된다
plt.plot(x,
         stats.norm.pdf(x = x, loc = 4, scale = 0.8),
         color = 'black')
 

# p154  3.4.9   표본을 얻는 절차의 추상화
# 위의 모집단 분포는 평균 4, 분산 0.64(표준편차 0.8)인 정규분포라고 볼 수 있다.
# 그러므로 모집단에서의 표본 추출은 정규분포를 따르는 난수 생성과 거의 같다고 볼 수 있음
# 그래서 이번에는 시작부터 정규분포를 따르는 난수 생성 함수인 stats.norm.rvs를 사용
sampling_norm = stats.norm.rvs(loc = 4, scale = 0.8, size = 10)     # 평균(loc), 표준편차(scale), 샘플사이즈(size)
sampling_norm

sp.mean(sampling_norm)      # 표본평균


# p155  3.4.10  유한모집단추정
# p156  3.4.11  모집단분포를 정규분포로 가정해도 좋은가
































