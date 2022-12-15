#####################################################################

# Chapter 3 파이썬을 이용한 데이터 분석

#####################################################################

"""
    3.1 ~ 3.3   기술통계
    3.4 ~ 3.6   추측통계
    3.7         추정
    3.8 ~ 3.11  통계적가설검정
"""

# 에러 제거
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd


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
junk_food = pd.read_csv("source/sample/3-8-1-junk-food-weight.csv")["weight"]
paired_test_data = pd.read_csv("source/sample/3-9-1-paired-t-test.csv")

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


### p157    3.5     표본 통계량 성질

# p157  3.5.1   시행
"""
    시행 : 1회의 조사를 수행
    시행횟수 : 몇 번이고 시행을 반복하는 게 가능한 경우 반복한 횟수
"""


# p157  3.5.2   표본분포
"""
    표본분포 : 표본의 통계량이 따르는 확률분포
    표본추출 시뮬레이션 10,000회 → 10,000개 표본(샘플사이즈 x) → 표본평균 10,000개
    → 10,000개의 표본평균이 따르는 확률분포가 표본분포
    <p158 그림 참고>
"""


# p158  3.5.3   라이브러리 임포트
# 수치계산에 사용하는 라이브러리
import numpy as np
import pandas as pd
import scipy as sp
from scipy import stats
# 그래프를 그리기 위한 라이브러리
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()
# 표시 자리수 지정
# %precision 3
# 그래프를 주피터 노트북에 그리기 위한 설정
# %matplotlib inline

population = stats.norm(loc = 4, scale = 0.8)   # 평균 4, 표준편차 0.8(분산 0.64)인 정규분포


# p159  3.5.4   표본평균을 여러 번 계산하기
sample_mean_array = np.zeros(10000)     # 표본평균 10,000개 얻기, 길이가 10,000개인 배열
np.random.seed(1)       # 이 배열에 10,000개의 표본평균 저장
for i in range(0, 10000):
    sample = population.rvs(size = 10)
    sample_mean_array[i] = sp.mean(sample)


# p160  3.5.5   표본평균의 평균값은 모평균에 가깝다
sp.mean(sample_mean_array)      # 표본평균의 평균
sp.std(sample_mean_array)       # 표본평균의 표준편차
sns.distplot(sample_mean_array, color = "black")        # 표본평균의 히스토그램


# p161  3.5.6   샘플사이즈가 크면 표본평균은 모평균에 가까워진다
"""
    대상 : 표본평균
    변화시키는 것 : 샘플사이즈
    알고 싶은 것 : 샘플사이즈가 커질수록 표본평균은 모평균에 가까워지는가?
"""
size_array = np.arange(start = 10, stop = 100100, step = 100)   # 10 ~ 100,010까지 100단위 변화하는 샘플사이즈
size_array

sample_mean_array_size = np.zeros(len(size_array))  # 표본평균을 저장할 변수
# 시뮬레이션 실행(표본평균을 구하는 시행을 샘플사이즈를 변화시켜가면서 몇 번이고 실행)
np.random.seed(1)
for i in range(0, len(size_array)):
    sample = population.rvs(size = size_array[i])
    sample_mean_array_size[i] = sp.mean(sample)

# 가로축이 샘플사이즈, 세로축이 표본평균인 그래프 그리기
plt.plot(size_array, sample_mean_array_size, color = "black")
plt.xlabel("sample size")
plt.ylabel("sample mean")
# → 샘플사이즈가 커질수록 표본평균이 모평균(4)에 가까워진다


# p163  3.5.7   표본평균을 몇 번이고 계산하는 함수 만들기
def calc_sample_mean(size, n_trial):
    sample_mean_array = np.zeros(n_trial)
    for i in range(0, n_trial):
        sample = population.rvs(size = size)
        sample_mean_array[i] = sp.mean(sample)
    return(sample_mean_array)

np.random.seed(1)
sp.mean(calc_sample_mean(size = 10, n_trial = 10000))       # 4.004202422791747


# p163  3.5.8   샘플사이즈를 바꿨을 때 표본평균의 분산
# 샘플사이즈를 바꿨을 때 표본평균의 분포를 바이올린플롯을 이용해서 확인
np.random.seed(1)
# 샘플사이즈 10
size_10 = calc_sample_mean(size = 10, n_trial = 10000)
size_10_df = pd.DataFrame({
    "sample_mean"   : size_10,
    "size"          : np.tile("size 10", 10000)})
# 샘플사이즈 20
size_20 = calc_sample_mean(size = 20, n_trial = 10000)
size_20_df = pd.DataFrame({
    "sample_mean"   : size_20,
    "size"          : np.tile("size 20", 10000)})
# 샘플사이즈 30
size_30 = calc_sample_mean(size = 30, n_trial = 10000)
size_30_df = pd.DataFrame({
    "sample_mean"   : size_30,
    "size"          : np.tile("size 30", 10000)})
# 종합
sim_result = pd.concat([size_10_df, size_20_df, size_30_df])
# 결과 표시
print(sim_result.head())

sns.violinplot(x = "size", y = "sample_mean",
                data = sim_result, color = 'gray')  # 바이올린 플롯
# → 샘플사이즈가 커질수록 표본평균이 흩어지는 정도가 작아져서 모평균(4)에 가깝게 모이는 것을 알 수 있음


# p165  3.5.9   표본평균의 표준편차는 모집단보다 작다
"""
    샘플사이즈가 커지면 표본평균의 흩어짐이 작아지는 현상을
    표본평균의 표준편차를 샘플사이즈별로 살펴봄으로써 다시 한 번 확인해본다.
    
    대상 : 표본평균의 표준편차
    변화시키는 것 : 샘플사이즈
    알고 싶은 것 : 샘플사이즈가 커질수록 표본평균의 표준편차는 작아진다.
                → 샘플사이즈가 커지면 표본평균은 보다 신뢰할 수 있는 값이 된다.
"""
size_array = np.arange(start = 2, stop = 102, step = 2)     # 2 ~ 100까지 2씩 차이 나게 샘플사이즈 준비
size_array

sample_mean_std_array = np.zeros(len(size_array))       # 표본평균의 표준편차를 저장할 변수 준비
# 시뮬레이션 실행(시행횟수 100)
np.random.seed(1)
for i in range(0, len(size_array)):
    sample_mean = calc_sample_mean(size = size_array[i], n_trial = 100)
    sample_mean_std_array[i] = sp.std(sample_mean, ddof = 1)

# 꺾은선그래프(가로축 = 샘플사이즈, 세로축 = 표본평균의 표준편차)
plt.plot(size_array, sample_mean_std_array, color = "black")
plt.xlabel("sample size")
plt.ylabel("mean_std value")
# → 샘플사이즈가 커질수록 표본평균의 표준편차가 작아짐.
# → 샘플사이즈를 크게 하면 흩어짐이 적은 신뢰할 수 있는 표본평균을 얻을 수 있음.


# p167  3.5.10  표준오차
"""
    표준오차(Standard error, SE) : 표본평균의 표준편차
    = σ(모집단의 표준편차) / N(샘플사이즈)
    → 샘플사이즈가 클수록 표준오차는 작아진다.
"""
# 표준오차
standard_error = 0.8 / np.sqrt(size_array)
standard_error

# 표준오차와 시뮬레이션 결과 비교(그래프)
# 표준오차는 linestyle = 'dotted'로 지정해서 점선으로 그리기
plt.plot(size_array, sample_mean_std_array, color = "black")        # 시뮬레이션 결과
plt.plot(size_array, standard_error, color = "black", linestyle = "dotted")
# → 시뮬레이션 결과와 표준오차의 값이 거의 일치함


# p168  3.5.11  표준오차의 직관적인 설명
"""
    표본평균의 표준편차(표준오차) < 원래 데이터의 표준편차
    [이유]
        ex) 10명 정원의 엘리베이터 안 사람들의 체중 or 키 흩어짐 정도
            100명 정원의 비행기 안 사람들의 체중 or 키 흩어짐 정도(극단적이 되기 어려움)
        → 탑승객수를 한 번 조사할 때의 샘플사이즈라고 생각한다면, 표본평균에도 같은 방법을 적용해서 생각할 수 있음.
          따라서, 표본평균들이 모평균에서 떨어져 있는 정도, 즉 흩어진 정도가 작아지게 됨
"""


# p169  3.5.12  표본분산의 평균값은 모분산과 차이가 있다
# 표본분산을 대상으로 시뮬레이션
# 표본분산을 10,000번 계산해서 표본분산의 평균값 구해보기
sample_var_array = np.zeros(10000)      # 표본분산을 저장할 변수 준비
# 시뮬레이션 실행(데이터 10개 골라서 표본분산을 구하는 시행 10,000번 반복)
np.random.seed(1)
for i in range(0, len(sample_var_array)):
    sample = population.rvs(size = 10)      # stat.norm.rvs(loc = 4, scale = 0.8, size = 10)와 같음
    sample_var_array[i] = sp.var(sample, ddof = 0)

sp.mean(sample_var_array)       # 표본분산의 평균값 : 0.5746886877332101
# → 모분산은 0.8(표준편차)의 제곱인 0.64
# → 하지만 표본분산의 평균값은 0.575로 분산이 과소평가됨


# p169  3.5.13  불편분산을 사용하면 편향이 사라진다
# 불편분산을 저장하는 변수
unbias_var_array = np.zeros(10000)
# 데이터를 10개 골라서 불편분산을 구하는 시행을 10,000번 반복
np.random.seed(1)
for i in range(0, len(unbias_var_array)):
    sample = population.rvs(size = 10)
    unbias_var_array[i] = sp.var(sample, ddof = 1)      # 불편분산
# 불편분산의 평균값
sp.mean(unbias_var_array)       # 0.6385429863702334
# → 모분산은 0.8(표준편차)의 제곱인 0.64
# → 불편분산의 평균값은 모분산이라고 간주해도 좋음


# p170  3.5.14  샘플사이즈가 크면 불편분산은 모분산에 가까워진다
"""
    대상 : 불편분산
    변화시키는 것 : 샘플사이즈
    알고 싶은 것 : 샘플사이즈가 커지면 불편분산은 모분산에 가까워진다.
"""
size_array = np.arange(start = 10, stop = 100100, step = 100)   # 10 ~ 100010까지 100단위로 변화하는 샘플사이즈
size_array

unbias_var_array_size = np.zeros(len(size_array))   # 불편분산을 저장하는 변수
# 시뮬레이션 실행
np.random.seed(1)
for i in range(0, len(size_array)):
    sample = population.rvs(size = size_array[i])
    unbias_var_array_size[i] = sp.var(sample, ddof = 1)

# 꺾은선 그래프(가로축 = 샘플사이즈, 세로축 = 불편분산)
plt.plot(size_array, unbias_var_array_size, color = "black")
plt.xlabel("sample size")
plt.ylabel("unbias var")
# → 샘플사이즈가 커지면 커질수록 불편분산은 모분산(0.64)에 가까워짐


# p172  3.5.15  불편성
"""
    불편성 : 추정량의 기댓값이 진짜 모수(모집단의 파라미터)가 되는 특성
    → 불편성이 있다는 것은 평균을 냈을 때 과대 또는 과소하지 않는다는 뜻, 이는 곧 편향이 없는 추정량
"""


# p172  3.5.16  일치성
"""
    일치성 : 샘플사이즈가 커지면 추정량이 진짜 모수에 가까워지는 특성
    → 일치성이 있다는 것은 샘플사이즈가 무한할 때, 추정량과 모수가 일치한다는 뜻
"""


# p172  3.5.17  모수에 대해 좋은 추정량
"""
    분석의 목적 : 모집단 분포 추정
        → 모집단 분포를 알면, 모르는 데이터에 대해 예측 및 추측을 할 수 있게 된다.
        ex) 모집단 분포 = 정규분포이면, 
            정규분포의 모수(파라미터)를 추정하는 것으로 모집단 분포를 추정하는 것이 가능해짐
    
    ex) 정규분포 : 정규분포의 모수는 평균과 분산 2개
    - 시뮬레이션을 통해 표본평균의 평균값은 모평균과 거의 값고,
                    불편분산의 평균값은 모분산과 거의 같음을 알 수 있음
    → 표본평균과 불편분산은 둘 다 불편성을 가짐
    
    - 샘플사이즈를 크게 하면, 표본평균은 모평균에 가까워지고, 불편분산은 모분산에 가까워짐
    → 표본평균과 불편분산 둘 다 일치성을 가짐
"""


# p173  3.5.18  큰수의 법칙
"""
    큰수의 법칙 : 표본의 크기가 커지면 표본평균이 모평균에 가까워지는 방법을 표현한 법칙
"""


# p173  3.5.19  중심극한정리
"""
    중심극한정리 : 모집단분포가 무엇이든 간에 샘플사이즈가 커지면 확률변수의 합은 정규분포에 가까워짐
"""
# ex) 동전을 10,000번 던졌을 때, 앞이 나온 횟수의 분포 구하기
# 샘플사이즈와 시행횟수
n_size = 10000
n_trial = 50000
# 앞면이면 1, 뒷면이면 0으로 표시
coin = np.array([0, 1])
# 앞면이 나온 횟수
count_coin = np.zeros(n_trial)
# 동전을 n_size번 던지는 시행을 n_trial번 수행
np.random.seed()
for i in range(0, n_trial):
    count_coin[i] = sp.sum(
        np.random.choice(coin, size = n_size, replace = True)
    )
# 히스토그램 그리기
sns.distplot(count_coin, color = "black")


# p174  3.6 정규분포의 응용
# p175  3.6.1   라이브러리 임포트
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


# p176  3.6.2   확률밀도
sp.pi       # 원주율(π)
sp.exp(1)       # 자연로그 e^x = exp(x), 즉 자연로그 e의 1승 = 2.718281828459045

# 정규분포의 확률밀도 계산
# ex) 평균 4, 분산 0.64(표준편차 0.8)인 정규분포에 대해 확률변수 3일 때의 확률밀도
# 즉, N(3│4, 0.8^2)
x = 3
mu = 4
sigma = 0.8
1 / (sp.sqrt(2 * sp.pi * sigma**2)) * \
    sp.exp(-((x - mu)**2) / (2 * sigma**2))

# 위의 수식 간단히 계산
stats.norm.pdf(loc = 4, scale = 0.8, x = 3)

# 평균 4, 표준편차 0.8인 정규분포의 인스턴스 생성해서 pdf함수 호출
norm_dist = stats.norm(loc = 4, scale = 0.8)
norm_dist.pdf(x = 3)

# 확률밀도 표시
x_plot = np.arange(start = 1, stop = 7.1, step = 0.1)
plt.plot(
    x_plot,
    stats.norm.pdf(x = x_plot, loc = 4, scale = 0.8),
    color = "black"
)


# p177  3.6.3   표본이 어떤 값 이하가 되는 비율
# 표본이 어떤 값 이하가 되는 비율 = (어떤 값 이하가 되는 데이터의 개수 / 샘플사이즈)

# 모집단 분포 N(x│4, 0.8^2)인 모집단에서 표본추출 시뮬레이션 실행(샘플사이즈 100,000)
np.random.seed(1)
simulated_sample = stats.norm.rvs(
    loc = 4, scale = 0.8, size = 100000)
simulated_sample

sp.sum(simulated_sample <= 3)       # 3 이하인 데이터 개수
sp.sum(simulated_sample <= 3) / len(simulated_sample)   # 샘플사이즈로 나누면 약 10.4%


# p178  3.6.4   누적분포함수
"""
    누적분포함수 or 분포함수 : F(X) = P(X<=x)    
    누적분포함수는 stats.norm의 cdf 함수(Cumulative Distribution Function) 사용
    → 데이터를 하나하나 세어보지 않고 적분을 이용해서 확률을 간단히 계산할 수 있는게
      모집단분포를 정규분포라고 가정하는 것의 장점
"""
# 모집단 분포가 N(x│4, 0.8^2)일 때 확률분포에서 얻은 확률변수가 3 이하가 될 확률 계산
stats.norm.cdf(loc = 4, scale = 0.8, x = 3)     # 0.10564977366685535, 약 10%

# 정규분포는 평균에 대해 좌우대칭이므로 데이터가 평균값 이하가 될 확률은 50%
stats.norm.cdf(loc = 4, scale = 0.8, x = 4)     # 0.5, 50%


# p179  3.6.5   하측확률과 퍼센트포인트
"""
    하측확률 : 데이터가 어떤 값 이하가 될 확률
    퍼센트포인트 : 어떤 확률이 될 기준치
    
    '확률변수 x가 N보다 낮을 확률은 M퍼센트다'
    - N(변수)을 고정하고 M(확률)을 구하는 경우 이 때의 M이 하측확률
    - M(확률)을 고정하고 N(변수)을 구하는 경우 이 때의 N이 퍼센트포인트
    
    퍼센트포인트 계산 → stats.norm의 ppf(percent point function) 함수 사용
"""
# 모집단 분포 N(x│4, 0.8^2)일 때 하측확률이 2.5%가 되는 퍼센트포인트
stats.norm.ppf(loc = 4, scale = 0.8, q = 0.025)

lower = stats.norm.cdf(loc = 4, scale = 0.8, x = 3)     # 확률변수의 값을 확률로 변환(3 이하가 될 확률)
stats.norm.ppf(loc = 4, scale = 0.8, q = lower)     # 확률이 다시 확률변수의 값으로 돌아옴(ppf의 인자로 cdf함수의 결과 넣음)

stats.norm.ppf(loc = 4, scale = 0.8, q = 0.5)   # 하측확률이 50%가 되는 퍼센트포인트, 4


# p180  3.6.6   표준정규분포
"""
    표준정규분포 : 평균 0, 분산(표준편차)가 1인 정규분포 → N(x│0, 1)
"""


# p180  3.6.7   t값
"""
    t값 = (표본평균 - 모평균) / 표준오차        ※ 표준화 = (데이터 - 평균) / 표준편차
"""


# p181  3.6.8   t값의 표본분포
# t값의 표본분포를 시뮬레이션으로 확인
# 시뮬레이션 방법
# 01 : 모집단 분포가 N(x│4, 0.8^2)인 모집단에서 표본추출 시뮬레이션(샘플사이즈 10)
# 02 : 얻은 표본에서 표본평균 구하기
# 03 : 얻은 표본에서 표준오차 구하기(표준오차는 표본평균의 표준오차)
# 04 : t = (표본평균 - 모평균) / 표준오차
# 05 : 10,000번 시행 반복

# 난수 시드 설정
np.random.seed(1)
# t값을 저장할 변수 설정
t_value_array = np.zeros(10000)
# 정규분포 클래스의 인스턴스
norm_dist = stats.norm(loc = 4, scale = 0.8)
# 시뮬레이션 실행
for i in range(0, 10000):
    sample = norm_dist.rvs(size = 10)
    sample_mean = sp.mean(sample)
    sample_std = sp.std(sample, ddof = 1)               # 표준편차
    sample_se = sample_std / sp.sqrt(len(sample))       # 표준오차 : 표준편차 / 루트 N
    t_value_array[i] = (sample_mean - 4) / sample_se        # t값 : (표본평균 - 모평균) / 표준오차

# 표준정규분포의 확률밀도 점선으로 그리기(linestyle = "dotted"로 지정)
# stats.norm.pdf(x = x)로 하여 loc와 scale을 지정하지 않을 경우 표준정규분포가 됨
# t값의 히스토그램
sns.distplot(t_value_array, color = "black")
# 표준정규분포의 확률밀도
x = np.arange(start = -8, stop = 8.1, step = 0.1)
plt.plot(x, stats.norm.pdf(x = x), color = "black", linestyle = "dotted")
# → 불편성 만족 : 표본평균의 평균값은 모평균을 따름
# → 때문에 '(표본평균 - 모평균) / 표준오차' 결과의 분포 중심이 0이 됨
# 분포의 밑단이 넓어지고 있음. 분산이 1보다 크기 때문
# 표본에서 계산한 표준오차로 나누었기 때문


# p183  3.6.9   t분포
"""
    t분포 : 모집단 분포가 정규분포일 때 t값의 표본분포
    
    자유도 : 샘플사이즈가 N일 때 N-1로 계산한 값
    t분포의 형태는 자유도에 따라 달라짐
    자유도가 n일 경우 t분포는 t(n)으로 표기
    
    t분포의 평균값 = 0
    t분포의 분산은 1보다 조금 크다
    
    t(n)의 분산(n이 2보다 클 때) = n / (n-2)
    
    자유도가 커질수록 분산은 1에 가까워지고, 표준정규분포와 거의 차이가 나지 않게 됨
    반대로 말하면 샘플사이즈가 작아질 경우 차이가 커짐
"""
# t분포의 확률밀도와 표준정규분포의 확률밀도 겹쳐 표시
plt.plot(x, stats.norm.pdf(x = x), color = "black", linestyle = "dotted")
plt.plot(x, stats.t.pdf(x = x, df = 9), color = "black")
# → 실선표시가 t분포인데, 밑단 쪽이 조금 더 넓은 분포라는 걸 알 수 있음
# → 즉, 평균값과 크게 다른 데이터가 발생하기 쉬워짐

# 표본에서 계산한 표준오차로 표준화된 표본평균의 분포와 t분포의 확률밀도 비교
# 시뮬레이션의 결과와 겹치면 커널밀도추정의 결과와 거의 일치
sns.distplot(t_value_array, color = "black", norm_hist = True)  # 표본에서 계산한 표준오차로 표준화된 표본평균의 분포
plt.plot(x, stats.t.pdf(x = x, df = 9),
         color = "black", linestyle = "dotted")     # t분포의 확률밀도
"""
    t분포의 의미 : 모분산을 모르는 상황에서도 표본평균의 분포에 대해 얘기할 수 있다
"""


# p184  3.7 추정
# p185  3.7.1   분석 준비
# 수치 계산에 사용하는 라이브러리
import numpy as np
import pandas as pd
import scipy as sp
from scipy import stats
# 그래프를 그리기 위한 라이브러리
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()
# 표시 자릿수 지정
# %precision 3
# 그래프를 주피터 노트북에 그리기 위한 설정
# %matplotlib inline

# 분석 대상 데이터 불러오기
fish = pd.read_csv("source/sample/3-7-1-fish_length.csv")["length"]
fish


# p186  3.7.2   점추정
"""
    점추정 : 모수(모집단 분포의 파라미터)를 어느 1개의 값으로 추정하는 추정 방법
            모평균을 추정하는 경우에는 표본평균을 추정값으로 사용함
     결국 우리가 하는 것은 표본에서 평균값을 계산하는 것. 이것으로 추정 완료
     여기서 표본평균을 사용해도 좋은 이유는 표본평균은 '불편성'과 '일치성'을 가지고 있는 통계량이기 때문       
"""
mu = sp.mean(fish)      # 표본평균 계산       4.187039324504523
mu      # 4.187039324504523

sigma_2 = sp.var(fish, ddof = 1)   # 모분산 추정(불편분산 사용)
sigma_2     # 0.6803017080832623


# p186  3.7.3   구간추정
"""
    구간추정 : 추정값이 폭을 가지게 하는 추정 방법
            추정값의 폭 계산에는 확률의 개념 사용
            폭을 가지므로 추정오차 생김 → 추정오차가 작으면, 구간추정 폭이 좁아짐
            샘플사이즈가 커져도 구간추정의 폭이 작아짐
"""


# p187  3.7.4   신뢰계수와 신뢰구간
"""
    신뢰계수 : 구간추정의 폭에 대한 신뢰 정도를 확률로 표현한 것 ex) 95%, 99%
    신뢰구간 : 특정 신뢰계수를 만족하는 구간
    
    똑같은 데이터를 대상으로 했을 경우 신뢰계수가 클수록 신뢰구간의 폭이 넓어짐
    신뢰할 수 있는 정도가 올라간다고 생각하면 아무래도 안전제일이기 때문에 폭이 넓어질 수 밖에 없음
"""


# p187  3.7.5   신뢰한계
"""
    신뢰한계 : 신뢰구간 하한값과 상한값. 각각 하측신뢰한계와 상측신뢰관계라고 함
"""


# p187  3.7.6   신뢰구간 계산 방법
"""
    '(표본평균 - 모평균) / 표준오차'로 계산한 t값은 t분포를 따름
    구간추정을 할 때는 t분포의 퍼센트포인트(어떤 확률이 되는 기준점) 사용
    ex) 신뢰계수를 95%라고 했을 때,
        t분포를 따른다면 2.5%, 97.5% 지점을 계산함
        t분포를 따르는 변수가 이 구간에 들어갈 확률은 95%라는 얘기가 되므로 이 구간을 사용하면 됨
"""


# p187  3.7.7   구간추정(실습)
# 구간추정에 필요한 정보는 자유도(샘플사이즈-1), 표본평균, 표준오차 3가지
df = len(fish) - 1      # 자유도(샘플사이즈-1)

sigma = sp.std(fish, ddof = 1)
se = sigma / sp.sqrt(len(fish))     # 표준오차

# 신뢰구간 stats.t.interval 함수 이용
# 신뢰계수 = alpha, 자유도 = df, 표본평균 = loc, 표준오차 = scale 지정
interval = stats.t.interval(
    alpha = 0.95, df = df, loc = mu, scale = se)
interval    # (3.5970100568358245, 4.777068592173221) (하측신뢰관계, 상측신뢰관계)


# p188  3.7.8   신뢰구간을 구하는 방법 상세 설명
"""
    1. 어떤 자유도를 가지는 t분포를 가지는 97.5% 지점 계산
    1.1 t분포를 따르는 97.5% 지점을 t0.975라고 표기
    1.2 t분포는 평균에 대해 좌우대칭이기 때문에 2.5% 지점은 -t0.975로 표기
    1.3 t분포를 따르는 변수가 -t0.975 이상 t0.975 이하가 되는 확률이 95%이다.
    1.3.1 이 때 95%가 신뢰계수가 된다.
    2. 표본평균 -t0.975 * 표준오차가 하측신뢰관계
    3. 표본평균 +t0.975 * 표준오차가 상측신뢰관계
"""
t_975 = stats.t.ppf(q = 0.975, df = df)     # 97.5% 지점      2.2621571627409915
t_025 = stats.t.ppf(q = 0.025, df = df)     # 2.5%(-97.5%) 지점       -2.262157162740992

lower = mu - t_975 * se     # 하측신뢰관계        3.5970100568358245
upper = mu + t_975 * se     # 상측신뢰관계        4.777068592173221


# p190  3.7.9   신뢰구간의 폭을 결정하는 요소
# 표본의 분산 크기가 크다
# → 데이터가 평균값에서 흩어져 있다
# → 평균값을 신뢰할 수 없게 된다

# 표본표준편차를 10배로 늘려서 95% 신뢰구간 계산
se2 = (sigma*10) / sp.sqrt(len(fish))
stats.t.interval(alpha = 0.95, df = df, loc = mu, scale = se2)  # (-1.7132533521824618, 10.087332001191509)
# → 신뢰구간의 폭이 꽤 넓어짐
# → 진짜 모평균이 어디 있는지 잘 모르게 된다

# 반대로 샘플사이즈가 커지면 표본평균을 신뢰할 수 있게 되므로 신뢰구간이 좁아짐
# 샘플사이즈가 커지면 자유도가 커지고 표준오차가 작아짐
df2 = (len(fish)*10) - 1
se3 = sigma / sp.sqrt(len(fish)*10)
stats.t.interval(alpha = 0.95, df = df2, loc = mu, scale = se3) # (4.0233803082774395, 4.350698340731607)

# 완전히 똑같은 데이터라고 해도 신뢰계수(95%)가 커질수록 안전해진다고 볼 수 있고, 신뢰구간의 폭이 넓어짐
stats.t.interval(alpha = 0.95, df = df, loc = mu, scale = se)   # (3.5970100568358245, 4.777068592173221)
stats.t.interval(alpha = 0.95, df = df, loc = mu, scale = se2)  # (-1.7132533521824618, 10.087332001191509)
stats.t.interval(alpha = 0.95, df = df2, loc = mu, scale = se3) # (4.0233803082774395, 4.350698340731607)
stats.t.interval(alpha = 0.99, df = df, loc = mu, scale = se)   # (3.3393979149413973, 5.034680734067649)


# p191  3.7.10  구간추정 결과의 해석
"""
    신뢰계수 95%의 95%는 다음과 같이 얻을 수 있다.
    01 : 원래 모집단 분포에서 표본 추출
    02 : 이번에도 같은 방법으로 95% 신뢰구간 계산
    03 : 이 시행을 여러 번 반복
    04 : 모든 시행 중 원래 모집단이 신뢰구간에 포함되는 비율이 95%
"""
# 시뮬레이션(시행횟수 20,000번)
# 95% 신뢰구간을 구하는 시행을 20,000번 시행
# 신뢰구간이 모평균(4)을 포함하면 True
be_included_array = np.zeros(20000, dtype = "bool")

np.random.seed(1)
norm_dist = stats.norm(loc = 4, scale = 0.8)
for i in range(0, 20000):
    sample = norm_dist.rvs(size = 10)
    df = len(sample) - 1
    mu = sp.mean(sample)
    std = sp.std(sample, ddof = 1)
    se = std / sp.sqrt(len(sample))
    interval = stats.t.interval(0.95, df, mu, se)
    if(interval[0] <= 4 and interval[1] >= 4):
        be_included_array[i] = True

sum(be_included_array) / len(be_included_array)     # 0.93855 (ddof = 0)
sum(be_included_array) / len(be_included_array)     # → 신뢰구간이 모평균(4)을 포함한 비율 0.948(ddof = 1)로 대략 95%


# p193  3.8 통계적가설검정
# p193  3.8.1   통계적가설검정
"""
    통계적가설검정 : 표본을 사용해서 모집단에 관한 통계적인 판단을 내리는 방법
"""


# p193  3.8.2   1변량 데이터의 t검정
"""
    대상 : 평균값
    판단하는 것 : 평균값이 어떤 값과 다른지 얘기할 수 있는지 여부
"""


# p194  3.8.3   유의미한 차이


# p194  3.8.4   t검정: 직관적인 생각
"""
    50g과 의미있는 차이, 즉 유의미한 차이가 있다고 생각할 수 있는 조건
    - 큰 샘플에서 조사했다 : 샘플사이즈가 크다
    - 정밀한 저울로 측정했다 : 데이터의 흩어짐(분산)이 작다
    - 중량의 평균값이 50g에서 크게 벗어난다 : 평균값의 차이가 크다
    → t검정에서 이 3가지 조건을 만족했을 때 유의미한 차이가 있다고 판단할 수 있다
"""


# p194  3.8.5   평균값의 차이가 큰 것만으로는 유의미한 차이를 얻을 수 없다


# p195  3.8.6   t값
"""
    t값 = (표본평균 - 비교대상값) / (표준편차 / root샘플사이즈)
        = (표본평균 - 비교대상값) / 표준오차
    t값은 절대값에 의미가 있다
"""


# p196  3.8.7   통계적가설검정의 틀: 귀무가설, 대립가설
"""
    귀무가설 : 기각 대상이 되는 첫번째 가설
    대립가설 : 귀무가설과 대립되는 가설
"""


# p196  3.8.8   p값
"""
    p값 : 표본과 귀무가설 간의 모순을 나타내는 지표
    p값이 작을수록 귀무가설과 표본이 모순된다고 생각할 수 있음
    p값은 확률로 표현
    p값과 신뢰구간 둘 다 완전히 같은 조건에서 몇 번이고 표본추출을 하고 t값 계산을 반복해서 구한 확률을 해석한다
    (3.11절에서 추가 설명)
"""


# p196  3.8.9   유의수준
"""
    유의수준 : 귀무가설을 기각하는 기준이 되는 값(위험률)
    p값이 유의수준을 밑돌면 귀무가설 기각
"""


# p197  3.8.10  t검정과 t분포의 관계


# p197  3.8.11  단측검정과 양측검정
"""
    단측검정 : 봉지과자의 평균중량이 50g보다 작다(or 크다)는 것을 알아보는 검정 방법(50g보다 큰 지(작은 지)는 상정하지 않음)
    양측검정 : 봉지과자의 평균중랴이 50g과 다르다는 것을 알아보는 검정 방법
"""


# p197  3.8.12  p값 계산 방법
"""
    p값 = (1 - α) * 2
        α : 표본에서 계산한 t값을 t표본이라고 할 때,
            t분포의 누적분포함수를 사용하면 모평균이 50이라고 가정했을 때 t값이 t표본보다 작을 확률
            이 때의 확률을 α라고 부름
        마지막에 *2를 한 것은 양측검정을 위함.
        봉지과자의 평균중량이 50g과 다르다는 확률을 계산하려면 큰 경우와 작은 경우 2가지를 고려해야 하므로
        (1-α)를 2배로 해야할 필요가 있다. 또한 단측검정의 경우 단순히 (1-α)가 p값이 됨
"""


# p198  3.8.13  t검정의 구현: 분석 준비
# 수치 계산에 사용하는 라이브러리
import numpy as np
import pandas as pd
import scipy as sp
from scipy import stats
# 그래프를 그리기 위한 라이브러리
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()
# 표시 자릿수 지정
# %precision 3
# 그래프를 주피터 노트북에 그리기 위한 설정
# %matplotlib inline

# 데이터 불러오기
junk_food = pd.read_csv("source/sample/3-8-1-junk-food-weight.csv")["weight"]   # 봉지과자 중량
junk_food.head()

# t검정 실시
# 귀무가설 : 봉지과자의 평균중량은 50g이다.
# 대립가설 : 봉지과자의 평균중량은 50g이 아니다.
# 유의수준 5% → p값이 0.05보다 낮다면 귀무가설 기각(봉지과자 중량이 50g과 다르다)


# p199  3.8.14  t검정의 구현: t값 계산
# t값 = (표본평균 - 비교대상값) / 표준오차
mu = sp.mean(junk_food)     # 표본평균      55.38496619666667
df = len(junk_food) - 1     # 자유도       19
se = sp.std(junk_food, ddof = 1) / sp.sqrt(len(junk_food))      # 표준오차      1.9579276805755885
t_value = (mu - 50) / se        # t값    2.7503396831713434


# p200  3.8.15  t검정의 구현: p값 계산(이론)
"""
    p값 복습
    - t분포의 누적분포함수를 사용하면 모평균을 50이라고 가정했을 때,
      t값이 t표본보다 작을 확률을 계산할 수 있음(이 확률을 α라고 함).
      (1-α)를 구하면 모평균을 50이라고 가정했을 때, t값이 t표본보다 클 확률을 계산할 수 있고,
      (1-α)가 작아지면 t값이 t표본보다 클 확률이 낮다(즉, t표본이 충분히 크다)
      라는 말이 되어 유의미한 차이를 얻을 수 있게 된다.
"""
alpha = stats.t.cdf(t_value, df = df)
(1 - alpha) * 2     # 양측검정      0.012725590012524268
# → p값이 유의수준 0.05보다 작으므로 유의미한 차이가 있다고 볼 수 있음
# → 즉, 봉지과자의 평균중량은 50g과 유의미하게 차이가 있다는 판단 가능

stats.ttest_1samp(junk_food, 50)    # Ttest_1sampResult(statistic=2.750339683171343, pvalue=0.012725590012524182)


# p201  3.8.16  시뮬레이션에 의한 p값 계산
"""
    p값 : 귀무가설이 옳다고 가정한 뒤 몇 번이고 표본추출을 하고
        t값 계산을 반복했을 때 t표본과 같거나 그보다 큰 t값을 얻는 비율로 해석
        양측검정의 경우 이 비율을 2배로 한 것이 p값
        이 비율이 작은 경우 t표본을 넘을 일이 거의 없다.
        다시 말해 t표본이 충분히 크다고 생각할 수 있어서 유의미한 차이를 얻을 수 있다고 판단
"""
size = len(junk_food)       # 샘플사이즈     20
sigma = sp.std(junk_food, ddof = 1)     # 표준편차      8.756118777591022

t_value_array = np.zeros(50000)     # 50000번 계산 t값 저장할 준비

np.random.seed(1)
norm_dist = stats.norm(loc = 50, scale = sigma)
for i in range(0, 50000):
    sample = norm_dist.rvs(size = size)
    sample_mean = sp.mean(sample)
    sample_std = sp.std(sample, ddof = 1)
    sample_se = sample_std / sp.sqrt(size)
    t_value_array[i] = (sample_mean - 50) / sample_se

(sum(t_value_array > t_value) / 50000) * 2      # 0.01324
# → 이론적으로 계산한 값과 거의 일치


# p202  3.9 평균값의 차이 검정
# p202  3.9.1   2집단 데이터에 대한 t검정
# p203  3.9.2   대응표본 t검정
# p203  3.9.3   분석준비
# 수치 계산에 사용하는 라이브러리
import numpy as np
import pandas as pd
import scipy as sp
from scipy import stats
# 그래프를 그리기 위한 라이브러리
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()
# 표시 자릿수 지정
# %precision 3
# 그래프를 주피터 노트북에 그리기 위한 설정
# %matplotlib inline

# 데이터 불러오기
paired_test_data = pd.read_csv("source/sample/3-9-1-paired-t-test.csv")
print(paired_test_data)

# t검정
# 귀무가설 : 약을 먹기 전과 후의 체온이 변하지 않는다
# 대립가설 : 약을 먹기 전과 후의 체온이 다르다


# p204  3.9.4   대응표본 t검정(실습)
# 약을 먹기 전과 후의 표본평균
before = paired_test_data.query('medicine == "before"')["body_temperature"]
after = paired_test_data.query('medicine == "after"')["body_temperature"]
# 배열형으로 변환
before = np.array(before)
after = np.array(after)
# 차이 계산
diff = after - before

# 차이값의 평균값이 0과 다른지 1집단 t검정
stats.ttest_1samp(diff, 0)      # Ttest_1sampResult(statistic=2.901693483620596, pvalue=0.044043109730074276)

















