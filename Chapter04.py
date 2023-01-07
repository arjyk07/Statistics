#####################################################################

# Chapter 4 통계모델 기본
# 파이썬으로 배우는 통계학(update: 2022.12.27)(~4.1)
# 파이썬으로 배우는 통계학(update: 2023.01.02)(~4.4 진행중)
# 파이썬으로 배우는 통계학(update: 2023.01.07)(~4.5 진행중)

#####################################################################

"""
    4.1 통계모델
    4.2 통계모델을 만드는 방법
    4.3 데이터의 표현과 모델의 명칭
    4.4 파라미터의 추정: 우도의 최대화
    4.5 파라미터 추정: 손실의 최소화
    4.6 예측 정확도의 평가와 변수 선택
"""

# p219  4.1 통계모델
# p219  4.1.1   모델(모형)


# p219  4.1.2   모델링
# 모델을 만드는 것


# p219  4.1.3   모델은 무엇에 도움이 되나
"""
    비행기의 형태를 한 작은 모형을 만들면 진짜 비행기를 사용하지 않고도 진짜 비행기의 특성,
    예를 들어 그 비행기가 날 수 있는지, 바람이 불면 어떻게 흔들리는지 등을 알아볼 수 있음.
    
    실제 세계의 모형(모델)을 이용하는 것으로 현실 세계의 이해와 예측에 활용할 수 있다. 
"""


# p220  4.1.4   복잡한 세계를 단순화하다
"""
    요인들은 모두 고려하는 것은 비효율적으로 이해할 수 없는 인과율이 튀어나올 수 있음
    너무 단순하게만 하면, 맞지 않는 모델이 될 수 있음
    → 인간이 이해할 수 있을 만큼 단순하면서도 복잡한 현실상을 어느 정도 잘 설명할 수 있는
    복잡한 세계를 위한 단순한 모델을 구축하자!
"""


# p220  4.1.5   복잡한 현상을 특정한 관점에서 다시 보게 한다
"""
    모델은 실제 현상을 어떤 측면에서 바라본 결과라고 얘기할 수 있다.
    ex) 기온과 맥주 매상관계라는 '그 날의 기온이라는 관점'에서 바라본 모델 구축
    → 분석의 목적에 맞춰서 작성하는 모델과 주목하는 관점을 바꾸는 것이 가능하다.
"""


# p220  4.1.6   수리모델
"""
    수리모델 : 현상을 수식으로 표현한 모델
"""


# p221  4.1.7   확률모델
"""
    확률모델 : 수리모델 중에서도 특히 확률적인 표현이 있는 모델
    
    ex) 맥주 매상 ~ N(20 + 4*기온, σ^2)       ※ 정규분포를 가정했을 경우
        → 맥주 매상은 평균이 (20 + 4*기온), 분산이 σ^2인 정규분포를 따른다.
        맥주 매상 = 20 + 4*기온 + ε,  ε ~ N(0, σ^2)
        → 맥주 매상은 (20 + 4*기온)에 대해 평균이 0, 분산이 σ^2인 정규분포를 따르며 노이즈가 있다.
"""


# p222  4.1.8   통계모델
"""
    통계모델 : 데이터에 적합하게 구축된 확률모델
    ex) 맥주 매상 ~ N(20 + 4*기온, σ^2)   → 맥주 매상 ~ N(10 + 5*기온, σ^2)
    → 확률모델의 구조를 생각하면서 데이터에 적합하게 파라미터를 조정해가며 통계모델 구축
"""


# p222  4.1.9   확률분포와 통계모델
"""
    모집단에서 랜덤 샘플링하여 표본 얻는 것 = 모집단분포를 따르는 난수를 발생시키는 것
    통계모델을 사용하면 확률분포의 모수(파라미터)의 변화 패턴을 명확히 할 수 있다.
"""


# p222  4.1.10  통계모델을 이용한 예측
"""
    ex) 맥주 매상 ~ N(10 + 5*기온, σ^2)
    → 기온이 10도일 때의 매상 예측은 '기대값이 60, 분산이 σ^2인 정규분포를 따르는 매상 데이터를 얻을 것'
    
    통계모델에 의한 예측은 기온이라는 독립변수를 얻는 것이 조건인 매상의 확률분포,
    즉, 조건부확률분포라는 형태로 얻을 수 있다.
    그리고 예측값의 대표값을 1개 고르는 경우에는 전부 기대값이 사용됨.
"""


# p223  4.1.11  통계모델과 고전적인 분석 절차의 비교
"""
    ex) 완전히 똑같은 상품에 대해 가격이 쌀 때와 비쌀 때의 매상 평균값 비교(평균값의 차이 검정)
        - 모델1 : 가격이 쌀 때와 비쌀 때의 매상 평균값은 변하지 않는다.
        - 모델2 : 가격이 쌀 때와 비쌀 때의 매상 평균값은 변한다.
        → 즉, 평균값의 차이 검정은 '1단계: 2단계 모델을 작성'과 '2단계: 어느 모델이 더 들어맞는지 판단하기'
         라는 2가지 작업이며, 그 중 2단계의 판단만 사람들에게 보이게 되는 것
         
    모델을 만드는 단계에 주목함으로써 보다 복잡한 현상에 대해서도 분석을 할 수 있게 된다.
"""


# p224  4.1.12  통계모델의 활용
# 통계모델은 어디까지나 '잠정적인' 세계의 모형일 뿐
# 그럼에도 불구하고 통계모델은 현대 데이터 분석의 표준 도구



# p224  4.2 통계모델을 만드는 방법
# p224  4.2.1   이 절의 예제
# ex) 맥주 매상 예측 모델 구축
# 기온, 날씨, 날씨(맑음, 비, 흐림), 맥주 가격 등


# p224  4.2.2   종속변수와 독립변수
"""
    종속변수(응답변수) : 어떤 요인에 종속된 변수, 다시 말해 어떤 변화에 응답하는 변수    → ex) 맥주 매상
    독립변수(설명변수) : 흥미 있는 대상의 변화를 설명하는 변수, 
                        모델 내의 다른 대상에 영향을 받지 않는(독립적인) 변수         → ex) 기온, 날씨 맥주 가격
    종속변수 ~ 독립변수
"""


# p225  4.2.3   파라메트릭 모델
"""
    파라메트릭 모델 : 가능한 한 현상을 단순화해서 소수의 파라미터만 사용하는 모델
"""


# p225  4.2.4   논파라메트릭 모델
"""
    논파라메트릭 모델 : 소수의 파라미터만 사용한다는 방침을 취하지 않는 모델
"""


# p225  4.2.5   선형모델
"""
    선형모델 : 종속변수와 독립변수의 관계를 선형으로 보는 모델
    
    ex) 맥주 매상(만원) = 20 + 4*기온
        → (기온이 20도거나 35도 일 때도) 기온이 1도 오르면 매상이 4만원 오른다.
"""


# p226  4.2.6   계수와 가중치
"""
    계수 : 통계모델에 사용되는 파라미터
    
    ex) 맥주 매상 ~ N(Β0, Β1 * 기온, σ^2)
        B0, B1: 계수  (B0 : 절편, B1 : 회귀계수)
        계수와 독립변수(여기서는 기온)만 있으면 모수(여기서는 정규분포의 평균값)를 추측(예측)할 수 있음
        통계학에서는 계수라고 부르지만, 
        머신러닝에서는 같은 내용을 나타내는 데 가중치라는 표현을 사용하는 경우도 있음.
"""


# p227  4.2.7   모델 구축 = 모델 정하기 + 파라미터 추정
"""
    모델 구축
    1) 모델 구조를 수식으로 표현 → 모델의 특정
    2) 파라미터 추정
    ⇒ 예측 정밀도가 좋지 않았을 때, 
      1)원래 구조가 좋지 않았던 건지, 2)구조는 올바른데 파라미터 추정이 틀린 것인지 음미할 필요가 있음.
      
"""


# p227  4.2.8   선형모델을 구축하는 방법
"""
    선형모델임을 가정했을 때, 모델의 구조를 바꾸는 방법
    1) 모델에 사용되는 독립변수를 바꾼다.
    2) 데이터가 따르는 확률분포를 바꾼다.
"""


# p228  4.2.9   변수 선택
"""
    변수 선택 : 모델에 사용될 독립변수를 고르는 작업
    → 변수 선택을 하기 위해서는 우선 여러 가지 변수 조합 모델을 만들어봐야 한다.
    → 변수 조합 중에서 가장 좋은 변수의 조합을 가진 모델을 선택하는 것
    → 가장 좋은 변수 조합을 선택하는 방법
        1) 통계적가설검정
        2) 정보 기준을 이용
    
    ex) 독립변수 A, B, C가 있는 경우
        종속변수 ~ 독립변수 없음
        종속변수 ~ A
        종속변수 ~ B
        종속변수 ~ C
        종속변수 ~ A + B
        종속변수 ~ A + C
        종속변수 ~ B + C
        종속변수 ~ A + B + C
    ex) 종속변수(맥주 매상), 독립변수(기온, 날씨, 맥주 가격 등)
    → 독립변수가 없을 경우의 모델은 맥주 매상의 평균이 언제나 일정하다고 가정한 모델이라고 해석할 수 있음
"""


# p229  4.2.10  Null 모델
"""
    Null 모델 : 독립변수가 없는 모델. Null은 아무 것도 없다는 뜻.
"""


# p229  4.2.11  검정을 이용한 변수 선택
"""
    ex) 맥주 매상 ~ N(B0 + B1 * 기온, σ^2)
    
    통계적가설검정을 이용하는 경우,
        - 귀무가설 : 독립변수의 계수 B1은 0이다.
        - 대립가설 : 독립변수의 계수 B1은 0이 아니다.
        → 귀무가설이 기각되는 경우에는 기온에 대한 계수가 0이 아니라고 판단할 수 있기 때문에
          모델에 기온이라는 독립변수가 필요하다고 판단할 수 있다.
        → 귀무가설이 기각할 수 없을 때는 모델은 간단한 쪽이 좋다는 원칙에 의해 변수(이 경우에는 기온)를
          모델에서 제거함. 이 경우 유일한 독립변수가 제거되는 것이기 때문에 Null모델이 됨.
"""


# p229  4.2.12  정보 기준을 이용한 변수 선택
"""
    정보 기준 : 추정한 모델의 좋은 정도(의 일면)를 정량화한 지표
    
    아카이케 정보 기준(AIC, Akaike's Infomation Criterion)이 자주 사용
    → AIC가 작을수록 좋은 모델이라고 판단할 수 있다.
      모델에서 가능한 변수의 패턴을 망라하여 모델을 구축하고, 각 모델의 AIC를 비교.
      AIC가 가장 작은 모델을 채택함으로써 변수 선택을 실행
"""


# p230  4.2.13  모델 평가
"""
    추정한 모델을 무조건 신뢰하는 것은 위험. 추정한 모델을 평가해야 할 필요가 있음.
    
    평가 관점 : 예측 정확도의 평가, 모델 구축 시 가정한 전제조건을 만족했는가 체크
"""


# p230  4.2.14  통계모델을 만들기 전에 분석의 목적을 정한다
"""
    ※ 파이썬 코드를 작성하기 전에 분석의 목적을 정하고 데이터를 수집하여 모델링하는 것이 중요함!
"""


# p230  4.3 데이터의 표현과 모델의 명칭
# p231  4.3.1   정규선형모델
"""
    정규선형모델 : 종속변수가 정규분포를 따르는 것을 가정한 선형모델
                 파라메트릭모델
                 -∞ ~ +∞
                 정규선형모델은 일반선형모델의 일종
"""


# p231  4.3.2   회귀분석
"""
    회귀분석 : 정규선형모델 중 독립변수가 연속형 변수인 모델(회귀모델)
"""


# p231  4.3.3   다중회귀분석
"""
    다중회귀분석 : 회귀분석 중에서도 독립변수가 여러 개 있는 것
    단일회귀분석 : 회귀분석 중에서도 독립변수가 1개인 회귀분석
"""


# p231  4.3.4   분산분석
"""
    분산분석 : 정규선형모델 중에서 독립변수가 카테고리형 변수인 모델
    일원분산분석 : 독립변수가 1종류일 때
    이원분산분석 : 독립변수가 2종류일 때
"""


# p232  4.3.5   일반선형모델
"""
    일반선형모델 : 종속변수가 확률분포를 정규분포 이외의 분포에도 사용 가능하게 한 선형모델
"""


# p232  4.3.6   머신러닝에서의 명칭
"""
    회귀(regression) : 종속변수가 연속형 변수인 모델
                      이 경우 정규선형모델은 넓은 의미에서 회귀가 됨
    분류(classification) : 종속변수가 카테고리형 변수인 모델
                          일반선형모델은 다루는 확률분포에 따라서 회귀모델이라고 하거나 식별모델이라고 함
                          모집단분포를 이항분포라고 가정했을 경우 식별모델,
                          푸아송분포라고 가정했을 경우 회귀모델
"""


# p232  4.4 파라미터 추정: 우도의 최대화
# p232  4.4.1   파라미터 추정 방법을 배우는 의미
"""
    ex) 텔레비전의 구조를 몰라도 텔레비전을 시청할 수 있음
    → 파라미터 추정의 원리를 몰라도 파이썬을 사용해서 통계모델을 구축하고, 예측이나 현상의 해석에 이용할 수 있음
    
    But, 텔레비전을 고칠 수 있는 사람은 텔레비전의 구조를 알고 있는 사람뿐이다.
        → 계산을 실행하면서 에러나 경고가 나왔을 때 원인을 찾는 것이 가능한 사람은 파라미터 추정의 원리를 아는 사람뿐이다.
          그리고 무엇보다도 새로운 기술이 나타났을 때 그것을 빨리 활용할 수 있는 사람은 원래부터 기술의 원리를 알고 있던 사람뿐이다.
"""


# p233  4.4.2   우도
"""
    우도 : 파라미터가 정해져 있을 때 표본을 얻을 수 있는 확률(밀도)
          우도의 우는 그럴듯하다는 뜻 → 우도는 그럴듯한 정도라는 뜻
          우도는 Likelihood의 머리글자를 사용하여 L로 표시
          
    ex) 앞면 확률 : 1/2 → 1/2이 파라미터
        2번 던져서 1) 앞면, 2) 뒷면 나왔다고 가정 (표본) 
        → 이 표본을 얻을 수 있는 확률 1/2 * 1/2 = 1/4
    
    ex) 앞면 확률 : 1/3
        2번 던져서 1) 앞면, 2) 뒷면 나왔다고 가정 (표본)
        → 우도 : 1/3 * 2/3 = 2/9 
"""


# p233  4.4.3   우도함수
"""
    우도함수 : 파라미터를 넘겨서 우도를 계산할 수 있는 함수
    
    ex) 앞면 확률 : 파라미터(θ)
        → 파라미터(θ)를 지정하여 우도를 구하는 우도함수 : L(θ)
        
    L(θ) = θ * (1-θ)
"""


# p233  4.4.4   로그우도
"""
    로그우도 : 우도에 로그를 취한 것(로그를 취하면 나중에 계산이 편해지는 경우가 많음)
"""


# p234  4.4.5   로그의 성질
"""
    지수 : 2^3 = 8(2의 3승)
    로그 : 'X의 Y승 = Z'의 관계에서 X와 Z를 고정해서 Y를 구하는 계산
      1) 단조증가한다
      2) 곱셈이 덧셈으로 바뀐다
      3) 절대값이 극단적으로 작은 값이 되기 어렵다
        ex) 1/1024 = 0.001 → log2(1/1024) = -10 
"""


# p236  4.4.6   최대우도법
"""
    최대우도법 : 우도나 로그우도의 결과를 최대로 하기 위한 파라미터를 추정할 때 사용하는 방법
    
    ex) 앞면 1/2, 뒷면 1/2 : 파라미터 θ가 1/2일 때, 우도는 1/4(= 1/2 * 1/2)
        앞면 1/3, 뒷면 2/3 : 파라미터 θ가 1/3일 때, 우도는 2/9(= 1/3 * 2/3)
        1/4과 2/9 중에서 1/4이 크기 때문에 파라미터 θ는 1/2이 좋다고 할 수 있다.
"""


# p237  4.4.7   최대우도추정량
"""
    최대우도추정량(θ^) : 최대우도법에 의해 추정된 파라미터
"""


# p237  4.4.8   최대화 로그우도
"""
    최대화 로그우도 : 최대우도추정량을 사용했을 때의 로그우도 logL(θ^)
"""


# p237  4.4.9   정규분포를 따르는 데이터의 우도
"""
    ex) 모집단이 정규분포를 따른다고 가정했을 때,
    y ~ N(μ, σ^2) → y는 평균 μ, 분산 σ^2인 정규분포를 따른다고 가정
        y1을 얻었을 때의 확률밀도 : N(y1 | μ, σ^2)
        y2를 얻었을 때으 확률밀도 : N(y2 | μ, σ^2)
        → 우도(L)는 이를 최대로 하는 파라미터 μ, σ^2을 계산하면 된다.
         L = N(y1 | μ, σ^2) * N(y2 | μ, σ^2)
"""


# p237  4.4.10  장애모수
"""
    장애모수 : 직접적인 관심의 대상이 아닌 파라미터
    ex) 정규분포의 모수는 평균과 분산 2가지. 하지만 평균을 추정할 수 있다면 분산 또한 알 수 있다.
        따라서 이미 알고 있는 것으로 취급함. 정규분포를 추정할 때의 최대우도법에서는 분산 σ^2를 장애모수 취급함.
        Null 모델의 경우 평균 μ만 추정하면 됨.
"""


# p238  4.4.11  정규선형모델의 우도
"""
    ex) 맥주 매상 ~ N(β0 + β1 * 기온, σ^2)
        계수 β0, β1을 결정했다고 하고, 그 때의 우도 계산(샘플사이즈 2)
        맥주 매상(y), 기온(x)라 할 때,
        우도(L) = N(y1 | β0 + β1 * x1, σ^2) * N(y2 | β0 + β1 * x2, σ^2)
        
        로그를 취하면 ∏(곱셈)이 ∑(덧셈)으로 변함.
        logL = ∏log[N(yi | β0 + β1 * xi, σ^2)]
        
        최대우도법 : 로그우도를 최대로 하는 파라미터 β0, β1을 추정치로 사용하는 것
        arg max : 함수의 결과값을 최대로 하는 파라미터를 구하는 것
            arg max LogL = arg max ∑log(N(yi | β0 + β1 * xi, σ^2)]

        파이썬의 stats.norm.pdf 함수를 사용해서 우도를 구해도 괜찮음.
        최대우도는 정규분포 이외의 확률분포에도 적용할 수 있다.             
"""


# p239  4.4.12  최대우도법 계산 예


# p240  4.4.13  최대우도추정량의 성질
"""
    최대우도추정량은 추정오차라는 관점에서 보면 매우 바람직한 성질을 가지고 있음.
     - 우선, 최대우도추정량은 N → ∞, 즉 샘플사이즈가 한없이 커질 때 추정량의 표본분포가 점근적으로
       정규분포를 따른다고 알려져 있음. 이를 점근적 정규성이라고 함.
       이것 자체로도 유용한 성질이며, 통계적가설검정을 할 때 활용됨.
     - 즉, 최대우도법은 점근 유효추정량이다.
       표본분산의 분산이 작다는 것은 추정치의 흩어짐이 작고, 추정의 오차가 작다는 의미이므로
       최대우도추정량은 바람직한 성질을 가진 추정량이라고 할 수 있다.
"""


# p241  4.5 파라미터 추정: 손실의 최소화
"""
    파라미터 추정의 기본 개념 : 모델에 잘 들어맞는 파라미터를 채용한다는 것.
    최대우도법은 모델에 들어맞는 정도를 우도로 수치화해서 그것이 최대가 되는 파라미터를 추정함.
    이 절에서는 머신러닝에서 자주 사용되는 개념인 손실의 최소화라는 측면에서 파라미터를 추정하는 방법을 살펴봄.
"""


# p241  4.5.1   손실함수
"""
    손실함수 : 파라미터 추정할 때 손실을 최소화하는 목적으로 사용됨
"""


# p241  4.5.2   잔차
"""
    잔차(residuals) : 실제 종속변수의 값과 모델을 이용해서 계산한 종속변수의 추정치와의 차이
    residuals = y - y^
    ex) 맥주 매상 ~ N(β0 + β1 * 기온, σ^2)
        기온 20도 → 맥주 매상의 기대값은 β0 + β1 * 20으로 계산할 수 있다.
                   이것이 기온 20도일 때의 맥주 매상의 추정치(점추정치)
"""


# p242  4.5.3   잔차의 합을 그대로 손실의 지표로 사용할 수 없는 이유
"""
    잔차의 합계가 같은 0이라도 모델 수준이 다음
"""


# p243  4.5.4   잔차제곱합
"""
    잔차제곱합 : 잔차를 제곱해서 합계를 구한 것
    잔차제곱합 = ∑[(yi - yi^)^2]
"""


# p243  4.5.5   최소제곱법
"""
    최소제곱법 : 잔차제곱합을 최소로 하는 파라미터를 채용하는 방법
                즉, 잔차제곱합을 사용하여 손실을 최소로 하는 파라미터를 추정치로 하는 방법
                OLS(Ordinary Least Squared)라고 씀.
"""


# p244  4.5.6   최소제곱법과 최대우도법의 관계
"""

"""






























