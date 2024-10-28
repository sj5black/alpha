# 도전 과제 - 영화 리뷰 감성 분석  

주제  

- 영화 리뷰 데이터를 사용하여 리뷰의 점수를 예측하는 모델을 만드는 프로젝트  

목표  

- Netflix의 영화 리뷰 데이터를 사용하여, 리뷰의 평점을 예측, 긍정과 부정 감정을 분류해보는 것이 목표이다.

과정  

1. 데이터셋 로드 :  
https://www.kaggle.com/datasets/ashishkumarak/netflix-reviews-playstore-daily-updated  
를 통해서 넷플릭스 리뷰 데이터셋을 불러온다. info()를 통해 잘 로드 되었고, 데이터의 상태 확인.  

2. 데이터 전처리 :  
확인 결과 contents의 결측치가 2개 발견. 따라서 해당 row 2줄을 제한 모든 데이터만 활용.  
대소문자나 쓸모없는 단어, 구두점 등 단어 성향 파악에 도움이 안 되는 요소들은 모두 처리.  

3. feature 분석 (EDA) :  
데이터의 특징을 분석해야한다. 넷플릭스 리뷰 데이터에는 리뷰가 1점부터 5점까지 존재. 해당 데이터의 분포를 그래프로 나타내어본다.

4. LSTM 모델 정의 :  
데이터셋과 로더를 활용해 모델에 학습시킬 데이터 정의.  
LSTM 모델 정의한 뒤 파라미터로 모델을 초기화.  

5. 손실 함수 및 옵티마이저 정의한 후 모델 학습 및 평가

6. 평가 후 새로운 리뷰 등록 후 예측.
