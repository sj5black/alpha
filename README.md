```#1 데이터셋 불러오기
import pandas as pd

df = pd.read_csv("/Users/ur/Desktop/AI부트캠프/02 13조/netflix_reviews.csv")

print(df)



#2 데이터 전처리
# 전처리 함수
import re

def preprocess_text(text):
    if isinstance(text, float):
        return ""
    text = text.lower()  # 대문자를 소문자로
    text = re.sub(r'[^\w\s]', '', text)  # 구두점 제거
    text = re.sub(r'\d+', '', text)  # 숫자 제거
    text = text.strip()  # 띄어쓰기 제외하고 빈 칸 제거
    return text



#3 feature 분석 (EDA)

import numpy as np
import seaborn as sns  # 그래프를 그리기 위한 seaborn 라이브러리 임포트
import matplotlib.pyplot as plt  # 그래프 표시를 위한 pyplot 임포트

#3-1 sns.barplot 사용하기
sns.set_theme(style="whitegrid") # figure, axes 스타일
x = [1, 2, 3, 4, 5] # x축에 들어갈 범주형 변수 = 리뷰컬럼
y = [0,10000, 20000, 30000, 40000] # 각 범주에 해당하는 숫자 = 리뷰갯수
sns.barplot(x=x, y=y) # 바 차트 생성



#3-2 plt.xlabel 사용하기
# plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
plt.xlabel('Score', labelpad=15)
plt.ylabel('Count', labelpad=20)
plt.title('Distribution of Scores')
plt.show()


# ?????????
```
