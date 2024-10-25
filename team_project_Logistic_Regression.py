#1 데이터셋 불러오기. seaborn 라이브러리에 있는 titanic 데이터 불러오기
import seaborn as sns
import pandas as pd
import numpy as np

titanic = sns.load_dataset('titanic')



#2-1 feature 분석. head 함수를 이용해 데이터의 feature를 파악
titanic.head()



#2-2 feature 분석. describe 함수를 통해 기본적인 통계 확인
titanic.describe()



#2-3 feature 분석. isnull() 함수와 sum() 함수를 이용해 각 열의 결측치 갯수 확인
#describe 함수를 통해 확인할 수 있는 count, std, min, 25%, 50%, 70%, max 가 각각 무슨 뜻인지



#2-4 feature 분석. isnull() 함수와 sum() 함수를 이용해 각 열의 결측치 갯수 확인
print(titanic.isnull().sum())



#3-1 결측치 처리. Age(나이)의 결측치는 중앙값으로, Embarked(승선 항구)의 결측치는 최빈값으로 대체, 결과를 isnull() 함수와 sum() 함수를 이용해 확인
titanic['age'].fillna(titanic['age'].median(), inplace=True)
titanic['embarked'].fillna(titanic['embarked'].mode()[0], inplace=True)

print(titanic['age'].isnull().sum())
print(titanic['embarked'].isnull().sum())



#3-2 수치형으로 인코딩. Sex(성별)는 남자:0, 여자:1. Embarked(승선 항구) ‘C’는 0, Q는 1, ‘S’는 2. 결과를 head 함수를 이용해 확인
titanic['sex'] = titanic['sex'].map({'male': 0, 'female': 1})
titanic['alive'] = titanic['alive'].map({'yes': 1, 'no': 0})
titanic['embarked'] = titanic['embarked'].map({'C': 0, 'Q': 1, 'S': 2,})

print(titanic['sex'].head())
print(titanic['alive'].head())
print(titanic['embarked'].head())



#3-3. 새로운 feature 생성. SibSip(타이타닉호에 동승한 자매 및 배우자의 수), Parch(타이타닉호에 동승한 부모 및 자식의 수)를 통해 family_size(가족크기) 생성
titanic['family_size'] = titanic['sibsp'] + titanic['parch'] + 1

print(titanic['family_size'].head())



#4-1 모델 학습 준비
titanic = titanic[['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'family_size']]
X = titanic.drop('survived', axis=1) # feature
y = titanic['survived'] # target



#4-2. Logistic Regression
# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 데이터 스케일링
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 모델 생성 및 학습
model = LogisticRegression()
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 평가
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")