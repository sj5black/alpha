import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

df = pd.read_csv("netflix_reviews.csv")  # 파일 불러오기

# 전처리 함수
def preprocess_text(text):
    if isinstance(text, float):
        return ""
    text = text.lower()  # 대문자를 소문자로
    text = re.sub(r'[^\w\s]', '', text)  # 구두점 제거
    text = re.sub(r'\d+', '', text)  # 숫자 제거
    text = text.strip()  # 띄어쓰기 제외하고 빈 칸 제거
    return text

# content 컬럼에 전처리 함수 적용
df['content'] = df['content'].apply(preprocess_text)

# 필요한 열 선택 및 결측값 처리
# df_ML = df[['content', 'score']].dropna().reset_index(drop=True)

import torch.nn.utils.rnn as rnn_utils

# 데이터셋 클래스 정의
class ReviewDataset(Dataset):
    def __init__(self, reviews, ratings, text_pipeline, label_pipeline):
        self.reviews = reviews
        self.ratings = ratings
        self.text_pipeline = text_pipeline
        self.label_pipeline = label_pipeline

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        review = self.text_pipeline(self.reviews[idx])
        rating = self.label_pipeline(self.ratings[idx])
        return torch.tensor(review), torch.tensor(rating)

# 특성과 타겟 분리
X = df['content']
y = df['score']

# 데이터 분할
train_reviews, test_reviews, train_ratings, test_ratings = train_test_split(X, y, test_size=0.2, random_state=42)

tokenizer = get_tokenizer('basic_english')

# 어휘 생성 함수
def yield_tokens(data_iter):
    for text in data_iter:
        yield tokenizer(text)

# 어휘 생성
vocab = build_vocab_from_iterator(yield_tokens(train_reviews), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])  # 기본 인덱스를 <unk>로 설정

# 텍스트 파이프라인 정의
def text_pipeline(text):
    return torch.tensor([vocab[token] for token in tokenizer(text)])

# 레이블 파이프라인 정의
label_encoder = LabelEncoder()
label_encoder.fit(df['score'].unique())  # 예시 레이블
def label_pipeline(label):
    return label_encoder.transform([label])[0]

# 데이터 로더 정의
BATCH_SIZE = 64

# 데이터셋 정의
train_dataset = ReviewDataset(train_reviews, train_ratings, text_pipeline, label_pipeline)
test_dataset = ReviewDataset(test_reviews, test_ratings, text_pipeline, label_pipeline)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# LSTM 모델 정의
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        output, (hidden, cell) = self.lstm(embedded)
        return self.fc(hidden[-1])

# 하이퍼파라미터 정의
VOCAB_SIZE = len(vocab)
EMBED_DIM = 64
HIDDEN_DIM = 128
OUTPUT_DIM = len(df['score'].unique())  # 예측할 점수 개수

# 모델 초기화
model = LSTMModel(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, OUTPUT_DIM)

# 손실 함수와 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 모델 학습은 직접 작성해보세요!!! (수정 필요)
num_epochs = 10
for epoch in range(num_epochs):
    total_loss = 0
    for reviews, ratings in train_dataloader:
        optimizer.zero_grad()
        outputs = model(reviews)
        loss = criterion(outputs, ratings)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 2 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss/len(train_dataloader):.4f}')

print('Finished Training')

# 예측 함수(예시)
def predict_review(model, review):
    model.eval()
    with torch.no_grad():
        review_tensor = torch.tensor(text_pipeline(review))
        review_tensor = review_tensor.unsqueeze(0)  # 배치 차원 추가
        output = model(review_tensor)
        prediction = output.argmax(1).item()
        return label_encoder.inverse_transform([prediction])[0]

# 새로운 리뷰에 대한 예측
new_review = "This app is great but has some bugs."
predicted_score = predict_review(model, new_review)
print(f'Predicted Score: {predicted_score}')

def predict_review(model, review):
    model.eval()
    with torch.no_grad():
        # text_pipeline을 통해 토큰화하고 인덱스화된 텐서를 생성
        tensor_review = torch.tensor(text_pipeline(review)).unsqueeze(0)  # 배치 차원 추가

        output = model(tensor_review)  # 모델에 입력
        prediction = output.argmax(1).item()
        return label_encoder.inverse_transform([prediction])[0]

# 새로운 리뷰에 대한 예측
new_review = "This app is so amazing!"
predicted_score = predict_review(model, new_review)
print(f'Predicted Score: {predicted_score}')