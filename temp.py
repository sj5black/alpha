import torch
import torch.nn as nn

# 임베딩 레이어 정의
vocab_size = 100  # 단어가 100개 있는 작은 어휘
embed_dim = 50    # 각 단어를 50차원 벡터로 표현
embedding_layer = nn.Embedding(vocab_size, embed_dim)

# 샘플 단어 인덱스 (3, 7, 15번 단어로 구성된 문장)
sample_input = torch.LongTensor([3, 7, 15])

# 임베딩을 통과
embedded_output = embedding_layer(sample_input)
print(sample_input)
print(embedded_output)