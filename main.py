from Data_preprocessing import TextDataset
from torch.utils.data import random_split


MAX_LENGTH = 25

dataset = TextDataset('ChatbotData.csv', min_length=3, max_length=MAX_LENGTH)

# 10번째 데이터 임의 추출
x, y = dataset[10]

# 80%의 데이터를 train에 할당합니다.
train_size = int(len(dataset) * 0.8)

# 나머지 20% 데이터를 test에 할당합니다.
test_size = len(dataset) - train_size

# 랜덤 스플릿으로 분할을 완료합니다.
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

from torch.utils.data import DataLoader, SubsetRandomSampler

train_loader = DataLoader(train_dataset,
                          batch_size=16,
                          shuffle=True)

test_loader = DataLoader(test_dataset,
                         batch_size=16,
                         shuffle=True)

# 1개의 배치 데이터를 추출합니다.
x, y = next(iter(train_loader))

# shape: (batch_size, sequence_length)
x.shape, y.shape