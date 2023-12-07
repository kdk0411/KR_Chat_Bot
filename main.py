import numpy as np
from Data_preprocessing import TextDataset
from torch.utils.data import random_split
from model import train


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

NUM_EPOCHS = 20
STATEDICT_PATH = 'models/seq2seq-chatbot-kor.pt'

best_loss = np.inf

for epoch in range(NUM_EPOCHS):
  loss = train(model, train_loader, optimizer, loss_fn, device)

  val_loss = evaluate(model, test_loader, loss_fn, device)

  if val_loss < best_loss:
    best_loss = val_loss
    torch.save(model.state_dict(), STATEDICT_PATH)

  if epoch % 5 == 0:
    print(f'epoch: {epoch + 1}, loss: {loss:.4f}, val_loss: {val_loss:.4f}')

  # Early Stop
  es(loss)
  if es.early_stop:
    break

  # Scheduler
  scheduler.step(val_loss)

model.load_state_dict(torch.load(STATEDICT_PATH))
torch.save(model.state_dict(), f'models/seq2seq-chatbot-kor-{best_loss:.4f}.pt')